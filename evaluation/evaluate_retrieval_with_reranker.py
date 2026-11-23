from pathlib import Path
import json
import re
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_FILE = (
    BASE_DIR
    / "employee-handbook-rag"
    / "data"
    / "documents"
    / "documents.jsonl"
)

EVALUATION_FILE = (
    BASE_DIR
    / "employee-handbook-rag"
    / "data"
    / "evaluation"
    / "evaluation.jsonl"
)

DB_PATH = BASE_DIR / "vector_db"
COLLECTION_NAME = "technova_knowledge"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 5
CANDIDATES = 20
RRF_K = 60

# CPU-friendly cross-encoder settings.
RERANK_BATCH_SIZE = 16

# Set to an integer such as 50 for a quick smoke test.
# Keep as None to evaluate the complete answerable set.
MAX_RERANK_QUESTIONS = None


# --------------------------------------------------
# 2. Load documents and evaluation data
# --------------------------------------------------

def load_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


documents = load_jsonl(DOCUMENTS_FILE)
evaluation_data = load_jsonl(EVALUATION_FILE)

print(f"Documents loaded: {len(documents)}")
print(f"Evaluation questions: {len(evaluation_data)}")

document_map = {
    doc["document_id"]: doc
    for doc in documents
}


# --------------------------------------------------
# 3. Build BM25 index
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_-]+", text.lower())


corpus = [
    tokenize(doc["content"])
    for doc in documents
]

bm25 = BM25Okapi(corpus)


# --------------------------------------------------
# 4. Load embedding model and ChromaDB
# --------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device="cpu"
)

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    COLLECTION_NAME
)

print(f"ChromaDB count: {collection.count()}")


# --------------------------------------------------
# 5. Prepare evaluation questions
# --------------------------------------------------

answerable_data = [
    item for item in evaluation_data
    if item["answerable"]
]

unanswerable_data = [
    item for item in evaluation_data
    if not item["answerable"]
]

print(f"Answerable questions: {len(answerable_data)}")
print(f"Unanswerable questions: {len(unanswerable_data)}")

# Reranking is evaluated on answerable questions only.
# This keeps the comparison focused and avoids spending
# CPU time reranking questions that have no relevant document.
rerank_data = answerable_data

if MAX_RERANK_QUESTIONS is not None:
    rerank_data = rerank_data[:MAX_RERANK_QUESTIONS]
    print(
        "\nSMOKE TEST MODE: reranking only "
        f"{len(rerank_data)} answerable questions."
    )

questions = [
    item["question"]
    for item in rerank_data
]


# --------------------------------------------------
# 6. Generate query embeddings
# --------------------------------------------------

print("\nGenerating query embeddings...")

query_embeddings = embedding_model.encode(
    questions,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)


# --------------------------------------------------
# 7. Retrieval methods
# --------------------------------------------------

def normalize_id(document_id):
    return document_id.removesuffix("_chunk_0")


def vector_search_batch(embeddings):
    all_results = []

    # Query Chroma in small batches to limit memory use.
    batch_size = 32

    for start in range(0, len(embeddings), batch_size):
        batch = embeddings[start:start + batch_size]

        results = collection.query(
            query_embeddings=batch.tolist(),
            n_results=CANDIDATES
        )

        all_results.extend([
            [normalize_id(doc_id) for doc_id in ids]
            for ids in results["ids"]
        ])

    return all_results


def bm25_search(query):
    scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    return [
        documents[index]["document_id"]
        for index in ranked_indices[:CANDIDATES]
    ]


def hybrid_search(vector_ids, bm25_ids):
    fused_scores = defaultdict(float)

    for ranked_list in [vector_ids, bm25_ids]:
        for rank, doc_id in enumerate(ranked_list, start=1):
            fused_scores[doc_id] += 1 / (RRF_K + rank)

    ranked_ids = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        doc_id
        for doc_id, _score in ranked_ids[:CANDIDATES]
    ]


# --------------------------------------------------
# 8. Cross-encoder reranking
# --------------------------------------------------

print("\nLoading cross-encoder reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL_NAME,
    device="cpu"
)

print("Cross-encoder loaded successfully!")


def rerank_one_query(query, candidate_ids):
    if not candidate_ids:
        return []

    valid_ids = [
        doc_id for doc_id in candidate_ids
        if doc_id in document_map
    ]

    if not valid_ids:
        return []

    pairs = [
        (query, document_map[doc_id]["content"])
        for doc_id in valid_ids
    ]

    scores = reranker.predict(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        show_progress_bar=False
    )

    scored_results = list(zip(valid_ids, scores))

    scored_results.sort(
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        doc_id
        for doc_id, _score in scored_results
    ]


def rerank_all_queries(queries, hybrid_results):
    reranked_results = []

    print("\nReranking hybrid candidates...")

    for index, (query, candidate_ids) in enumerate(
        zip(queries, hybrid_results),
        start=1
    ):
        reranked_ids = rerank_one_query(
            query,
            candidate_ids
        )

        reranked_results.append(reranked_ids)

        if index % 50 == 0 or index == len(queries):
            print(
                f"Reranked {index}/{len(queries)} questions"
            )

    return reranked_results


# --------------------------------------------------
# 9. Calculate evaluation metrics
# --------------------------------------------------

def calculate_metrics(results, evaluation_items):
    if not evaluation_items:
        return {
            "Recall@5": 0.0,
            "Precision@5": 0.0,
            "MRR": 0.0,
            "Hit Rate@5": 0.0
        }

    total_relevant = 0
    total_retrieved_relevant = 0
    reciprocal_rank_sum = 0.0
    hit_count = 0
    precision_sum = 0.0

    for item, retrieved_ids in zip(
        evaluation_items,
        results
    ):
        relevant_ids = set(
            item["relevant_document_ids"]
        )

        retrieved_top_k = retrieved_ids[:TOP_K]

        relevant_retrieved = [
            doc_id for doc_id in retrieved_top_k
            if doc_id in relevant_ids
        ]

        total_relevant += len(relevant_ids)
        total_retrieved_relevant += len(relevant_retrieved)

        if relevant_retrieved:
            hit_count += 1

        precision_sum += len(relevant_retrieved) / TOP_K

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                reciprocal_rank_sum += 1 / rank
                break

    number_of_questions = len(evaluation_items)

    return {
        "Recall@5": (
            total_retrieved_relevant / total_relevant
            if total_relevant else 0.0
        ),
        "Precision@5": precision_sum / number_of_questions,
        "MRR": reciprocal_rank_sum / number_of_questions,
        "Hit Rate@5": hit_count / number_of_questions
    }


# --------------------------------------------------
# 10. Run retrieval evaluation
# --------------------------------------------------

print("\nRunning vector search...")

vector_results = vector_search_batch(query_embeddings)

print("\nRunning BM25 search...")

bm25_results = [
    bm25_search(question)
    for question in questions
]

print("\nRunning hybrid RRF search...")

hybrid_results = [
    hybrid_search(vector_ids, keyword_ids)
    for vector_ids, keyword_ids in zip(
        vector_results,
        bm25_results
    )
]

print("\nRunning hybrid + cross-encoder reranking...")

reranked_results = rerank_all_queries(
    questions,
    hybrid_results
)


# --------------------------------------------------
# 11. Calculate metrics for all four methods
# --------------------------------------------------

vector_metrics = calculate_metrics(
    vector_results,
    rerank_data
)

bm25_metrics = calculate_metrics(
    bm25_results,
    rerank_data
)

hybrid_metrics = calculate_metrics(
    hybrid_results,
    rerank_data
)

reranked_metrics = calculate_metrics(
    reranked_results,
    rerank_data
)

all_metrics = {
    "vector": vector_metrics,
    "bm25": bm25_metrics,
    "hybrid_rrf": hybrid_metrics,
    "hybrid_rrf_cross_encoder": reranked_metrics
}


# --------------------------------------------------
# 12. Display comparison
# --------------------------------------------------

print("\n" + "=" * 105)
print("RAG RETRIEVAL EVALUATION RESULTS")
print("=" * 105)

print(f"Total dataset questions: {len(evaluation_data)}")
print(f"Answerable questions in this run: {len(rerank_data)}")
print(f"Unanswerable questions in dataset: {len(unanswerable_data)}")
print("\nMetrics are calculated on the answerable questions evaluated.")

headers = [
    "Metric",
    "Vector",
    "BM25",
    "Hybrid RRF",
    "Hybrid + Reranker"
]

print(
    f"\n{headers[0]:<18}"
    f"{headers[1]:<15}"
    f"{headers[2]:<15}"
    f"{headers[3]:<18}"
    f"{headers[4]:<20}"
)

print("-" * 86)

for metric in vector_metrics:
    print(
        f"{metric:<18}"
        f"{vector_metrics[metric]:<15.4f}"
        f"{bm25_metrics[metric]:<15.4f}"
        f"{hybrid_metrics[metric]:<18.4f}"
        f"{reranked_metrics[metric]:<20.4f}"
    )


# --------------------------------------------------
# 13. Breakdown by question type and difficulty
# --------------------------------------------------

def evaluate_by_group(group_field, evaluation_items, result_sets):
    print(f"\n\nPERFORMANCE BY {group_field.upper()}")
    print("=" * 105)

    groups = sorted({
        item.get(group_field, "unknown")
        for item in evaluation_items
    })

    print(
        f"{'Category':<20}"
        f"{'Questions':<12}"
        f"{'Vector R@5':<15}"
        f"{'BM25 R@5':<15}"
        f"{'Hybrid R@5':<15}"
        f"{'Reranked R@5':<15}"
    )

    print("-" * 92)

    for group in groups:
        indices = [
            index
            for index, item in enumerate(evaluation_items)
            if item.get(group_field, "unknown") == group
        ]

        group_data = [
            evaluation_items[index]
            for index in indices
        ]

        group_scores = []

        for method_results in result_sets:
            group_results = [
                method_results[index]
                for index in indices
            ]

            metrics = calculate_metrics(
                group_results,
                group_data
            )

            group_scores.append(metrics["Recall@5"])

        print(
            f"{str(group):<20}"
            f"{len(indices):<12}"
            f"{group_scores[0]:<15.4f}"
            f"{group_scores[1]:<15.4f}"
            f"{group_scores[2]:<15.4f}"
            f"{group_scores[3]:<15.4f}"
        )


result_sets = [
    vector_results,
    bm25_results,
    hybrid_results,
    reranked_results
]

evaluate_by_group(
    "question_type",
    rerank_data,
    result_sets
)

evaluate_by_group(
    "difficulty",
    rerank_data,
    result_sets
)


# --------------------------------------------------
# 14. Save evaluation results
# --------------------------------------------------

output = {
    "total_dataset_questions": len(evaluation_data),
    "answerable_questions_in_run": len(rerank_data),
    "unanswerable_questions_in_dataset": len(unanswerable_data),
    "reranker_model": RERANKER_MODEL_NAME,
    "candidate_count": CANDIDATES,
    "final_top_k": TOP_K,
    "metrics": all_metrics,
    "evaluation_scope": (
        "Complete answerable set"
        if len(rerank_data) == len(answerable_data)
        else "Smoke test subset of answerable questions"
    )
}

output_file = BASE_DIR / "evaluation_results_with_reranker.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(
        output,
        file,
        indent=4
    )

print(f"\nResults saved to: {output_file}")
print("\nEvaluation completed!")
