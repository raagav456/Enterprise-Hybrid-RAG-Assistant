
from pathlib import Path
import json
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from reranker import CrossEncoderReranker


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_FILE = (
    BASE_DIR
    / "employee-handbook-rag"
    / "data"
    / "documents"
    / "documents.jsonl"
)

DB_PATH = BASE_DIR / "vector_db"

COLLECTION_NAME = "technova_knowledge"

FINAL_TOP_K = 5
VECTOR_CANDIDATES = 20
BM25_CANDIDATES = 20
RRF_CANDIDATES = 20
RRF_K = 60


# --------------------------------------------------
# 2. Load documents
# --------------------------------------------------

def load_documents():

    documents = []

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as file:

        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    print(f"Documents loaded: {len(documents)}")

    return documents


# --------------------------------------------------
# 3. Tokenization for BM25
# --------------------------------------------------

def tokenize(text):

    return re.findall(r"[a-zA-Z0-9_-]+", text.lower())


def normalize_text(text):

    return " ".join(
        re.findall(r"[a-z0-9]+", text.lower())
    )


# --------------------------------------------------
# 4. Build BM25 index and lookup maps
# --------------------------------------------------

documents = load_documents()

corpus = [
    tokenize(doc["content"])
    for doc in documents
]

bm25 = BM25Okapi(corpus)

document_map = {
    doc["document_id"]: doc
    for doc in documents
}


# Project title lookup map

project_documents = [
    doc for doc in documents
    if doc.get("document_type") == "project_record"
]

project_title_map = {
    normalize_text(doc["title"]): doc
    for doc in project_documents
}


# Employee name lookup map

employee_documents = [
    doc for doc in documents
    if doc.get("document_type") == "employee_profile"
]

employee_name_map = {}

for doc in employee_documents:

    match = re.search(
        r"Employee profile:\s*(.*?)\s*\(",
        doc.get("content", "")
    )

    if match:

        employee_name = normalize_text(match.group(1))

        employee_name_map[employee_name] = doc


# --------------------------------------------------
# 5. Connect to ChromaDB
# --------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    COLLECTION_NAME
)

print(f"ChromaDB collection count: {collection.count()}")


# --------------------------------------------------
# 6. Load Cross-Encoder reranker
# --------------------------------------------------

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)


# --------------------------------------------------
# 7. Vector search
# --------------------------------------------------

def vector_search(query):

    query_embedding = model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=VECTOR_CANDIDATES
    )

    ranked_results = []

    for rank, chroma_id in enumerate(
        results["ids"][0], start=1
    ):

        ranked_results.append({
            "id": chroma_id,
            "rank": rank
        })

    return ranked_results


# --------------------------------------------------
# 8. BM25 keyword search
# --------------------------------------------------

def keyword_search(query):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    ranked_results = []

    for index in ranked_indices[:BM25_CANDIDATES]:

        document = documents[index]

        chroma_id = (
            document["document_id"] + "_chunk_0"
        )

        ranked_results.append({
            "id": chroma_id,
            "rank": len(ranked_results) + 1,
            "score": float(scores[index])
        })

    return ranked_results


# --------------------------------------------------
# 9. Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    vector_results,
    keyword_results
):

    fused_scores = {}
    component_ranks = {}

    for result_list, source in [
        (vector_results, "vector"),
        (keyword_results, "bm25")
    ]:

        for result in result_list:

            doc_id = result["id"]
            rank = result["rank"]

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0)
                + 1 / (RRF_K + rank)
            )

            component_ranks.setdefault(
                doc_id, {}
            )[source] = rank

    sorted_results = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        {
            "id": doc_id,
            "rrf_score": score,
            "ranks": component_ranks[doc_id]
        }
        for doc_id, score in sorted_results[:RRF_CANDIDATES]
    ]


# --------------------------------------------------
# 10. Exact project title search
# --------------------------------------------------

def exact_project_search(query):

    normalized_query = normalize_text(query)

    matches = []

    for normalized_title, document in project_title_map.items():

        if normalized_title in normalized_query:

            source_ids = document.get(
                "source_record_ids", []
            )

            project_id = next(
                (
                    str(source_id)
                    for source_id in source_ids
                    if str(source_id).startswith("PROJ-")
                ),
                None
            )

            matches.append({
                "id": document["document_id"] + "_chunk_0",
                "identifier": project_id,
                "document": document
            })

    if len(matches) == 1:
        return matches

    return []


# --------------------------------------------------
# 11. Exact employee name search
# --------------------------------------------------

def exact_employee_search(query):

    normalized_query = normalize_text(query)

    matches = []

    for employee_name, document in employee_name_map.items():

        if employee_name in normalized_query:

            employee_id_match = re.search(
                r"\bEMP-\d{5}\b",
                document.get("content", "")
            )

            employee_id = (
                employee_id_match.group(0)
                if employee_id_match
                else None
            )

            matches.append({
                "id": document["document_id"] + "_chunk_0",
                "identifier": employee_id,
                "document": document
            })

    if len(matches) == 1:
        return matches

    return []


# --------------------------------------------------
# 12. Exact identifier search
# --------------------------------------------------

def exact_identifier_search(query):

    identifiers = re.findall(
        r"\b[A-Z]{2,}-\d{3,}\b",
        query.upper()
    )

    if not identifiers:
        return []

    matches = []

    for identifier in identifiers:

        for doc in documents:

            document_id = doc["document_id"].upper()

            source_ids = [
                str(source_id).upper()
                for source_id in doc.get("source_record_ids", [])
            ]

            if (
                identifier == document_id
                or identifier in source_ids
            ):

                matches.append({
                    "id": doc["document_id"] + "_chunk_0",
                    "identifier": identifier,
                    "document": doc
                })

    return matches


# --------------------------------------------------
# 13. Hybrid search + Cross-Encoder reranking
# --------------------------------------------------

def hybrid_search(query):

    print("\n" + "=" * 75)
    print(f"QUERY: {query}")
    print("=" * 75)

    # First: Exact project title matching

    project_results = exact_project_search(query)

    if project_results:

        print("\nEXACT PROJECT TITLE MATCH FOUND")

        for position, result in enumerate(
            project_results, start=1
        ):

            document = result["document"]

            print(f"\nRank: {position}")
            print(f"Project ID: {result['identifier']}")
            print(f"Document ID: {result['id']}")
            print(f"Title: {document['title']}")
            print(f"Content: {document['content'][:500]}")

        return project_results

    # Second: Exact employee name matching

    employee_results = exact_employee_search(query)

    if employee_results:

        print("\nEXACT EMPLOYEE NAME MATCH FOUND")

        for position, result in enumerate(
            employee_results, start=1
        ):

            document = result["document"]

            print(f"\nRank: {position}")
            print(f"Employee ID: {result['identifier']}")
            print(f"Document ID: {result['id']}")
            print(f"Content: {document['content'][:500]}")

        return employee_results

    # Third: Exact identifier matching

    exact_results = exact_identifier_search(query)

    if exact_results:

        print("\nEXACT IDENTIFIER MATCH FOUND")

        for position, result in enumerate(
            exact_results, start=1
        ):

            document = result["document"]

            print(f"\nRank: {position}")
            print(f"Identifier: {result['identifier']}")
            print(f"Document ID: {result['id']}")
            print(f"Content: {document['content'][:500]}")

        return exact_results

    # Fourth: Vector search

    vector_results = vector_search(query)

    # Fifth: BM25 keyword search

    keyword_results = keyword_search(query)

    # Sixth: Reciprocal Rank Fusion

    hybrid_candidates = reciprocal_rank_fusion(
        vector_results,
        keyword_results
    )

    # Seventh: Prepare documents for reranking

    candidate_documents = []

    for candidate in hybrid_candidates:

        document_id = candidate["id"].removesuffix(
            "_chunk_0"
        )

        document = document_map.get(document_id)

        if document:

            candidate_document = document.copy()

            candidate_document["retrieval_id"] = candidate["id"]
            candidate_document["rrf_score"] = candidate["rrf_score"]
            candidate_document["component_ranks"] = candidate["ranks"]

            candidate_documents.append(candidate_document)

    if not candidate_documents:

        print("\nNo candidate documents found.")
        return []

    # Eighth: Cross-Encoder reranking

    reranked_results = reranker.rerank(
        query=query,
        documents=candidate_documents,
        top_k=FINAL_TOP_K
    )

    # Ninth: Display final results

    print("\nHYBRID + CROSS-ENCODER RESULTS")
    print("=" * 75)

    for position, result in enumerate(
        reranked_results, start=1
    ):

        print(f"\nRank: {position}")
        print(f"Document ID: {result['document_id']}")
        print(f"Reranker Score: {result['reranker_score']:.4f}")
        print(f"RRF Score: {result['rrf_score']:.5f}")
        print(f"Component Ranks: {result['component_ranks']}")
        print(f"Content: {result['content'][:500]}")

    return reranked_results


# --------------------------------------------------
# 14. Test queries
# --------------------------------------------------

if __name__ == "__main__":

    test_queries = [

        "How many annual leave days do employees receive?",

        "What is the employee ID of Aryan Maharaj?",

        "What is the department of Aryan Maharaj?",

        "What is the approval status of expense EXP-000967?",

        "What is the approved budget for project Security Compliance Upgrade - 052?",

        "What is the approved budget for project Customer Analytics Platform - 077?"

    ]

    for query in test_queries:
        hybrid_search(query)
