
import sys
import json
import asyncio
import argparse

from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

GENERATION_DIR = (
    BASE_DIR / "employee-handbook-rag" / "generation"
)

EVALUATION_FILE = (
    BASE_DIR / "evaluation" / "balanced_evaluation_sample.jsonl"
)

OUTPUT_DIR = BASE_DIR / "evaluation" / "results"

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ragas_evaluation_results.json"

sys.path.insert(0, str(GENERATION_DIR))

from rag_pipeline import ask_question

from ragas_config import (
    evaluator_llm,
    evaluator_embeddings
)

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall
)


# --------------------------------------------------
# 2. Initialize Ragas metrics
# --------------------------------------------------

faithfulness = Faithfulness(llm=evaluator_llm)

answer_relevancy = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

context_precision = ContextPrecisionWithoutReference(
    llm=evaluator_llm
)

context_recall = ContextRecall(llm=evaluator_llm)


# --------------------------------------------------
# 3. Load evaluation questions
# --------------------------------------------------

def load_evaluation_data():

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:

        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


# --------------------------------------------------
# 4. Run RAG and evaluate using Ragas
# --------------------------------------------------

async def evaluate_dataset(limit=None):

    evaluation_data = load_evaluation_data()

    if limit:
        evaluation_data = evaluation_data[:limit]

    results = []

    print("\nTotal questions:", len(evaluation_data))

    for index, item in enumerate(evaluation_data, start=1):

        question = item["question"]
        reference = item["reference_answer"]
        answerable = item["answerable"]

        print("\n" + "=" * 60)
        print(f"Question {index}/{len(evaluation_data)}")
        print(question)

        # Run the existing RAG pipeline
        rag_result = ask_question(question)

        answer = rag_result["answer"]
        contexts = rag_result["contexts"]

        print("RAG status:", rag_result["status"])

        # Default metric values
        scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None
        }

        # Evaluate only when actual retrieved context exists
        if contexts:

            faith_result = await faithfulness.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts
            )

            scores["faithfulness"] = faith_result.value

            precision_result = await context_precision.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts
            )

            scores["context_precision"] = precision_result.value

        # Answer relevancy can also evaluate abstention responses
        relevance_result = await answer_relevancy.ascore(
            user_input=question,
            response=answer
        )

        scores["answer_relevancy"] = relevance_result.value

        # Context recall applies to answerable questions
        if answerable and contexts:

            recall_result = await context_recall.ascore(
                user_input=question,
                retrieved_contexts=contexts,
                reference=reference
            )

            scores["context_recall"] = recall_result.value

        result = {
            "question": question,
            "reference_answer": reference,
            "answerable": answerable,
            "generated_answer": answer,
            "retrieved_contexts": contexts,
            "sources": rag_result["sources"],
            "rag_status": rag_result["status"],
            "scores": scores
        }

        results.append(result)

        print("Scores:", scores)

        # Save progress after every question
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    print("\n========================================")
    print("RAGAS EVALUATION COMPLETED")
    print("========================================")

    print("Questions evaluated:", len(results))
    print("Results saved to:", OUTPUT_FILE)


# --------------------------------------------------
# 5. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to evaluate"
    )

    args = parser.parse_args()

    asyncio.run(evaluate_dataset(args.limit))
