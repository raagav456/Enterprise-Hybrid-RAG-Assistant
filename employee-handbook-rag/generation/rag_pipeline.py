
import sys
import importlib.util
from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
GENERATION_DIR = CURRENT_DIR

HYBRID_FILE = RETRIEVAL_DIR / "hybrid_search.py"

print("Project root:", PROJECT_ROOT)
print("Retrieval directory:", RETRIEVAL_DIR)
print("Retrieval file exists:", HYBRID_FILE.exists())

# --------------------------------------------------
# 2. Add local directories to Python path
# --------------------------------------------------

sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(GENERATION_DIR))

# --------------------------------------------------
# 3. Import local hybrid_search.py directly
# --------------------------------------------------

if not HYBRID_FILE.exists():
    raise FileNotFoundError(
        f"Local hybrid_search.py not found: {HYBRID_FILE}"
    )

spec = importlib.util.spec_from_file_location(
    "technova_hybrid_search",
    HYBRID_FILE
)

hybrid_module = importlib.util.module_from_spec(spec)
sys.modules["technova_hybrid_search"] = hybrid_module

spec.loader.exec_module(hybrid_module)

run_hybrid_search = hybrid_module.hybrid_search

print("Hybrid search imported from:")
print(HYBRID_FILE)

# --------------------------------------------------
# 4. Import generation components
# --------------------------------------------------

from context_builder import build_context
from answer_generator import generate_answer

# --------------------------------------------------
# 5. Configuration
# --------------------------------------------------

# Experimental threshold.
# Validate using an independent evaluation dataset.

RERANKER_THRESHOLD = -5.3537

ABSTENTION_MESSAGE = (
    "I could not find sufficient information in the "
    "available knowledge base to answer this question."
)

# --------------------------------------------------
# 6. Identify exact identifier results
# --------------------------------------------------

def is_exact_identifier_result(results):

    if not results:
        return False

    first = results[0]

    return (
        "identifier" in first
        and "document" in first
    )


# --------------------------------------------------
# 7. Identify generated abstention responses
# --------------------------------------------------

def is_abstention_answer(answer):

    if not answer or not answer.strip():
        return True

    normalized_answer = answer.strip().casefold()
    normalized_message = ABSTENTION_MESSAGE.casefold()

    return normalized_answer.startswith(normalized_message)


# --------------------------------------------------
# 8. Main RAG pipeline
# --------------------------------------------------

def ask_question(question):

    print("\nProcessing your question...")
    print("--------------------------------")

    # Default evaluation data.
    # These values are updated as the pipeline progresses.

    context = ""
    source_ids = []

    # ----------------------------------------------
    # Stage 1: Hybrid retrieval + reranking
    # ----------------------------------------------

    results = run_hybrid_search(question)

    if not results:
        return {
            "question": question,
            "answer": ABSTENTION_MESSAGE,
            "contexts": [],
            "sources": [],
            "status": "abstained"
        }

    exact_match = is_exact_identifier_result(results)

    # ----------------------------------------------
    # Stage 2: Apply abstention threshold
    # ----------------------------------------------

    if exact_match:

        print("Exact identifier match found.")

    else:

        top_score = results[0].get("reranker_score")

        if top_score is None:
            return {
                "question": question,
                "answer": ABSTENTION_MESSAGE,
                "contexts": [],
                "sources": [],
                "status": "abstained"
            }

        print(
            "Top Cross-Encoder score:",
            round(top_score, 4)
        )

        if top_score < RERANKER_THRESHOLD:

            print("Retrieval confidence below threshold.")

            return {
                "question": question,
                "answer": ABSTENTION_MESSAGE,
                "contexts": [],
                "sources": [],
                "status": "abstained"
            }

    # ----------------------------------------------
    # Stage 3: Build context
    # ----------------------------------------------

    context, source_ids = build_context(results)

    if not context.strip():

        return {
            "question": question,
            "answer": ABSTENTION_MESSAGE,
            "contexts": [],
            "sources": [],
            "status": "abstained"
        }

    # Ragas expects contexts as a list of text strings.
    # Our context builder returns one combined context string.

    contexts = [context]

    print("Context prepared.")
    print("Sources:", source_ids)

    # ----------------------------------------------
    # Stage 4: Generate answer using Qwen
    # ----------------------------------------------

    print("Generating answer using Qwen...")

    answer = generate_answer(
        question=question,
        context=context
    )

    # ----------------------------------------------
    # Stage 5: Determine final response status
    # ----------------------------------------------

    if is_abstention_answer(answer):

        print("Generation abstained.")

        return {
            "question": question,
            "answer": ABSTENTION_MESSAGE,
            "contexts": contexts,
            "sources": source_ids,
            "status": "abstained"
        }

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "sources": source_ids,
        "status": "answered"
    }


# --------------------------------------------------
# 9. Interactive application
# --------------------------------------------------

if __name__ == "__main__":

    print("\n======================================")
    print("TECHNOVA ENTERPRISE KNOWLEDGE ASSISTANT")
    print("======================================")

    print("Model: Qwen 2.5 3B")
    print("Runtime: Ollama")
    print("Retrieval: Hybrid + Cross-Encoder")

    print("\nType 'exit' to stop.")

    while True:

        question = input("\nEnter your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Exiting assistant.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            result = ask_question(question)

            print("\n======================================")
            print("FINAL ANSWER")
            print("======================================")

            print(result["answer"])

            if result["sources"]:

                print("\nSources:")

                for source in result["sources"]:
                    print("-", source)

            print("\nStatus:", result["status"])

        except Exception as error:

            print("\nAn error occurred:")
            print(error)
