
import json
from pathlib import Path
from collections import Counter

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_FILE = BASE_DIR / "data" / "documents" / "documents.jsonl"
EVALUATION_FILE = BASE_DIR / "data" / "evaluation" / "evaluation.jsonl"

def load_jsonl(file_path):
    """Read a JSONL file and return its records."""
    records = []

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                )

    return records


def main():
    print("\n===== DATASET VALIDATION =====\n")

    # 1. Load documents and evaluation questions
    documents = load_jsonl(DOCUMENTS_FILE)
    evaluation = load_jsonl(EVALUATION_FILE)

    print(f"Documents loaded: {len(documents)}")
    print(f"Evaluation questions loaded: {len(evaluation)}")

    # 2. Check document IDs
    document_ids = [
        doc.get("document_id")
        for doc in documents
        if doc.get("document_id")
    ]

    missing_document_ids = len(documents) - len(document_ids)
    duplicate_ids = [
        doc_id
        for doc_id, count in Counter(document_ids).items()
        if count > 1
    ]

    print("\n--- Document ID checks ---")
    print(f"Documents without IDs: {missing_document_ids}")
    print(f"Duplicate document IDs: {len(duplicate_ids)}")

    if duplicate_ids:
        print("Examples:", duplicate_ids[:5])

    # 3. Check document content
    missing_content = [
        doc.get("document_id", "UNKNOWN")
        for doc in documents
        if not str(doc.get("content", "")).strip()
    ]

    print("\n--- Content checks ---")
    print(f"Documents with missing content: {len(missing_content)}")

    # 4. Check evaluation references
    valid_document_ids = set(document_ids)
    invalid_references = []

    for index, question in enumerate(evaluation, start=1):
        relevant_ids = question.get("relevant_document_ids", [])

        if not isinstance(relevant_ids, list):
            invalid_references.append(
                f"Question {index}: relevant_document_ids is not a list"
            )
            continue

        for doc_id in relevant_ids:
            if doc_id not in valid_document_ids:
                invalid_references.append(
                    f"Question {index}: unknown document ID {doc_id}"
                )

    print("\n--- Evaluation reference checks ---")
    print(f"Invalid document references: {len(invalid_references)}")

    for error in invalid_references[:10]:
        print(error)

    # 5. Display sample records
    print("\n--- Sample document ---")
    if documents:
        print(json.dumps(documents[0], indent=2, ensure_ascii=False)[:1500])

    print("\n--- Sample evaluation question ---")
    if evaluation:
        print(json.dumps(evaluation[0], indent=2, ensure_ascii=False)[:1500])

    # 6. Final result
    passed = (
        missing_document_ids == 0
        and len(duplicate_ids) == 0
        and len(missing_content) == 0
        and len(invalid_references) == 0
    )

    print("\n===== VALIDATION RESULT =====")
    print("PASSED" if passed else "CHECK REQUIRED")


if __name__ == "__main__":
    main()
