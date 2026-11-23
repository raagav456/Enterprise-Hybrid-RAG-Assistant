
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_FILE = BASE_DIR / "employee-handbook-rag" / "data" / "documents" / "documents.jsonl"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 64


# -----------------------------
# Load documents
# -----------------------------

def load_documents():
    documents = []

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    return documents


# -----------------------------
# Split document into chunks
# -----------------------------

def chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Try to avoid cutting words in half
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


# -----------------------------
# Main ingestion pipeline
# -----------------------------

def main():
    print("\nLoading documents...")

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    collection = client.get_or_create_collection(
        name="technova_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    total_chunks = 0
    batch_ids = []
    batch_texts = []
    batch_metadata = []

    print("\nStarting document chunking and ingestion...")

    for document in tqdm(documents, desc="Processing documents"):
        document_id = document["document_id"]
        content = document.get("content", "")

        chunks = chunk_text(content)

        for index, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{index}"

            metadata = {
                "document_id": document_id,
                "document_type": str(document.get("document_type", "unknown")),
                "title": str(document.get("title", "Untitled")),
                "department": str(document.get("department", "Unknown")),
                "chunk_index": index
            }

            batch_ids.append(chunk_id)
            batch_texts.append(chunk)
            batch_metadata.append(metadata)

            if len(batch_ids) >= BATCH_SIZE:
                embeddings = model.encode(
                    batch_texts,
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadata,
                    embeddings=embeddings.tolist()
                )

                total_chunks += len(batch_ids)

                batch_ids = []
                batch_texts = []
                batch_metadata = []

    # Process remaining records
    if batch_ids:
        embeddings = model.encode(
            batch_texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadata,
            embeddings=embeddings.tolist()
        )

        total_chunks += len(batch_ids)

    print("\n========== INGESTION COMPLETE ==========")
    print("Documents processed:", len(documents))
    print("Chunks processed:", total_chunks)
    print("ChromaDB collection count:", collection.count())
    print("Vector database location:", VECTOR_DB_DIR)


if __name__ == "__main__":
    main()
