# Enterprise Knowledge Assistant — Hybrid RAG System

An end-to-end Retrieval-Augmented Generation (RAG) project that answers enterprise knowledge questions using a synthetic company dataset.

The system combines semantic search, keyword retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and a locally hosted Large Language Model (LLM) to generate context-grounded answers.

## 1. Project Overview

This project simulates an enterprise knowledge assistant for a fictional company, TechNova Solutions.

Users can ask questions about:
- Employee profiles and departments
- Projects and budgets
- Vendors and payment terms
- Expenses
- IT support tickets
- Company policies

The assistant retrieves relevant documents, constructs context, and generates answers using only the retrieved evidence. It can also abstain when sufficient evidence is unavailable.

## 2. Architecture

```text
                 User Question
                       |
                       v
                Hybrid Retrieval
                 /           \
                v             v
          Vector Search    BM25 Search
                \             /
                 \           /
                       v
                Reciprocal Rank
                   Fusion
                       |
                       v
              Cross-Encoder
                 Reranking
                       |
                       v
                Context Builder
                       |
                       v
                Prompt Template
                       |
                       v
              Ollama - Qwen 2.5
                       |
                       v
              Generated Answer
                       |
                       v
              Answer / Abstention
```

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Vector database | ChromaDB |
| Embedding model | all-MiniLM-L6-v2 |
| Keyword retrieval | BM25 |
| Hybrid ranking | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Local LLM | Qwen 2.5 3B |
| LLM runtime | Ollama |
| Evaluation | Custom Python scripts |

## 4. Dataset

The project uses a synthetic enterprise dataset for TechNova Solutions.

- 10 departments
- 1,000 employee records
- 300 projects
- 300 vendors
- 3,000 expenses
- 1,500 IT tickets
- 10 policy documents
- 5,000 project update documents

**Total: 11,110 documents**

The evaluation dataset contains 1,525 examples, including answerable and unanswerable questions.

## 5. Key Features

### Hybrid Retrieval
Combines semantic vector search and BM25 keyword search to improve document retrieval.

### Reciprocal Rank Fusion
Combines rankings from multiple retrieval methods into a unified ranking.

### Cross-Encoder Reranking
Uses a cross-encoder model to improve the relevance ordering of retrieved documents.

### Context Building
Selects relevant documents and constructs a bounded context for the language model.

### Grounded Answer Generation
Uses a local Qwen model through Ollama. The prompt instructs the model to answer using retrieved evidence only.

### Abstention
The system can abstain when retrieved evidence is insufficient.

### Evaluation
Includes separate retrieval, generation, threshold, and faithfulness evaluation scripts.

## 6. Retrieval Evaluation Results

Evaluation on 1,515 answerable questions:

| Retrieval Method | Recall@5 | Precision@5 |
|---|---:|---:|
| Vector Search | 47.85% | 9.57% |
| BM25 | 84.49% | 16.90% |
| Hybrid RRF | 94.59% | 18.92% |
| Hybrid + Cross-Encoder | 97.62% | 19.52% |

The cross-encoder improved Recall@5 by 3.03 percentage points over hybrid RRF.

MRR improved from 0.7423 to 0.9459 (MRR@20).

## 7. Generation Evaluation Results

Evaluation on a balanced sample of 85 questions:

| Metric | Result |
|---|---:|
| Answerability accuracy | 100% |
| Reference answer containment | 93.33% |
| Token F1 | 35.44% |
| Source precision | 92.89% |
| Source recall | 98.67% |
| Unanswerable abstention accuracy | 100% |
| Citation ID validity | 100% |

Citation ID validity was measured on 21 answers with explicit citations. It checks source-ID matching, not claim-level evidence support.

## 8. Faithfulness Evaluation

An exploratory LLM-as-judge evaluation was performed using Qwen 2.5 3B on 10 answerable questions.

- Initial judge result: 9/10 faithful.
- One false negative was identified through manual source verification.
- After manual verification, all 10 answers in this sample were supported by their source documents.

This small sample is not a statistically reliable overall faithfulness benchmark. The same local model was used for generation and judging, so evaluator bias is possible.

## 9. Project Structure

```text
RAG Project/
│
├── employee-handbook-rag/
│   ├── data/
│   │   ├── source/
│   │   ├── documents/
│   │   └── evaluation/
│   │
│   └── generation/
│       ├── context_builder.py
│       ├── prompt_template.py
│       ├── answer_generator.py
│       └── rag_pipeline.py
│
├── ingestion/
│   └── ingest.py
│
├── retrieval/
│   ├── hybrid_search.py
│   ├── reranker.py
│   └── test_retrieval.py
│
├── evaluation/
│   ├── evaluate_retrieval.py
│   ├── evaluate_generation.py
│   ├── evaluate_threshold.py
│   ├── evaluate_unanswerable.py
│   └── evaluate_faithfulness.py
│
└── vector_db/
```

## 10. Local Setup

### Prerequisites

- Python
- Ollama
- Git
- Sufficient memory to load the embedding and reranking models

### Install Ollama model

```bash
ollama pull qwen2.5:3b
```

### Install Python dependencies

```bash
pip install chromadb sentence-transformers rank-bm25 ollama tqdm
```

### Run Ollama

Ensure the Ollama service is running before executing the generation pipeline.

### Data and indexing

The project includes ingestion and evaluation scripts. The vector database is built from the synthetic document dataset.

Refer to the corresponding Python scripts for the existing ingestion and execution workflow.

## 11. Limitations

- The dataset is synthetic and does not represent real enterprise records.
- The benchmark contains many direct factual lookup questions.
- Policy and paraphrased question categories are relatively small.
- The threshold was selected and evaluated on the same small evaluation set, so its results are optimistic.
- The faithfulness evaluation uses an LLM judge and requires further independent validation.
- Retrieval and generation metrics should not be interpreted as production guarantees.

## 12. Future Improvements

- Expand multi-hop and adversarial evaluation questions.
- Use a separate model or human reviewers for faithfulness assessment.
- Evaluate claim-level citation support.
- Test retrieval and abstention thresholds on a separate validation set.
- Add a user interface and conversation history.

## 13. Project Objective

The objective of this project is to demonstrate practical understanding of enterprise RAG architecture, hybrid information retrieval, reranking, grounded LLM generation, abstention, and systematic evaluation using locally hosted models.

**Built with Python, ChromaDB, Sentence Transformers, BM25, and Ollama.**