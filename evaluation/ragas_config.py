
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings


# 1. Connect to local Ollama
ollama_client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


# 2. Configure Qwen as the evaluation LLM
evaluator_llm = llm_factory(
    "qwen2.5:3b",
    provider="openai",
    client=ollama_client
)


# 3. Configure the local embedding model
evaluator_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True,
    batch_size=32
)
