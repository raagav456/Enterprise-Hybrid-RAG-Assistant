
# --------------------------------------------------
# LLM Answer Generator using Ollama
# --------------------------------------------------

import ollama

from prompt_template import create_prompt


# Your locally downloaded model
MODEL_NAME = "qwen2.5:3b"


def generate_answer(question, context):
    """
    Generate an answer using the local Qwen model.

    Args:
        question: User's question
        context: Retrieved document context

    Returns:
        Generated answer
    """

    if not context.strip():
        return (
            "I could not find sufficient information in the "
            "available knowledge base to answer this question."
        )

    # Create the RAG prompt
    prompt = create_prompt(
        question=question,
        context=context
    )

    # Call local Ollama model
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    answer = response["message"]["content"]

    return answer.strip()
