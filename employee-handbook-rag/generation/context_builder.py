
def build_context(
    results,
    max_context_chars=12000,
    max_documents=3,
    score_margin=3.0
):
    """
    Build an LLM context using the most relevant retrieved documents.

    Supports:
    1. Normal hybrid-search results containing reranker_score.
    2. Exact identifier results containing a nested document.

    Documents are selected using a relative Cross-Encoder score margin.
    This is a heuristic, not a calibrated probability.
    """

    if not results:
        return "", []

    # --------------------------------------------------
    # 1. Identify exact identifier matches
    # --------------------------------------------------

    first_result = results[0]

    if "identifier" in first_result and "document" in first_result:
        selected_results = [first_result]

    else:
        # --------------------------------------------------
        # 2. Select documents using reranker scores
        # --------------------------------------------------

        scored_results = []

        for result in results:
            score = result.get("reranker_score")

            if score is None:
                continue

            scored_results.append((score, result))

        if not scored_results:
            return "", []

        # Highest relevance score first
        scored_results.sort(
            key=lambda item: item[0],
            reverse=True
        )

        top_score = scored_results[0][0]

        # Keep documents sufficiently close to the top score
        minimum_score = top_score - score_margin

        selected_results = []

        for score, result in scored_results:

            if score < minimum_score:
                continue

            selected_results.append(result)

            if len(selected_results) >= max_documents:
                break

    # --------------------------------------------------
    # 3. Prepare context
    # --------------------------------------------------

    context_parts = []
    source_ids = []
    seen_ids = set()

    remaining_chars = max_context_chars

    for result in selected_results:

        # Exact identifier results have nested documents
        if "identifier" in result and "document" in result:
            document = result["document"]
        else:
            document = result

        document_id = document.get("document_id")
        content = document.get("content", "")

        if not document_id or not content:
            continue

        if document_id in seen_ids:
            continue

        header = f"Source: {document_id}\n"
        separator = "\n\n"

        available_chars = (
            remaining_chars
            - len(header)
            - len("Content: ")
            - len(separator)
        )

        if available_chars <= 0:
            break

        content = content[:available_chars]

        formatted_document = (
            header
            + "Content: "
            + content
        )

        context_parts.append(formatted_document)
        source_ids.append(document_id)

        seen_ids.add(document_id)

        remaining_chars -= (
            len(formatted_document)
            + len(separator)
        )

        if remaining_chars <= 0:
            break

    context = "\n\n".join(context_parts)

    return context, source_ids
