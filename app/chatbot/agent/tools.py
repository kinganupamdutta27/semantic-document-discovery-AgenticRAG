"""LangChain tool wrappers for the agentic RAG graph."""

import logging

from langchain.tools import tool

from app.admin.runtime_config import rc
from app.chatbot.exceptions import RetrievalError

logger = logging.getLogger(__name__)

_DEFAULT_K = 10
_DEFAULT_MIN_SCORE = 0.15


@tool
def retrieve_documents(query: str) -> str:
    """Search the knowledge base and return relevant document chunks for a user question.

    Use this tool whenever the user asks a question that may be answered by
    the uploaded documents.  Pass a concise, keyword-rich search query.
    """
    from app.vectorstore.vectorstore import vsm

    if vsm.rag_blocked:
        rebuild_st = vsm.rebuild_status.get("status", "")
        if rebuild_st == "running":
            progress = vsm.rebuild_status.get("progress", 0)
            total = vsm.rebuild_status.get("total", 0)
            return (
                "The knowledge base is currently being rebuilt with a new embedding model "
                f"(progress: {progress}/{total}). Please wait a few minutes and try again."
            )
        return (
            f"The knowledge base index for {vsm.mode} mode has not been built yet. "
            "An admin needs to go to the Settings page (/chat/admin/settings) and click "
            "'Rebuild Vector Store' to create the index before document search can work."
        )

    vs = vsm.active
    if vs is None:
        return (
            f"The knowledge base index for {vsm.mode} mode is not available. "
            "An admin needs to go to the Settings page (/chat/admin/settings) and click "
            "'Rebuild Vector Store' to create the index before document search can work."
        )

    k = rc.get_int("rag_retrieval_k", _DEFAULT_K)
    min_score = _DEFAULT_MIN_SCORE
    try:
        raw_results = vs.similarity_search_with_score(query, k=k)
    except Exception as exc:
        logger.error("Vector-store retrieval failed: %s", exc, exc_info=True)
        raise RetrievalError(f"Vector-store retrieval failed: {exc}") from exc

    if not raw_results:
        return "No relevant documents found in the knowledge base."

    results = [(doc, score) for doc, score in raw_results if score >= min_score]

    if not results:
        logger.info(
            "All %d results below score threshold %.2f for query: %s",
            len(raw_results), min_score, query,
        )
        return "No relevant documents found in the knowledge base."

    logger.info(
        "Retrieved %d documents (k=%d, min_score=%.2f, raw=%d) for query: %s",
        len(results), k, min_score, len(raw_results), query,
    )

    parts: list[str] = []
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata or {}
        parts.append(
            f"[Doc {i}]\n"
            f"Content:\n{doc.page_content}\n"
            f"Source:\n"
            f"  file_name: {meta.get('file_name', 'n/a')}\n"
            f"  page_number: {meta.get('page_number', 'n/a')}"
        )
    return "\n\n".join(parts)
