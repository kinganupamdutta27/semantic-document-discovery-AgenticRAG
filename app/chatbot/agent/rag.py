"""Agentic RAG graph with conversation summarisation.

Flow
----
START
  --(should_continue: classifies greeting vs RAG question)-->

  GREETING PATH:
    chat_node  (no tools, uses summary context)
      -> should_summarize
           |-- messages > threshold -> summary_node -> END
           |-- else                 -> END

  RAG PATH:
    rag_chat_node  (tools bound, uses summary context)
      --(route_after_rag)-->
           |-- tool call -> retrieve -> grade_documents
           |       |-- relevant  -> generate_answer -> should_summarize -> …
           |       |-- not relevant & retries left  -> rewrite_question -> rag_chat_node
           |       |-- not relevant & max retries   -> generate_answer -> …
           |-- no tool call -> should_summarize -> …

The ``should_continue`` classifier uses the ``flow_decision`` prompt to
decide whether the user's message is casual chat or a document question.
This prevents the LLM from eagerly invoking the retrieval tool for simple
greetings.

The ``summary_node`` condenses the conversation into a compact summary,
trims old messages (keeping only the most recent pair), and stores the
summary in state.  On subsequent turns both ``chat_node`` and
``rag_chat_node`` inject the summary into the system prompt so the LLM
retains long-term context without an ever-growing message list.
"""

import logging
import operator
import re
from typing import Annotated, Literal

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import RemoveMessage
from langgraph.prebuilt import ToolNode

from app.chatbot.agent.llm import get_model
from app.chatbot.agent.memory import recall_memories
from app.chatbot.agent.tools import retrieve_documents
from app.chatbot.config.prompts import get_prompt
from app.chatbot.exceptions import (
    GenerationError,
    LLMConnectionError,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgenticRAGState(MessagesState):
    """Extends MessagesState with a conversation summary and retry counter."""

    summary: str
    retry_count: Annotated[int, operator.add]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_tools = [retrieve_documents]

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def chat_node(state: AgenticRAGState) -> dict:
    """Normal conversation node for greetings and casual chat (no tools).

    Injects the running conversation summary and long-term memories (if any)
    into the system prompt for full context awareness.
    """
    summary = _content_to_str(state.get("summary", ""))
    question = _extract_latest_user_question(state["messages"])
    system_content = get_prompt("system")
    if summary:
        system_content += "\n\nSummary of the earlier conversation:\n" + summary

    memories = await recall_memories(question)
    if memories:
        system_content += "\n\n" + memories

    system_msg = SystemMessage(content=system_content)
    messages = _sanitize_messages([system_msg] + list(state["messages"]))

    try:
        response = await get_model().ainvoke(messages)
    except Exception as exc:
        logger.error("LLM invocation failed in chat_node: %s", exc, exc_info=True)
        raise LLMConnectionError(f"LLM invocation failed: {exc}") from exc

    return {"messages": [response]}


async def rag_chat_node(state: AgenticRAGState) -> dict:
    """RAG conversation node with tools bound for document retrieval.

    The LLM sees the full conversation history (or summary + recent
    messages + long-term memories) and decides how to formulate the
    search query.
    """
    summary = _content_to_str(state.get("summary", ""))
    question = _extract_latest_user_question(state["messages"])
    system_content = get_prompt("system")
    if summary:
        system_content += "\n\nSummary of the earlier conversation:\n" + summary

    memories = await recall_memories(question)
    if memories:
        system_content += "\n\n" + memories

    system_msg = SystemMessage(content=system_content)
    messages = _sanitize_messages([system_msg] + list(state["messages"]))

    try:
        response = await get_model().bind_tools(_tools).ainvoke(messages)
    except Exception as exc:
        logger.error("LLM invocation failed in rag_chat_node: %s", exc, exc_info=True)
        raise LLMConnectionError(f"LLM invocation failed: {exc}") from exc

    return {"messages": [response]}


async def generate_answer(state: AgenticRAGState) -> dict:
    """Generate the final answer using retrieved context."""
    messages = state["messages"]
    question = _extract_latest_user_question(messages)
    context = _extract_tool_response(messages)

    system_msg = SystemMessage(content=get_prompt("generate_answer"))
    context_msg = SystemMessage(
        content=(
            "Use the following retrieved documents as your sole factual basis:\n\n"
            f"{context}"
        )
    )
    user_msg = HumanMessage(content=question)

    try:
        response = await get_model().ainvoke([system_msg, context_msg, user_msg])
    except Exception as exc:
        logger.error("LLM invocation failed in generate_answer: %s", exc, exc_info=True)
        raise GenerationError(f"Answer generation failed: {exc}") from exc

    return {"messages": [response]}


async def rewrite_question(state: AgenticRAGState) -> dict:
    """Rewrite the original question for better retrieval."""
    messages = state["messages"]
    question = _extract_latest_user_question(messages)

    prompt_template = get_prompt("rewrite_question")
    prompt_text = prompt_template.replace("{question}", question)

    try:
        response = await get_model().ainvoke(
            [HumanMessage(content=prompt_text)]
        )
    except Exception as exc:
        logger.error("LLM invocation failed in rewrite_question: %s", exc, exc_info=True)
        raise LLMConnectionError(f"Query rewrite failed: {exc}") from exc

    logger.info("Rewrote question: %s -> %s", question, response.content)
    return {
        "messages": [HumanMessage(content=response.content)],
        "retry_count": 1,
    }


async def summary_node(state: AgenticRAGState) -> dict:
    """Summarise the conversation and trim older messages.

    Keeps the two most recent messages (typically the last user question
    and the assistant answer) and replaces everything older with a
    compact summary stored in ``state["summary"]``.
    """
    summary = _content_to_str(state.get("summary", ""))

    if summary:
        prompt_template = get_prompt("extend_summary")
        summary_prompt = SystemMessage(
            content=prompt_template.replace("{summary}", summary),
        )
    else:
        summary_prompt = SystemMessage(
            content=get_prompt("summarize_conversation"),
        )

    messages = list(state["messages"]) + [summary_prompt]

    try:
        response = await get_model().ainvoke(messages)
    except Exception as exc:
        logger.warning("Summary generation failed (%s) – keeping messages as-is", exc)
        return {}

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    logger.info(
        "Conversation summarised; trimmed %d messages", len(delete_messages),
    )
    return {
        "messages": delete_messages,
        "summary": response.content,
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


async def should_continue(state: AgenticRAGState) -> str:
    """Classify the user's input to route between casual chat and RAG.

    Uses the ``flow_decision`` prompt (a ``chat``/``rag`` classifier)
    to decide whether the message needs document retrieval or can be
    answered directly.
    """
    question = _extract_latest_user_question(state["messages"])
    prompt = get_prompt("flow_decision")

    try:
        response = await get_model().ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        decision = _content_to_str(response.content).strip().lower()
    except Exception as exc:
        logger.warning("Flow classification failed (%s) – defaulting to chat path", exc)
        decision = "chat"

    if decision.startswith("rag"):
        logger.info("Classified as document question – routing to rag_chat_node")
        return "rag_chat_node"

    logger.info("Classified as casual/general chat – routing to chat_node")
    return "chat_node"


def route_after_rag(state: AgenticRAGState) -> str:
    """Route from rag_chat_node: to retrieve if tool call, else check summary."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "retrieve"
    if len(state["messages"]) > settings.SUMMARY_MESSAGE_THRESHOLD:
        return "summary_node"
    return END


async def grade_documents(
    state: AgenticRAGState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Assess whether retrieved documents are relevant.

    Uses a fast entity-match shortcut before falling back to the LLM
    grader.  This makes relevance detection model-agnostic and avoids
    false negatives when small LLMs are confused by large context.

    Returns the name of the next node to route to.
    Falls open (-> generate_answer) if grading itself fails.
    """
    messages = state["messages"]
    question = _extract_latest_user_question(messages)
    context = _extract_tool_response(messages)
    max_retries = settings.MAX_QUERY_RETRIES
    current_retries = state.get("retry_count", 0)

    if not context or context.strip() == "No relevant documents found in the knowledge base.":
        if current_retries < max_retries:
            logger.info("Empty retrieval – rewriting (retry %d/%d)", current_retries + 1, max_retries)
            return "rewrite_question"
        logger.info("Empty retrieval – max retries reached, generating best-effort answer")
        return "generate_answer"

    if _entity_match(question, context):
        logger.info("Documents pass entity-match shortcut – skipping LLM grading")
        return "generate_answer"

    grade_prompt = get_prompt("grade_documents")
    combined = (
        f"{grade_prompt}\n\n"
        f"Retrieved documents:\n{context}\n\n"
        f"User question: {question}"
    )

    try:
        response = await get_model().ainvoke([HumanMessage(content=combined)])
        score = _content_to_str(response.content).strip().lower()
    except Exception as exc:
        logger.warning("Document grading failed (%s) – defaulting to relevant", exc)
        return "generate_answer"

    if score.startswith("yes"):
        logger.info("Documents graded as relevant")
        return "generate_answer"

    if current_retries < max_retries:
        logger.info("Documents graded as NOT relevant – rewriting (retry %d/%d)", current_retries + 1, max_retries)
        return "rewrite_question"

    logger.info("Documents graded as NOT relevant – max retries reached, generating best-effort answer")
    return "generate_answer"


def should_summarize(state: AgenticRAGState) -> str:
    """Check if conversation has grown past the summary threshold."""
    if len(state["messages"]) > settings.SUMMARY_MESSAGE_THRESHOLD:
        return "summary_node"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:
    workflow = StateGraph(AgenticRAGState)

    # Nodes
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("rag_chat_node", rag_chat_node)
    workflow.add_node("retrieve", ToolNode(_tools))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("summary_node", summary_node)

    # START -> classifier decides chat vs RAG
    workflow.add_conditional_edges(
        START,
        should_continue,
        {"chat_node": "chat_node", "rag_chat_node": "rag_chat_node"},
    )

    # chat_node (greeting) -> check summary -> END
    workflow.add_conditional_edges(
        "chat_node",
        should_summarize,
        {"summary_node": "summary_node", END: END},
    )

    # rag_chat_node -> retrieve (tool call) or check summary (no tool call)
    workflow.add_conditional_edges(
        "rag_chat_node",
        route_after_rag,
        {"retrieve": "retrieve", "summary_node": "summary_node", END: END},
    )

    # retrieve -> grade relevance
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )

    # generate_answer -> check summary -> END
    workflow.add_conditional_edges(
        "generate_answer",
        should_summarize,
        {"summary_node": "summary_node", END: END},
    )

    # rewrite loop
    workflow.add_edge("rewrite_question", "rag_chat_node")

    # summary always terminates
    workflow.add_edge("summary_node", END)

    return workflow


_workflow = _build_graph()

# Module-level graph without checkpointer (used by LangGraph Studio / CLI).
rag_graph = _workflow.compile()

# Checkpointer for conversation memory (thread-level persistence).
# Initialised lazily via ``get_rag_graph`` so that the async event loop
# is available.
_compiled_graph = None
_checkpointer: AsyncSqliteSaver | None = None


async def get_rag_graph():
    """Return the compiled graph with an active checkpointer.

    Must be called from within an async context (e.g. a FastAPI handler).
    The checkpointer and compiled graph are created once and reused.
    """
    global _compiled_graph, _checkpointer

    if _compiled_graph is not None:
        return _compiled_graph

    db_path = str(settings.CONVERSATION_DB_PATH)
    conn = await aiosqlite.connect(db_path)
    _checkpointer = AsyncSqliteSaver(conn)
    await _checkpointer.setup()
    _compiled_graph = _workflow.compile(checkpointer=_checkpointer)
    logger.info("Agentic RAG graph compiled with SQLite checkpointer at %s", db_path)
    return _compiled_graph


async def shutdown_checkpointer() -> None:
    """Cleanly close the checkpointer connection (call on app shutdown)."""
    global _compiled_graph, _checkpointer
    if _checkpointer is not None:
        try:
            await _checkpointer.conn.close()
        except Exception as exc:
            logger.warning("Error closing checkpointer: %s", exc)
        _checkpointer = None
        _compiled_graph = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_messages(messages: list) -> list:
    """Remove orphaned ToolMessages that lack a preceding AIMessage with tool_calls.

    OpenAI strictly requires every ``tool`` message to follow an ``assistant``
    message that contains ``tool_calls``.  Ollama is lenient about this, so
    when switching providers the persisted conversation may contain orphaned
    tool messages.  This function strips them to prevent 400 errors.
    """
    tool_call_ids: set = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id") or tc.get("tool_call_id")
                if tc_id:
                    tool_call_ids.add(tc_id)

    cleaned: list = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id and tc_id not in tool_call_ids:
                logger.debug("Stripped orphaned ToolMessage (tool_call_id=%s)", tc_id)
                continue
            if not tool_call_ids and tc_id:
                logger.debug("Stripped ToolMessage — no tool_calls found in history")
                continue
        cleaned.append(msg)
    return cleaned


def _content_to_str(content) -> str:
    """Normalise message content that may be ``str`` or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return str(content)


def _extract_latest_user_question(messages: list) -> str:
    """Walk messages backwards to find the most recent HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _content_to_str(msg.content)
    return ""


def _extract_tool_response(messages: list) -> str:
    """Return the content of the most recent ToolMessage."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return _content_to_str(msg.content)
    return ""


_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "you", "your", "he", "she", "it", "we", "they", "this", "that",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "as", "into", "through", "and", "or", "not", "no", "but", "if",
    "all", "any", "some", "what", "which", "who", "whom", "how", "when",
    "where", "please", "provide", "check", "need", "want", "give",
    "information", "details", "tell", "show", "find", "get",
})


def _entity_match(question: str, context: str, min_hits: int = 2) -> bool:
    """Check if key terms from the question appear in the retrieved context.

    Extracts significant words (non-stopword, 3+ chars) from the question
    and checks how many appear in the context.  If at least ``min_hits``
    significant terms match, the context is considered relevant.

    This is a lightweight, model-agnostic heuristic that prevents the LLM
    grader from incorrectly rejecting obviously relevant documents.
    """
    try:
        tokens = re.findall(r"[A-Za-z0-9]+", question.lower())
        significant = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]
        if not significant:
            return False
        context_lower = context.lower()
        hits = sum(1 for t in significant if t in context_lower)
        ratio = hits / len(significant) if significant else 0
        matched = hits >= min_hits and ratio >= 0.3
        if matched:
            logger.debug(
                "Entity match: %d/%d significant terms found (ratio=%.2f)",
                hits, len(significant), ratio,
            )
        return matched
    except Exception:
        return False
