"""Long-term memory powered by LangMem.

Provides cross-conversation memory that persists user preferences, facts,
and knowledge extracted from chat interactions.  Uses ``InMemoryStore``
with JSON file persistence for server-restart durability.

Architecture
------------
- **Store**: ``InMemoryStore`` with vector search (nomic-embed-text, 768d)
- **Extraction**: ``create_memory_store_manager`` runs in the background
  after each conversation turn — zero latency impact on responses.
- **Recall**: ``recall_memories(query)`` performs semantic search over the
  store and returns formatted memories for prompt injection.
- **Persistence**: On shutdown the store is serialised to a JSON file;
  on startup it is rehydrated.  This avoids needing Postgres or a
  SQLite BaseStore (unavailable in LangGraph OSS).
"""

import json
import logging
from pathlib import Path

from langgraph.store.memory import InMemoryStore

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (initialised lazily via ``init_memory_store``)
# ---------------------------------------------------------------------------

_store: InMemoryStore | None = None
_memory_manager = None
_NAMESPACE = ("memories",)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _build_embeddings():
    """Return the same embedding model used by the FAISS vectorstore."""
    from app.chatbot.agent.llm import get_embeddings
    return get_embeddings()


async def init_memory_store() -> InMemoryStore:
    """Create (or rehydrate) the long-term memory store.

    Call once during application startup.
    """
    global _store, _memory_manager

    if _store is not None:
        return _store

    embed = _build_embeddings()
    _store = InMemoryStore(
        index={
            "dims": settings.DINMS,
            "embed": embed,
        }
    )

    store_path = Path(settings.MEMORY_STORE_PATH)
    if store_path.exists():
        try:
            raw = json.loads(store_path.read_text(encoding="utf-8"))
            count = 0
            for item in raw:
                _store.put(
                    namespace=tuple(item["namespace"]),
                    key=item["key"],
                    value=item["value"],
                )
                count += 1
            logger.info("Rehydrated %d long-term memories from %s", count, store_path)
        except Exception as exc:
            logger.warning("Failed to load memory store from %s: %s", store_path, exc)

    try:
        from langmem import create_memory_store_manager
        from app.chatbot.agent.llm import get_model

        _memory_manager = create_memory_store_manager(
            get_model(),
            namespace=_NAMESPACE,
            store=_store,
            instructions=(
                "Extract noteworthy facts, user preferences, and key information "
                "from the conversation. Focus on details the user would expect the "
                "assistant to remember in future conversations."
            ),
        )
        logger.info("LangMem memory manager initialised")
    except Exception as exc:
        logger.warning("Failed to init LangMem memory manager: %s — background extraction disabled", exc)
        _memory_manager = None

    return _store


def get_memory_store() -> InMemoryStore | None:
    """Return the store if initialised, else ``None``."""
    return _store


async def shutdown_memory_store() -> None:
    """Persist the in-memory store to disk and clean up."""
    global _store, _memory_manager

    if _store is None:
        return

    store_path = Path(settings.MEMORY_STORE_PATH)
    try:
        items = _store.search(_NAMESPACE, limit=10000)
        serialisable = [
            {
                "namespace": list(item.namespace),
                "key": item.key,
                "value": item.value,
            }
            for item in items
        ]
        store_path.write_text(
            json.dumps(serialisable, default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Persisted %d long-term memories to %s", len(serialisable), store_path)
    except Exception as exc:
        logger.error("Failed to persist memory store: %s", exc, exc_info=True)

    _store = None
    _memory_manager = None


# ---------------------------------------------------------------------------
# Background extraction
# ---------------------------------------------------------------------------


async def extract_memories_background(messages: list[dict]) -> None:
    """Extract and store memories from a conversation turn.

    Runs in the background (via ``asyncio.create_task``) so it never
    blocks the chat response.  Failures are logged and swallowed.
    """
    if _memory_manager is None or _store is None:
        return

    try:
        await _memory_manager.ainvoke({"messages": messages})
        logger.debug("Background memory extraction completed for %d messages", len(messages))
    except Exception as exc:
        logger.warning("Background memory extraction failed: %s", exc)


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


async def recall_memories(query: str, limit: int = 5) -> str:
    """Search the memory store and return formatted context for prompt injection.

    Returns an empty string if no memories are found or the store is
    unavailable — callers can safely concatenate the result.
    """
    if _store is None or not query:
        return ""

    try:
        results = _store.search(_NAMESPACE, query=query, limit=limit)
        if not results:
            return ""

        parts = []
        for item in results:
            value = item.value
            if isinstance(value, dict):
                content = value.get("content", value)
                if isinstance(content, dict):
                    content = content.get("content", str(content))
                parts.append(str(content))
            else:
                parts.append(str(value))

        if not parts:
            return ""

        return "Relevant memories from past conversations:\n- " + "\n- ".join(parts)
    except Exception as exc:
        logger.warning("Memory recall failed: %s", exc)
        return ""
