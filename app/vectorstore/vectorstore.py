"""Dual-index FAISS vector store manager.

Manages two independent FAISS indexes — one for local (Ollama) embeddings
and one for remote (OpenAI / Anthropic / etc.) embeddings.  The admin panel
toggle ``use_remote_models`` selects which index is active.

Switching between pre-built indexes is instant (pointer swap).  When an
index doesn't exist yet or the embedding model changes, the admin triggers
a rebuild that re-embeds every document.  RAG queries are blocked during
rebuild; casual chat continues.
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_REMOTE_DIR_NAME = "vectorstore_index_remote"


class VectorStoreManager:
    """Thread-safe manager for local and remote FAISS indexes."""

    def __init__(self) -> None:
        self._local_vs: Optional[FAISS] = None
        self._remote_vs: Optional[FAISS] = None
        self._mode: str = "local"
        self._lock = threading.Lock()

        self.rag_blocked: bool = False
        self.rebuild_status: dict = {
            "status": "idle",
            "progress": 0,
            "total": 0,
            "message": "",
            "rebuild_required": False,
        }

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def _local_path(self) -> Path:
        return settings.VECTORSTORE_PATH

    @property
    def _remote_path(self) -> Path:
        return settings.VECTORSTORE_PATH.parent / _REMOTE_DIR_NAME

    # ------------------------------------------------------------------
    # Active store
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def active(self) -> Optional[FAISS]:
        """Return the currently active FAISS index (may be ``None``)."""
        if self._mode == "remote":
            return self._remote_vs
        return self._local_vs

    # ------------------------------------------------------------------
    # Initialization (called during app startup)
    # ------------------------------------------------------------------

    def init_local(self) -> None:
        """Load local index from disk or create an empty one."""
        try:
            from app.chatbot.agent.llm import get_embeddings
            emb = get_embeddings()
        except Exception:
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.LLM_BASE_URL,
            )

        if self._local_path.exists():
            try:
                self._local_vs = FAISS.load_local(
                    str(self._local_path), emb,
                    allow_dangerous_deserialization=True,
                )
                logger.info(
                    "Local vector store loaded: %d chunks, dims=%d",
                    self._local_vs.index.ntotal, self._local_vs.index.d,
                )
                return
            except Exception as exc:
                logger.warning("Failed to load local vector store: %s", exc)

        self._local_vs = self._create_empty(emb, settings.DINMS)
        logger.info("Created empty local vector store (dims=%d)", settings.DINMS)

    def init_remote(self) -> None:
        """Try to load remote index from disk (non-fatal if absent)."""
        if not self._remote_path.exists():
            logger.info("No remote vector store on disk — will require rebuild")
            return
        try:
            from app.chatbot.agent.llm import _build_remote_embeddings
            emb = _build_remote_embeddings()
            self._remote_vs = FAISS.load_local(
                str(self._remote_path), emb,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "Remote vector store loaded: %d chunks, dims=%d",
                self._remote_vs.index.ntotal, self._remote_vs.index.d,
            )
        except Exception as exc:
            logger.info("Remote vector store not loaded (will need rebuild): %s", exc)
            self._remote_vs = None

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def switch_mode(self, mode: str) -> dict:
        """Switch between 'local' and 'remote'.

        Returns a status dict indicating whether the target index is ready.
        """
        if mode not in ("local", "remote"):
            return {"success": False, "message": f"Invalid mode: {mode}"}

        with self._lock:
            self._mode = mode
            target = self.active

            if target is None:
                self.rebuild_status["rebuild_required"] = True
                self.rag_blocked = True
                logger.warning(
                    "Switched to %s mode but index not available — RAG blocked, rebuild required",
                    mode,
                )
                return {
                    "success": True,
                    "message": f"Switched to {mode} mode. Rebuild required — no index available.",
                    "rebuild_required": True,
                    "chunks": 0,
                }

            chunks = target.index.ntotal
            self.rag_blocked = False
            self.rebuild_status["rebuild_required"] = False
            logger.info("Switched to %s mode (%d chunks)", mode, chunks)
            return {
                "success": True,
                "message": f"Switched to {mode} mode ({chunks} chunks)",
                "rebuild_required": False,
                "chunks": chunks,
            }

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    async def rebuild(self, target_mode: str) -> dict:
        """Re-embed ALL documents and create a fresh index for *target_mode*.

        Blocks RAG during the rebuild.  On success the active mode is set
        to *target_mode*.
        """
        if self.rebuild_status.get("status") == "running":
            return {"success": False, "message": "A rebuild is already running"}

        self.rag_blocked = True
        self.rebuild_status = {
            "status": "running",
            "progress": 0,
            "total": 0,
            "message": "Starting rebuild...",
            "rebuild_required": False,
        }

        try:
            result = await self._do_rebuild(target_mode)
            return result
        except Exception as exc:
            logger.error("Rebuild failed: %s", exc, exc_info=True)
            self.rebuild_status = {
                "status": "error",
                "progress": 0,
                "total": 0,
                "message": f"Rebuild failed: {exc}",
                "rebuild_required": True,
            }
            return {"success": False, "message": str(exc)}
        finally:
            if self.active is not None:
                self.rag_blocked = False

    async def _do_rebuild(self, target_mode: str) -> dict:
        from app.chatbot.agent.llm import get_embedding_dimensions

        if target_mode == "remote":
            from app.chatbot.agent.llm import _build_remote_embeddings
            emb = _build_remote_embeddings()
        else:
            from app.chatbot.agent.llm import _build_local_embeddings
            emb = _build_local_embeddings()

        self.rebuild_status["message"] = "Detecting embedding dimensions..."
        dims = get_embedding_dimensions(emb)
        logger.info("Rebuild: detected %d dimensions for %s mode", dims, target_mode)

        new_vs = self._create_empty(emb, dims)

        file_list = self._collect_all_files()
        total = len(file_list)
        self.rebuild_status["total"] = total

        if total == 0:
            logger.info("Rebuild: no files found — empty index created")
        else:
            from app.utils.document_converstion import process_file
            errors = []
            for i, (fpath, folder) in enumerate(file_list):
                self.rebuild_status["progress"] = i
                self.rebuild_status["message"] = f"Processing {Path(fpath).name} ({i + 1}/{total})..."
                try:
                    if not Path(fpath).exists():
                        logger.warning("Rebuild: file missing, skipped: %s", fpath)
                        continue
                    _, chunks = await asyncio.to_thread(
                        process_file,
                        file_path=fpath,
                        folder_name=folder,
                    )
                    if chunks:
                        await new_vs.aadd_documents(chunks)
                except Exception as exc:
                    errors.append(f"{Path(fpath).name}: {exc}")
                    logger.warning("Rebuild: error processing %s: %s", fpath, exc)

        with self._lock:
            if target_mode == "remote":
                self._remote_vs = new_vs
                self._save_index(new_vs, self._remote_path)
            else:
                self._local_vs = new_vs
                self._save_index(new_vs, self._local_path)

            self._mode = target_mode

        chunk_count = new_vs.index.ntotal
        self.rebuild_status = {
            "status": "completed",
            "progress": total,
            "total": total,
            "message": f"Rebuild complete: {chunk_count} chunks, {dims}d",
            "rebuild_required": False,
        }
        self.rag_blocked = False

        logger.info(
            "Rebuild complete for %s mode: %d files, %d chunks, %dd",
            target_mode, total, chunk_count, dims,
        )
        return {
            "success": True,
            "mode": target_mode,
            "files_processed": total,
            "chunks": chunk_count,
            "dimensions": dims,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, vs: Optional[FAISS] = None) -> None:
        """Save the given (or active) vector store to its disk path."""
        if vs is None:
            vs = self.active
        if vs is None:
            return
        path = self._remote_path if self._mode == "remote" else self._local_path
        self._save_index(vs, path)

    def save_both(self) -> None:
        """Save both indexes to disk (called on shutdown)."""
        if self._local_vs is not None:
            self._save_index(self._local_vs, self._local_path)
        if self._remote_vs is not None:
            self._save_index(self._remote_vs, self._remote_path)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a summary of both indexes for the admin UI."""
        def _info(vs: Optional[FAISS], path: Path) -> dict:
            if vs is None:
                return {"available": False, "chunks": 0, "dims": 0, "on_disk": path.exists()}
            return {
                "available": True,
                "chunks": vs.index.ntotal,
                "dims": vs.index.d,
                "on_disk": path.exists(),
            }

        return {
            "mode": self._mode,
            "rag_blocked": self.rag_blocked,
            "local": _info(self._local_vs, self._local_path),
            "remote": _info(self._remote_vs, self._remote_path),
            "rebuild": self.rebuild_status,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_all_files() -> list[tuple[str, str]]:
        """Discover files from ALL sources: hash registry, filesystem, and file sources.

        Returns a deduplicated list of ``(file_path, folder_name)`` tuples.
        """
        import os
        seen: set[str] = set()
        files: list[tuple[str, str]] = []

        allowed = {ext.lower() for ext in settings.ALLOWED_EXTENSIONS}

        try:
            from app.utils.hash_registry import get_all_hashes
            for rec in get_all_hashes():
                fp = rec.file_path
                if fp not in seen and Path(fp).exists():
                    seen.add(fp)
                    files.append((fp, rec.folder_name))
        except Exception as exc:
            logger.warning("Rebuild: hash_registry scan failed: %s", exc)

        try:
            base = settings.BASE_DATA_FOLDER
            if base.exists():
                for root, _dirs, fnames in os.walk(base):
                    for fn in fnames:
                        if Path(fn).suffix.lower() not in allowed:
                            continue
                        fp = str(Path(root) / fn)
                        if fp not in seen:
                            seen.add(fp)
                            files.append((fp, Path(root).name))
        except Exception as exc:
            logger.warning("Rebuild: Data folder scan failed: %s", exc)

        try:
            from app.filesource.database import get_session as get_fs_session
            from app.filesource.models import FileSourceConfig
            from sqlmodel import select
            session = get_fs_session()
            try:
                sources = session.exec(
                    select(FileSourceConfig).where(FileSourceConfig.is_enabled == True)  # noqa: E712
                ).all()
                for src in sources:
                    src_base = Path(src.base_path)
                    if src_base.exists() and src_base.is_dir():
                        for root, _dirs, fnames in os.walk(src_base):
                            for fn in fnames:
                                if Path(fn).suffix.lower() not in allowed:
                                    continue
                                fp = str(Path(root) / fn)
                                if fp not in seen:
                                    seen.add(fp)
                                    files.append((fp, Path(root).name or src.name))
            finally:
                session.close()
        except Exception as exc:
            logger.warning("Rebuild: file source scan failed: %s", exc)

        logger.info("Rebuild: collected %d files from all sources", len(files))
        return files

    @staticmethod
    def _create_empty(embeddings, dims: int) -> FAISS:
        index = faiss.IndexFlatIP(dims)
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            normalize_L2=True,
        )

    @staticmethod
    def _save_index(vs: FAISS, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            vs.save_local(str(path))
            logger.info("Vector store saved to %s (%d chunks)", path, vs.index.ntotal)
        except Exception as exc:
            logger.error("Failed to save vector store to %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

vsm = VectorStoreManager()
vsm.init_local()

# Try loading remote too (non-fatal)
try:
    vsm.init_remote()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

def _get_active():
    """Return the active FAISS index (never None — falls back to local)."""
    vs = vsm.active
    if vs is None:
        vs = vsm._local_vs
    return vs


# These module-level names are captured at import time.  For code that does
# ``from app.vectorstore.vectorstore import vector_store`` at the TOP of
# a file, this will be the LOCAL index created at startup.  Code that
# needs the *currently active* index should call ``vsm.active`` or
# use the functions in operations.py.
vector_store = _get_active()

retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
) if vector_store else None


def save_vectorstore(vs: Optional[FAISS] = None) -> None:
    """Backward-compatible save — delegates to vsm."""
    vsm.save(vs)
