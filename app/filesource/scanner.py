"""
Direct scan-and-vectorize engine for file sources.

Reads files **directly** from the source path and feeds them through
the existing processing pipeline.

CRITICAL DESIGN DECISION — separate hash tracking:
  The scanner tracks processed files in ``filesource_processed_hash``
  (inside ``filesource_config.db``), NOT in the main ``hash_registry``.
  This prevents the existing ``sync_data_folder_changes`` from treating
  file-source entries as "deleted files" and removing their chunks
  every cycle.

The auto-scan job is registered on the existing APScheduler so it
runs on the same interval as the BASE_DATA_FOLDER sync.
"""

import hashlib
import logging
import os
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from apscheduler.triggers.interval import IntervalTrigger
from sqlmodel import select

from app.core.config import settings
from app.filesource import crypto
from app.filesource.adapters.registry import create_adapter
from app.filesource.database import get_session as get_fs_session
from app.filesource.models import DEFAULT_PORTS, FileSourceConfig, FilesourceProcessedHash

logger = logging.getLogger("app.filesource")

_ALLOWED_EXT: Set[str] = {ext.lower() for ext in settings.ALLOWED_EXTENSIONS}
FILESOURCE_JOB_ID = "filesource-auto-scan"


# ── Hash tracking (separate from main hash_registry) ─────

def _hash_exists(content_hash: str) -> bool:
    """Check if a file was already processed — checks BOTH registries."""
    session = get_fs_session()
    try:
        hit = session.exec(
            select(FilesourceProcessedHash).where(
                FilesourceProcessedHash.content_hash == content_hash
            )
        ).first()
        if hit:
            return True
    finally:
        session.close()

    try:
        from app.utils.hash_registry import lookup_hash
        return lookup_hash(content_hash).exists
    except Exception:
        return False


def _record_processed(content_hash: str, file_name: str, source_name: str, chunk_count: int) -> None:
    """Record a processed file in the file-source tracking table."""
    session = get_fs_session()
    try:
        existing = session.exec(
            select(FilesourceProcessedHash).where(
                FilesourceProcessedHash.content_hash == content_hash
            )
        ).first()
        if existing:
            return
        entry = FilesourceProcessedHash(
            content_hash=content_hash,
            file_name=file_name,
            source_name=source_name,
            chunk_count=chunk_count,
            processed_at=datetime.utcnow(),
        )
        session.add(entry)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning("[SCANNER] Failed to record hash: %s", exc)
    finally:
        session.close()


def _calculate_hash(file_path: str) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ── Direct scan (single source) ──────────────────────────

async def scan_and_vectorize(source_id: int) -> Dict[str, Any]:
    """
    Scan files at the source path and vectorize them in-place.

    For any locally-accessible path the files are read directly.
    For truly remote sources (SFTP/FTP/SMB) files are downloaded
    to a temp directory, processed, then cleaned up.
    """
    session = get_fs_session()
    try:
        src = session.get(FileSourceConfig, source_id)
        if not src:
            return _result(False, source_id, "", "Source not found")
        if not src.is_enabled:
            return _result(False, source_id, src.name, "Source is disabled")

        base = Path(src.base_path)
        is_locally_accessible = base.exists() and base.is_dir()

        if is_locally_accessible:
            return await _scan_local_path(src)
        elif src.protocol in ("sftp", "smb", "ftp"):
            return await _scan_remote_via_temp(src)
        else:
            return _result(
                False, source_id, src.name,
                f"Path not accessible and protocol '{src.protocol}' has no remote adapter"
            )
    except Exception as exc:
        logger.error("[SCANNER] scan_and_vectorize error: %s", exc, exc_info=True)
        return _result(False, source_id, "", f"Scan error: {exc}")
    finally:
        session.close()


async def _scan_local_path(src: FileSourceConfig) -> Dict[str, Any]:
    """Directly scan a locally-accessible path — zero file copying."""
    from app.utils.document_converstion import process_file
    from app.vectorstore.operations import add_documents
    from app.vectorstore.vectorstore import vsm, save_vectorstore

    base = Path(src.base_path)
    new_files = 0
    skipped = 0
    chunks_added = 0
    errors: List[str] = []

    for root, _dirs, files in os.walk(base):
        for fname in files:
            if not any(fname.lower().endswith(ext) for ext in _ALLOWED_EXT):
                continue
            fpath = os.path.join(root, fname)
            folder_name = Path(root).name or src.name

            try:
                content_hash = _calculate_hash(fpath)
                if _hash_exists(content_hash):
                    skipped += 1
                    continue

                doc_info, chunks = await asyncio.to_thread(
                    process_file,
                    file_path=fpath,
                    folder_name=folder_name,
                )
                if chunks:
                    await add_documents(chunks)
                    chunks_added += len(chunks)

                _record_processed(content_hash, fname, src.name, len(chunks))
                new_files += 1
            except Exception as exc:
                errors.append(f"{fname}: {exc}")
                logger.error("[SCANNER] Error processing %s: %s", fname, exc)

    if chunks_added > 0:
        save_vectorstore()
        active = vsm.active
        logger.info("[SCANNER] Vectorstore saved (%d chunks total)", active.index.ntotal if active else 0)

    msg = f"Scan complete: {new_files} new, {skipped} skipped, {chunks_added} chunks"
    if errors:
        msg += f", {len(errors)} errors"
    logger.info("[SCANNER] %s — source '%s' (direct)", msg, src.name)

    return {
        "success": True,
        "source_id": src.id,
        "source_name": src.name,
        "new_files": new_files,
        "skipped": skipped,
        "chunks_added": chunks_added,
        "errors": errors,
        "message": msg,
        "mode": "direct",
    }


async def _scan_remote_via_temp(src: FileSourceConfig) -> Dict[str, Any]:
    """
    For truly remote sources: download to temp dir, vectorize, clean up.
    Nothing permanent is stored locally except the vectors in FAISS.
    """
    from app.utils.document_converstion import process_file
    from app.vectorstore.operations import add_documents
    from app.vectorstore.vectorstore import vsm, save_vectorstore

    adapter_cfg = _build_config(src)

    try:
        adapter = create_adapter(adapter_cfg)
    except (ValueError, ImportError) as exc:
        return _result(False, src.id, src.name, f"Adapter error: {exc}")

    new_files = 0
    skipped = 0
    chunks_added = 0
    errors: List[str] = []

    with tempfile.TemporaryDirectory(prefix="filesource_scan_") as tmpdir:
        try:
            async with adapter:
                remote_files = await adapter.list_files(_ALLOWED_EXT)
                logger.info("[SCANNER] Remote '%s': %d files found", src.name, len(remote_files))

                for rf in remote_files:
                    local_tmp = os.path.join(tmpdir, rf["relative_path"])
                    try:
                        await adapter.download_file(rf["path"], local_tmp)

                        content_hash = _calculate_hash(local_tmp)
                        if _hash_exists(content_hash):
                            skipped += 1
                            continue

                        doc_info, chunks = await asyncio.to_thread(
                            process_file,
                            file_path=local_tmp,
                            folder_name=src.name,
                        )
                        if chunks:
                            await add_documents(chunks)
                            chunks_added += len(chunks)

                        _record_processed(content_hash, rf["name"], src.name, len(chunks))
                        new_files += 1
                    except Exception as exc:
                        errors.append(f"{rf['name']}: {exc}")
                        logger.error("[SCANNER] Remote scan error %s: %s", rf["name"], exc)

        except Exception as exc:
            logger.error("[SCANNER] Remote scan failed '%s': %s", src.name, exc, exc_info=True)
            return _result(False, src.id, src.name, f"Remote scan failed: {exc}", errors=errors)

    if chunks_added > 0:
        save_vectorstore()
        active = vsm.active
        logger.info("[SCANNER] Vectorstore saved (%d chunks total)", active.index.ntotal if active else 0)

    msg = f"Scan complete: {new_files} new, {skipped} skipped, {chunks_added} chunks"
    if errors:
        msg += f", {len(errors)} errors"
    logger.info("[SCANNER] %s — source '%s' (remote)", msg, src.name)

    return {
        "success": True,
        "source_id": src.id,
        "source_name": src.name,
        "new_files": new_files,
        "skipped": skipped,
        "chunks_added": chunks_added,
        "errors": errors,
        "message": msg,
        "mode": "remote_temp",
    }


# ── Auto-scan (runs on the EXISTING scheduler) ───────────

async def _auto_scan_all_enabled():
    """
    Scheduled job: scan every enabled source.

    Runs on the same APScheduler and interval as the BASE_DATA_FOLDER
    sync so that one interval governs all scanning.
    """
    from app.vectorstore.vectorstore import vsm as _vsm
    if _vsm.rag_blocked:
        logger.info("[SCANNER] Auto-scan skipped — vector store rebuild in progress")
        return
    if _vsm.active is None:
        logger.info("[SCANNER] Auto-scan skipped — no active vector store (rebuild required)")
        return

    session = get_fs_session()
    try:
        sources = session.exec(
            select(FileSourceConfig).where(
                FileSourceConfig.is_enabled == True,  # noqa: E712
            )
        ).all()

        if not sources:
            return

        logger.info("[SCANNER] Auto-scan cycle: %d enabled source(s)", len(sources))
        for src in sources:
            try:
                result = await scan_and_vectorize(src.id)
                logger.info("[SCANNER] Auto-scan '%s': %s", src.name, result.get("message"))
            except Exception as exc:
                logger.error("[SCANNER] Auto-scan error '%s': %s", src.name, exc)
        logger.info("[SCANNER] Auto-scan cycle complete")
    except Exception as exc:
        logger.error("[SCANNER] Auto-scan cycle error: %s", exc, exc_info=True)
    finally:
        session.close()


def register_filesource_scan_job() -> None:
    """
    Add the file-source scan job to the **existing** scheduler.

    Call this after ``start_scheduler()`` so the scheduler is already
    running.  The job uses the same interval as ``SYNC_INTERVAL_SECONDS``
    so both BASE_DATA_FOLDER and file-source scans happen together.
    """
    from app.core.config import settings
    from app.core.scheduler import scheduler as existing_scheduler

    existing_scheduler.add_job(
        _auto_scan_all_enabled,
        trigger=IntervalTrigger(seconds=settings.SYNC_INTERVAL_SECONDS),
        id=FILESOURCE_JOB_ID,
        name="File Source Auto-Scan",
        replace_existing=True,
        max_instances=1,
    )
    logger.info(
        "[SCANNER] Registered on existing scheduler (interval=%ds)",
        settings.SYNC_INTERVAL_SECONDS,
    )


# ── Helpers ───────────────────────────────────────────────

def _build_config(src: FileSourceConfig) -> Dict[str, Any]:
    return {
        "name": src.name,
        "protocol": src.protocol,
        "host": src.host,
        "port": src.port or DEFAULT_PORTS.get(src.protocol),
        "base_path": src.base_path,
        "share_name": src.share_name,
        "domain": src.domain,
        "auth_type": src.auth_type,
        "username": src.username,
        "password": crypto.decrypt(src.encrypted_password or ""),
        "key_file_path": src.key_file_path,
        "passphrase": crypto.decrypt(src.encrypted_passphrase or ""),
        "timeout_seconds": src.timeout_seconds,
        "max_retries": src.max_retries,
    }


def _result(success, source_id, name, msg, errors=None):
    return {
        "success": success,
        "source_id": source_id,
        "source_name": name,
        "new_files": 0,
        "skipped": 0,
        "chunks_added": 0,
        "errors": errors or [],
        "message": msg,
        "mode": "none",
    }
