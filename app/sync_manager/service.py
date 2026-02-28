"""
Core Sync Management Service.

Provides an independent synchronization control layer with:
- Immediate sync triggers
- IST-based scheduled syncs
- Interval adjustment for the existing background scheduler
- Fail-safe sync with exponential-backoff retries and fallback
- Full state tracking and logging

All datetime operations use IST (UTC+05:30).
This module imports — but never modifies — the existing sync logic.
"""

import asyncio
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.core.logging import logger

IST = timezone(timedelta(hours=5, minutes=30))

_MAX_HISTORY = 50


def _now_ist() -> datetime:
    return datetime.now(IST)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")


class SyncManager:
    """
    Singleton-style manager for all admin-initiated sync operations.

    Thread-safe via asyncio.Lock — only one sync operation can run at a time,
    preventing race conditions with concurrent API calls.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._status: str = "idle"
        self._last_sync_result: Optional[Dict[str, Any]] = None
        self._next_scheduled_ist: Optional[str] = None
        self._scheduled_task: Optional[asyncio.Task] = None
        self._total_syncs_completed: int = 0
        self._total_syncs_failed: int = 0
        self._current_interval: Optional[int] = None
        self._sync_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def trigger_sync(self) -> Dict[str, Any]:
        """Trigger an immediate synchronization."""
        if self._lock.locked():
            logger.warning("[SYNC_MANAGER] Trigger rejected — sync already running")
            return {
                "success": False,
                "message": "A sync operation is already in progress. Please wait for it to complete.",
            }

        async with self._lock:
            return await self._execute_sync("manual_trigger")

    async def schedule_sync(self, scheduled_time_ist_str: str) -> Dict[str, Any]:
        """
        Schedule a one-time sync at the given IST datetime string.

        Args:
            scheduled_time_ist_str: Datetime in ``YYYY-MM-DD HH:MM:SS`` (IST).
        """
        try:
            naive = datetime.strptime(scheduled_time_ist_str.strip(), "%Y-%m-%d %H:%M:%S")
            scheduled_dt = naive.replace(tzinfo=IST)
        except ValueError:
            return {
                "success": False,
                "message": "Invalid datetime format. Required: YYYY-MM-DD HH:MM:SS",
            }

        now = _now_ist()
        if scheduled_dt <= now:
            return {
                "success": False,
                "message": (
                    f"Scheduled time ({_fmt(scheduled_dt)}) must be in the future. "
                    f"Current IST time: {_fmt(now)}"
                ),
            }

        if self._scheduled_task and not self._scheduled_task.done():
            self._scheduled_task.cancel()
            logger.info("[SYNC_MANAGER] Previous scheduled sync cancelled")

        delay = (scheduled_dt - now).total_seconds()
        self._next_scheduled_ist = _fmt(scheduled_dt)
        self._scheduled_task = asyncio.create_task(
            self._delayed_sync(delay, self._next_scheduled_ist)
        )

        logger.info(f"[SYNC_MANAGER] Sync scheduled for {self._next_scheduled_ist} (in {delay:.0f}s)")
        return {
            "success": True,
            "message": f"Sync scheduled for {self._next_scheduled_ist}",
            "scheduled_time_ist": self._next_scheduled_ist,
            "delay_seconds": round(delay, 2),
        }

    def update_interval(self, new_interval: int) -> Dict[str, Any]:
        """
        Update the sync interval for ALL scheduled scan jobs at runtime.

        Reschedules both the BASE_DATA_FOLDER sync and the file-source
        auto-scan so they stay on the same cadence.
        """
        from apscheduler.triggers.interval import IntervalTrigger
        from app.core.scheduler import scheduler as existing_scheduler
        from app.filesource.scanner import FILESOURCE_JOB_ID

        old_interval = self.current_interval

        try:
            new_trigger = IntervalTrigger(seconds=new_interval)

            existing_scheduler.reschedule_job(
                "sync-data-folder-every-2-minutes",
                trigger=new_trigger,
            )

            if existing_scheduler.get_job(FILESOURCE_JOB_ID):
                existing_scheduler.reschedule_job(
                    FILESOURCE_JOB_ID,
                    trigger=IntervalTrigger(seconds=new_interval),
                )
                logger.info(f"[SYNC_MANAGER] File-source scan also rescheduled to {new_interval}s")

            self._current_interval = new_interval
            logger.info(
                f"[SYNC_MANAGER] Background interval changed: {old_interval}s -> {new_interval}s"
            )
            return {
                "success": True,
                "message": f"Sync interval updated from {old_interval}s to {new_interval}s",
                "old_interval_seconds": old_interval,
                "new_interval_seconds": new_interval,
            }
        except Exception as e:
            logger.error(f"[SYNC_MANAGER] Interval update failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to update interval: {e}",
            }

    def get_status(self) -> Dict[str, Any]:
        """Return the current sync manager state."""
        from app.core.scheduler import scheduler as existing_scheduler

        next_auto_run_ist: Optional[str] = None
        try:
            job = existing_scheduler.get_job("sync-data-folder-every-2-minutes")
            if job and job.next_run_time:
                next_auto_run_ist = _fmt(job.next_run_time.astimezone(IST))
        except Exception:
            pass

        return {
            "status": self._status,
            "current_interval_seconds": self.current_interval,
            "next_auto_sync_ist": next_auto_run_ist,
            "next_scheduled_sync_ist": self._next_scheduled_ist,
            "last_sync_result": self._last_sync_result,
            "total_syncs_completed": self._total_syncs_completed,
            "total_syncs_failed": self._total_syncs_failed,
        }

    async def failsafe_sync(self, max_retries: int = 3) -> Dict[str, Any]:
        """
        Robust fail-safe synchronization with retry and fallback.

        Strategy:
        1. Attempt normal sync up to *max_retries* times with exponential backoff.
        2. If all attempts fail, execute a lightweight fallback (registry-only sync).
        3. Every step is logged; errors are collected and returned.
        """
        if self._lock.locked():
            logger.warning("[SYNC_MANAGER] Failsafe rejected — sync already running")
            return {
                "success": False,
                "message": "A sync operation is already in progress",
                "attempts": 0,
                "errors": ["Concurrent sync detected — cannot proceed"],
                "fallback_used": False,
            }

        async with self._lock:
            attempts = 0
            errors: List[str] = []

            for attempt in range(1, max_retries + 1):
                attempts = attempt
                logger.info(f"[SYNC_MANAGER] Failsafe attempt {attempt}/{max_retries}")
                try:
                    result = await self._execute_sync(f"failsafe_attempt_{attempt}")
                    if result.get("success"):
                        return {
                            "success": True,
                            "message": f"Sync succeeded on attempt {attempt}",
                            "attempts": attempt,
                            "sync_result": result,
                            "errors": errors,
                            "fallback_used": False,
                        }
                    errors.append(f"Attempt {attempt}: {result.get('message', 'Unknown error')}")
                except Exception as exc:
                    errors.append(f"Attempt {attempt}: {exc}")
                    logger.error(f"[SYNC_MANAGER] Failsafe attempt {attempt} error: {exc}")

                if attempt < max_retries:
                    backoff = 2 ** attempt
                    logger.info(f"[SYNC_MANAGER] Waiting {backoff}s before retry…")
                    await asyncio.sleep(backoff)

            # All retries exhausted — fallback
            logger.warning("[SYNC_MANAGER] All retries exhausted, executing fallback sync")
            try:
                fallback_result = await self._fallback_sync()
                return {
                    "success": fallback_result.get("success", False),
                    "message": (
                        "Fallback registry-only sync completed"
                        if fallback_result.get("success")
                        else "Fallback sync also failed"
                    ),
                    "attempts": attempts,
                    "sync_result": fallback_result,
                    "errors": errors,
                    "fallback_used": True,
                }
            except Exception as exc:
                errors.append(f"Fallback: {exc}")
                logger.error(f"[SYNC_MANAGER] Fallback failed: {exc}", exc_info=True)
                return {
                    "success": False,
                    "message": "All sync attempts including fallback have failed",
                    "attempts": attempts,
                    "errors": errors,
                    "fallback_used": True,
                }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_interval(self) -> int:
        if self._current_interval is not None:
            return self._current_interval
        from app.core.config import settings
        return settings.SYNC_INTERVAL_SECONDS

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _delayed_sync(self, delay_seconds: float, label: str) -> None:
        """Sleep until the scheduled time, then execute sync."""
        try:
            logger.info(f"[SYNC_MANAGER] Waiting {delay_seconds:.0f}s for scheduled sync at {label}")
            await asyncio.sleep(delay_seconds)
            if self._lock.locked():
                logger.warning("[SYNC_MANAGER] Scheduled sync skipped — another sync is running")
                return
            async with self._lock:
                await self._execute_sync("scheduled")
        except asyncio.CancelledError:
            logger.info("[SYNC_MANAGER] Scheduled sync cancelled")
        except Exception as exc:
            logger.error(f"[SYNC_MANAGER] Scheduled sync error: {exc}", exc_info=True)
        finally:
            self._next_scheduled_ist = None

    async def _execute_sync(self, trigger_source: str) -> Dict[str, Any]:
        """
        Core sync execution — wraps the existing ``sync_data_folder_changes``.

        Returns a structured result dict.  Never raises to the caller.
        """
        sync_id = uuid.uuid4().hex[:8]
        started_at = _now_ist()
        self._status = "running"

        logger.info(f"[SYNC_MANAGER] Sync {sync_id} started (source: {trigger_source})")

        try:
            from app.core.config import settings
            from app.core.hash_database import init_hash_db
            from app.utils.hash_registry import sync_data_folder_changes
            from app.vectorstore.vectorstore import vsm, save_vectorstore

            init_hash_db()
            results = await sync_data_folder_changes(settings.BASE_DATA_FOLDER)

            if results.get("chunks_added", 0) > 0 or results.get("chunks_removed", 0) > 0:
                save_vectorstore()
                active = vsm.active
                logger.info(f"[SYNC_MANAGER] Vectorstore saved ({active.index.ntotal if active else 0} chunks)")

            completed_at = _now_ist()
            duration = (completed_at - started_at).total_seconds()

            result = {
                "success": True,
                "message": "Sync completed successfully",
                "sync_id": sync_id,
                "trigger_source": trigger_source,
                "started_at_ist": _fmt(started_at),
                "completed_at_ist": _fmt(completed_at),
                "duration_seconds": round(duration, 2),
                "details": results,
                "errors": results.get("errors", []),
            }

            self._record_result(result, success=True)
            logger.info(f"[SYNC_MANAGER] Sync {sync_id} completed in {duration:.2f}s")
            return result

        except Exception as exc:
            completed_at = _now_ist()
            duration = (completed_at - started_at).total_seconds()

            result = {
                "success": False,
                "message": f"Sync failed: {exc}",
                "sync_id": sync_id,
                "trigger_source": trigger_source,
                "started_at_ist": _fmt(started_at),
                "completed_at_ist": _fmt(completed_at),
                "duration_seconds": round(duration, 2),
                "errors": [str(exc), traceback.format_exc()],
            }

            self._record_result(result, success=False)
            logger.error(f"[SYNC_MANAGER] Sync {sync_id} failed: {exc}", exc_info=True)
            return result

        finally:
            self._status = "idle"

    async def _fallback_sync(self) -> Dict[str, Any]:
        """
        Lightweight fallback: synchronise the hash registry only (no vectorstore).

        This is useful when the full sync pipeline fails (e.g. embedding model
        is unreachable) but we still want to detect file-system changes.
        """
        logger.info("[SYNC_MANAGER] Running fallback registry-only sync")
        self._status = "running"
        try:
            from app.core.config import settings
            from app.core.hash_database import init_hash_db
            from app.utils.hash_registry import sync_all_folders

            init_hash_db()
            results = sync_all_folders(settings.BASE_DATA_FOLDER)

            total_registered = sum(results.values())
            logger.info(
                f"[SYNC_MANAGER] Fallback sync done — {total_registered} new files registered"
            )
            return {
                "success": True,
                "message": f"Fallback registry-only sync completed ({total_registered} new files)",
                "details": results,
            }
        except Exception as exc:
            logger.error(f"[SYNC_MANAGER] Fallback sync error: {exc}", exc_info=True)
            return {
                "success": False,
                "message": f"Fallback sync failed: {exc}",
            }
        finally:
            self._status = "idle"

    def _record_result(self, result: Dict[str, Any], *, success: bool) -> None:
        self._last_sync_result = result
        if success:
            self._total_syncs_completed += 1
        else:
            self._total_syncs_failed += 1
        self._sync_history.append(result)
        if len(self._sync_history) > _MAX_HISTORY:
            self._sync_history = self._sync_history[-_MAX_HISTORY:]


# Module-level singleton — imported by the router
sync_manager = SyncManager()
