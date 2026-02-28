"""Background task scheduler using APScheduler."""

import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.core.config import settings
from app.core.hash_database import init_hash_db
from app.core.logging import logger
from app.vectorstore.vectorstore import vsm, save_vectorstore


scheduler = AsyncIOScheduler()


async def sync_data_folder_changes_job():
    """
    Background job to sync data folder changes.
    
    Runs every 2 minutes to:
    1. Detect new files added to Data folder
    2. Detect deleted files from Data folder
    3. Update hash registry and vectorstore
    """
    from app.utils.hash_registry import sync_data_folder_changes

    if vsm.rag_blocked:
        logger.info("[SYNC JOB] Skipped — vector store rebuild in progress")
        return
    if vsm.active is None:
        logger.info("[SYNC JOB] Skipped — no active vector store (rebuild required)")
        return

    logger.info("[SYNC JOB] Starting sync_data_folder_changes...")
    logger.info(f"[SYNC JOB] Scanning folder: {settings.BASE_DATA_FOLDER}")
    
    try:
        init_hash_db()
        
        # Run the async sync function
        results = await sync_data_folder_changes(settings.BASE_DATA_FOLDER)
        
        # Log results
        logger.info(f"[SYNC JOB] Sync status!\nNew files added: {results['new_files_added']}\nDuplicates skipped: {results['new_files_skipped_duplicate']}\nDeleted files removed: {results['deleted_files_removed']}\nChunks added: {results['chunks_added']}\nChunks removed: {results['chunks_removed']}")
        # logger.info(f"[SYNC JOB] New files added: {results['new_files_added']}")
        # logger.info(f"[SYNC JOB] Duplicates skipped: {results['new_files_skipped_duplicate']}")
        # logger.info(f"[SYNC JOB] Deleted files removed: {results['deleted_files_removed']}")
        # logger.info(f"[SYNC JOB] Chunks added: {results['chunks_added']}")
        # logger.info(f"[SYNC JOB] Chunks removed: {results['chunks_removed']}")
        
        if results['errors']:
            logger.warning(f"[SYNC JOB] Errors: {len(results['errors'])}")
            for error in results['errors']:
                logger.error(f"[SYNC JOB]    - {error}")
        
        if results['chunks_added'] > 0 or results['chunks_removed'] > 0:
            save_vectorstore()
            active = vsm.active
            count = active.index.ntotal if active else 0
            logger.info(f"[SYNC JOB] Vectorstore saved with {count} chunks.")
        logger.info("[SYNC JOB] Sync complete!")
        return results
        
    except Exception as e:
        logger.error(f"[SYNC JOB] Error during sync: {str(e)}", exc_info=True)
        logger.info("[SYNC JOB] Sync complete!")
        raise


def start_scheduler():
    """Start the background task scheduler."""
    logger.info("Starting APScheduler...")
    
    # Add the sync job: runs every Ssettings.SYNC_INTERVAL_SECONDS seconds
    scheduler.add_job(
        sync_data_folder_changes_job,
        trigger=IntervalTrigger(seconds=settings.SYNC_INTERVAL_SECONDS),
        id='sync-data-folder-every-2-minutes',
        name='Sync Data Folder Changes',
        replace_existing=True,
        max_instances=1,  # Prevent concurrent execution
    )
    
    scheduler.start()
    logger.info("APScheduler started successfully!")


def stop_scheduler():
    """Stop the background task scheduler."""
    if scheduler.running:
        logger.info("Stopping APScheduler...")
        scheduler.shutdown(wait=True)
        logger.info("APScheduler stopped.")
