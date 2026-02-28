"""Sync utilities for data folder changes.

This module contains functions for syncing data folder changes.
Note: Task scheduling is now handled by APScheduler in app.core.scheduler
"""

import asyncio
from app.core.config import settings
from app.core.hash_database import init_hash_db
from app.vectorstore.vectorstore import save_vectorstore


async def sync_data_folder_changes():
    """
    Async function to sync data folder changes.
    
    Runs every 2 minutes to:
    1. Detect new files added to Data folder
    2. Detect deleted files from Data folder
    3. Update hash registry and vectorstore
    
    Returns:
        dict: Results with counts of files added, deleted, and chunks modified
    """
    from app.utils.hash_registry import sync_data_folder_changes as sync_func
    
    # Initialize hash database for this operation
    init_hash_db()
    
    # Run the sync function
    results = await sync_func(settings.BASE_DATA_FOLDER)
    
    # Save vectorstore if changes were made
    if results['chunks_added'] > 0 or results['chunks_removed'] > 0:
        save_vectorstore()
    
    return results

