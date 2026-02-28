import asyncio
import logging

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional

from app.core.config import settings
from app.utils.folder_management import get_base_data_folder
from app.utils.hash_registry import (
    calculate_hash_from_bytes,
    lookup_hash,
    register_hash,
    update_processing_status,
)
from app.utils.document_converstion import process_file
from app.vectorstore.operations import add_documents
from app.vectorstore.vectorstore import vsm, save_vectorstore


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/all_files_in_hash_registry/")
async def list_files_in_hash_registry(folder_name: Optional[str] = None):
    """
    List all registered files, optionally filtered by folder.

    Args:
        folder_name: Optional folder to filter by

    Returns:
        List of registered files with their info
    """
    from app.utils.hash_registry import get_all_hashes, get_hashes_by_folder

    try:
        if folder_name:
            files = get_hashes_by_folder(folder_name)
        else:
            files = get_all_hashes()

        return {
            "count": len(files),
            "files": [
                {
                    "id": f.id,
                    "file_name": f.file_name,
                    "folder_name": f.folder_name,
                    "file_type": f.file_type,
                    "file_size": f.file_size,
                    "is_processed": f.is_processed,
                    "chunk_count": f.chunk_count,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in files
            ],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/uploadfile/")
async def create_upload_file(
    file: UploadFile = File(...),
    folder_name: str = "default",
    auto_process: bool = True,
):
    """
    Upload a file to a specified folder with deduplication.

    Flow:
    1. Read file bytes and calculate hash
    2. Check if file with same content already exists (global search)
    3. If duplicate → return info about existing file
    4. If new → save, process, index in vectorstore, register hash

    Args:
        file: The file to upload
        folder_name: Target folder name (default: "default")
        auto_process: Whether to process and index the file (default: True)

    Returns:
        JSON with upload result and processing info
    """
    try:
        # 1. Validate file extension
        file_extension = "." + file.filename.split(".")[-1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "File type not allowed",
                    "allowed_types": list(settings.ALLOWED_EXTENSIONS),
                },
            )

        # 2. Read file bytes
        file_bytes = await file.read()
        file_size = len(file_bytes)

        # 3. Calculate hash BEFORE saving
        content_hash = calculate_hash_from_bytes(file_bytes)

        # 4. Check for duplicates (global search across all folders)
        lookup_result = lookup_hash(content_hash)
        if lookup_result.exists:
            return JSONResponse(
                status_code=409,  # Conflict
                content={
                    "error": "Duplicate file detected",
                    "message": lookup_result.message,
                    "existing_file": {
                        "file_name": lookup_result.file_name,
                        "file_path": lookup_result.file_path,
                        "folder_name": lookup_result.folder_name,
                    },
                    "duplicate": True,
                },
            )

        # 5. File is new - save to target folder
        target_folder = get_base_data_folder() / folder_name
        target_folder.mkdir(parents=True, exist_ok=True)

        file_path = target_folder / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(file_bytes)

        # 6. Register hash in registry
        file_type = file_extension.lstrip(".")
        register_hash(
            content_hash=content_hash,
            file_name=file.filename,
            file_path=str(file_path),
            folder_name=folder_name,
            file_type=file_type,
            file_size=file_size,
            is_processed=False,
        )

        result = {
            "success": True,
            "filename": file.filename,
            "folder": folder_name,
            "file_path": str(file_path),
            "file_size": file_size,
            "content_hash": content_hash,
            "processed": False,
            "chunks": 0,
        }

        # 7. Process file if auto_process is enabled
        if auto_process:
            try:
                # Process file into chunks with metadata
                # Offload sync parsing/chunking to worker thread to avoid blocking event loop.
                doc_info, chunks = await asyncio.to_thread(
                    process_file,
                    file_path=str(file_path),
                    folder_name=folder_name,
                )

                # Add chunks to vectorstore
                if chunks:
                    await add_documents(chunks)
                    # Persist immediately to reduce data loss window on crash/restart.
                    try:
                        await asyncio.to_thread(save_vectorstore)
                    except Exception as persist_error:
                        logger.warning(
                            "Vectorstore persistence failed after upload '%s': %s",
                            file.filename,
                            persist_error,
                        )
                        result["persistence_warning"] = (
                            "Indexed in memory, but immediate disk persistence failed. "
                            "Data will be retried on normal shutdown."
                        )

                # Update processing status in registry
                update_processing_status(
                    content_hash=content_hash,
                    is_processed=True,
                    chunk_count=len(chunks),
                )

                result["processed"] = True
                result["chunks"] = len(chunks)
                result["total_pages"] = doc_info.total_pages
                result["document_info"] = {
                    "file_name": doc_info.file_name,
                    "file_type": doc_info.file_type,
                    "total_pages": doc_info.total_pages,
                    "total_chunks": doc_info.total_chunks,
                }

            except Exception as process_error:
                # File saved but processing failed
                result["processed"] = False
                result["processing_error"] = str(process_error)

        return JSONResponse(status_code=201, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.delete("/delete_file/")
async def delete_file_endpoint(file_name: str):
    """
    Delete a specific file from disk, hash registry, and vectorstore.

    Args:
        file_name: Name of the file to delete
        folder_name: Folder where the file is located

    Returns:
        JSON with deletion result
    """
    from app.utils.folder_management import delete_file
    from app.utils.hash_registry import delete_hashes_by_file_name
    from app.vectorstore.operations import delete_documents_by_file_name
    from app.utils.hash_registry import get_all_hashes

    try:
        # Check if file exists in registry
        files = get_all_hashes()
        file_exists = False
        folder_name = ""
        for file in files:
            if file.file_name == file_name:
                file_exists = True
                folder_name = file.folder_name
                break
        if not file_exists:
            return JSONResponse(
                status_code=404,
                content={"error": f"File '{file_name}' not found in registry."},
            )
        # 1. Delete from disk
        success, message = delete_file(file_name, folder_name)
        if not success:
            return JSONResponse(status_code=400, content={"message": message})
        # 2. Delete from hash registry
        hash_deleted = delete_hashes_by_file_name(file_name)

        # 3. Delete from vectorstore
        docs_deleted = delete_documents_by_file_name(file_name)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"File '{file_name}' deleted successfully.",
                "hash_registry_entry_deleted": hash_deleted,
                "vectorstore_chunks_deleted": docs_deleted,
            },
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/all_files_in_base_folder/")
async def list_all_files_in_base_folder():
    """
    List all files in the base data folder across all subfolders.

    Returns:
        List of file paths
    """
    from app.utils.folder_management import list_all_files_in_base_folder,get_base_data_folder

    try:
        files = list_all_files_in_base_folder()
        base_folder = get_base_data_folder()
        return {
            "base_folder": str(base_folder),
            "count": len(files),
            "files": [str(f) for f in files],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})