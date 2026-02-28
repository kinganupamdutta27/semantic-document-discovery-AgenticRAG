"""Vector store CRUD operations.

All functions use ``vsm.active`` to get the currently active FAISS index
(local or remote depending on the admin toggle).  This ensures that
document adds, deletes, and queries always target the correct index.
"""

from langchain_core.documents import Document
from app.core.logging import logger


def _vs():
    """Return the active vector store, raising a clear error if unavailable."""
    from app.vectorstore.vectorstore import vsm
    vs = vsm.active
    if vs is None:
        raise RuntimeError(
            "Vector store unavailable (rebuild in progress or index not built). "
            "Please wait for the rebuild to complete."
        )
    return vs


async def add_documents(documents: list[Document]):
    """Add documents to the active vector store."""
    try:
        await _vs().aadd_documents(documents)
    except RuntimeError:
        raise
    except Exception as e:
        logger.error("Error adding documents: %s", e, exc_info=True)


async def retrieve_similar(query, k=5):
    """Retrieve similar documents from the active vector store."""
    try:
        results = _vs().similarity_search_with_score(query, k=k)
        return [doc for doc, score in results]
    except RuntimeError:
        return []
    except Exception as e:
        logger.error("Error retrieving documents: %s", e, exc_info=True)
        return []


def delete_documents_by_folder(folder_name: str) -> int:
    """Delete all documents from a specific folder."""
    try:
        vs = _vs()
    except RuntimeError:
        return 0

    ids_to_delete = []
    for index_id, doc_id in list(vs.index_to_docstore_id.items()):
        try:
            doc = vs.docstore.search(doc_id)
            if doc and hasattr(doc, 'metadata'):
                if doc.metadata.get('folder_name') == folder_name:
                    ids_to_delete.append(doc_id)
        except Exception:
            continue

    if ids_to_delete:
        try:
            vs.delete(ids_to_delete)
            return len(ids_to_delete)
        except Exception as e:
            logger.error("Error deleting documents by folder: %s", e)

    return 0


def delete_documents_by_file_name(file_name: str) -> int:
    """Delete all documents/chunks from a specific file by its name."""
    try:
        vs = _vs()
    except RuntimeError:
        return 0

    ids_to_delete = []
    for index_id, doc_id in list(vs.index_to_docstore_id.items()):
        try:
            doc = vs.docstore.search(doc_id)
            if doc and hasattr(doc, 'metadata'):
                if doc.metadata.get('file_name') == file_name:
                    ids_to_delete.append(doc_id)
        except Exception:
            continue

    if ids_to_delete:
        try:
            vs.delete(ids_to_delete)
            return len(ids_to_delete)
        except Exception as e:
            logger.error("Error deleting documents by file_name: %s", e)

    return 0


def update_folder_name_in_metadata(old_folder_name: str, new_folder_name: str) -> int:
    """Update folder_name in metadata for all documents from a folder."""
    try:
        vs = _vs()
    except RuntimeError:
        return 0

    updated_count = 0
    for index_id, doc_id in list(vs.index_to_docstore_id.items()):
        try:
            doc = vs.docstore.search(doc_id)
            if doc and hasattr(doc, 'metadata'):
                if doc.metadata.get('folder_name') == old_folder_name:
                    doc.metadata['folder_name'] = new_folder_name
                    if 'file_path' in doc.metadata and old_folder_name in doc.metadata['file_path']:
                        doc.metadata['file_path'] = doc.metadata['file_path'].replace(
                            f"/{old_folder_name}/", f"/{new_folder_name}/"
                        )
                    updated_count += 1
        except Exception:
            continue

    return updated_count


def delete_documents_by_file_path(file_path: str) -> int:
    """Delete all documents/chunks from a specific file path."""
    try:
        vs = _vs()
    except RuntimeError:
        return 0

    ids_to_delete = []
    for index_id, doc_id in list(vs.index_to_docstore_id.items()):
        try:
            doc = vs.docstore.search(doc_id)
            if doc and hasattr(doc, 'metadata'):
                if doc.metadata.get('file_path') == file_path:
                    ids_to_delete.append(doc_id)
        except Exception:
            continue

    if ids_to_delete:
        try:
            vs.delete(ids_to_delete)
            return len(ids_to_delete)
        except Exception as e:
            logger.error("Error deleting documents by file_path: %s", e)

    return 0


def get_total_docs_count() -> int:
    """Get the total number of documents in the active vector store."""
    try:
        return _vs().index.ntotal
    except Exception:
        return 0
