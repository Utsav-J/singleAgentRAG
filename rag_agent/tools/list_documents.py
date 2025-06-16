"""
Tool for listing all documents in the documents folder.
"""

from typing import Dict, List

from ..config import CORPORA_DIR
from .local_utils import list_documents


def list_documents_tool() -> dict:
    """
    List all documents in the documents folder.

    Returns:
        dict: A dictionary containing:
            - status: Success or error status
            - message: Description of the result
            - documents: List of document information including:
                - name: Document filename
                - path: Full path to the document
                - size: File size in bytes
                - last_modified: Last modification timestamp
    """
    try:
        documents = list_documents()
        
        return {
            "status": "success",
            "message": f"Found {len(documents)} documents",
            "documents": documents
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing documents: {str(e)}",
            "documents": []
        } 