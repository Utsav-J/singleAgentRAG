"""
File processor for handling different document types.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from docx import Document
from pypdf import PdfReader

from ..config import SUPPORTED_EXTENSIONS
from .local_utils import chunk_text

logger = logging.getLogger(__name__)


def process_text_file(file_path: Path) -> Optional[str]:
    """Process a text file and return its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error processing text file {file_path}: {str(e)}")
        return None


def process_pdf_file(file_path: Path) -> Optional[str]:
    """Process a PDF file and return its content."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {str(e)}")
        return None


def process_docx_file(file_path: Path) -> Optional[str]:
    """Process a DOCX file and return its content."""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error processing DOCX file {file_path}: {str(e)}")
        return None


def process_file(file_path: Path) -> Optional[List[str]]:
    """Process a file and return its chunks."""
    try:
        # Get file extension
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported file type: {ext}")
            return None

        # Process file based on type
        content = None
        if SUPPORTED_EXTENSIONS[ext] == "text":
            content = process_text_file(file_path)
        elif SUPPORTED_EXTENSIONS[ext] == "pdf":
            content = process_pdf_file(file_path)
        elif SUPPORTED_EXTENSIONS[ext] == "docx":
            content = process_docx_file(file_path)

        if content is None:
            return None

        # Chunk the content
        return chunk_text(content)

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None


def process_files(file_paths: List[Path]) -> Dict[str, List[str]]:
    """Process multiple files and return their chunks."""
    results = {}
    for file_path in file_paths:
        chunks = process_file(file_path)
        if chunks:
            results[str(file_path)] = chunks
    return results 