"""
Utility script to set up the vector store once.
This script processes all PDF files in the current directory and creates a FAISS index.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Chunking configuration
CHUNK_SIZE = 400  # number of characters
CHUNK_OVERLAP = 50

def load_pdf_text(pdf_path: str) -> List[str]:
    """Extract text from each page of a PDF."""
    print(f"Processing {pdf_path}...")
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text.strip())
    return pages

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk the text using character-based splitting with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_all_pdf_chunks() -> List[Dict]:
    """Process all PDFs and return a list of chunks with metadata."""
    chunks_with_metadata = []
    current_dir = Path.cwd()
    
    for filename in current_dir.glob("*.pdf"):
        pages = load_pdf_text(str(filename))
        for i, page_text in enumerate(pages):
            chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for j, chunk in enumerate(chunks):
                chunks_with_metadata.append({
                    "content": chunk,
                    "source_file": filename.name,
                    "page": i + 1
                })
    return chunks_with_metadata

def setup_vector_store() -> bool:
    """
    Set up the vector store with all PDF files in the current directory.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Get all chunks from PDFs
        print("Processing PDFs and creating chunks...")
        chunks_with_metadata = get_all_pdf_chunks()
        
        if not chunks_with_metadata:
            print("No chunks found in PDFs")
            return False
            
        print(f"Created {len(chunks_with_metadata)} chunks from PDFs")

        # Extract texts for embedding
        texts = [chunk["content"] for chunk in chunks_with_metadata]

        # Create embeddings
        print("Computing embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Create and save FAISS index
        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Using inner product for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize for cosine similarity
        index.add(embeddings)

        # Save index and metadata
        print("Saving index and metadata...")
        faiss.write_index(index, "vector.faiss")
        
        with open("metadata.pkl", "wb") as f:
            pickle.dump(chunks_with_metadata, f)

        print("✅ FAISS index and metadata saved successfully")
        return True

    except Exception as e:
        print(f"Error setting up vector store: {str(e)}")
        return False


def main():
    if setup_vector_store():
        print("Vector store setup completed successfully")
    else:
        print("Vector store setup failed")
        exit(1)


if __name__ == "__main__":
    main() 