#!/usr/bin/env python3
"""
Example: Chunking PDF documents.

This example shows how to chunk PDF files using Chunky Monkey.

Requirements:
    pip install chunky-monkey[pdf]
    
    Or install a specific PDF backend:
    pip install pymupdf     # Best quality
    pip install pypdf       # Lightweight
    pip install pdfplumber  # Good for tables
"""

from pathlib import Path

# Method 1: Use chunk_file for automatic format detection
from monkey import chunk_file

def chunk_pdf_simple(pdf_path: str):
    """Simplest way to chunk a PDF file."""
    chunks = chunk_file(
        pdf_path,
        max_tokens=1000,  # Tokens per chunk
        overlap=1,        # Sentences overlap
    )
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Page: {chunk.metadata.context.get('page', 'N/A')}")
        print(f"Tokens: {chunk.token_count}")
        print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
    
    return chunks


# Method 2: Use PDFParser directly for more control
def chunk_pdf_advanced(pdf_path: str):
    """Advanced PDF chunking with custom settings."""
    from monkey.parsers import PDFParser
    from monkey import ChunkyMonkey
    
    # Create PDF parser with custom settings
    parser = PDFParser(
        backend="auto",           # Try pymupdf, then pypdf, then pdfplumber
        extract_tables=True,      # Extract tables as markdown
        page_separator="\n\n---\n\n",  # Page break marker
    )
    
    # Parse the PDF
    doc = parser.parse_file(pdf_path)
    
    print(f"Title: {doc.title}")
    print(f"Pages: {doc.metadata.get('page_count', 'unknown')}")
    print(f"Elements extracted: {len(doc.elements)}")
    
    # Create chunker
    chunker = ChunkyMonkey(
        max_tokens=1000,
        overlap=2,
        parser=parser,
    )
    
    # Chunk the document
    chunks = chunker.chunk(doc.get_text(), source=pdf_path)
    
    return chunks


# Method 3: Chunk multiple PDFs from a directory
def chunk_pdf_directory(directory: str):
    """Chunk all PDFs in a directory."""
    from monkey import chunk_directory
    
    chunks = chunk_directory(
        directory,
        extensions=[".pdf"],
        max_tokens=1000,
        recursive=True,  # Search subdirectories
    )
    
    # Group by source file
    by_file = {}
    for chunk in chunks:
        source = chunk.metadata.source or "unknown"
        if source not in by_file:
            by_file[source] = []
        by_file[source].append(chunk)
    
    for source, file_chunks in by_file.items():
        print(f"\n{source}: {len(file_chunks)} chunks")
    
    return chunks


# Method 4: Process PDF bytes (e.g., from API response)
def chunk_pdf_bytes(pdf_bytes: bytes, filename: str = "uploaded.pdf"):
    """Chunk PDF from bytes data."""
    from monkey.parsers import PDFParser
    from monkey import ChunkyMonkey
    
    parser = PDFParser()
    doc = parser.parse_bytes(pdf_bytes, source=filename)
    
    chunker = ChunkyMonkey(max_tokens=1000)
    chunks = chunker.chunk(doc.get_text(), source=filename)
    
    return chunks


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_chunking.py <pdf_file>")
        print("\nThis example requires a PDF backend:")
        print("  pip install pymupdf     # Best quality")
        print("  pip install pypdf       # Lightweight")
        print("  pip install pdfplumber  # Good for tables")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Chunking: {pdf_path}")
    print("=" * 50)
    
    try:
        chunks = chunk_pdf_simple(pdf_path)
        print(f"\n\nTotal chunks: {len(chunks)}")
        
        # Export to JSON
        import json
        output = [c.to_dict() for c in chunks]
        print(f"\nJSON output (first chunk):")
        print(json.dumps(output[0], indent=2)[:500])
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall a PDF backend:")
        print("  pip install pymupdf")
