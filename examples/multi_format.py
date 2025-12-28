#!/usr/bin/env python3
"""
Example: Chunking multiple file formats.

This example demonstrates chunking different file types
with automatic format detection.

Supported formats:
- Plain text (.txt)
- Markdown (.md)
- PDF (.pdf) - requires: pip install chunky-monkey[pdf]
- HTML (.html) - enhanced: pip install chunky-monkey[html]
- Word (.docx) - requires: pip install chunky-monkey[docx]
- JSON (.json)
- CSV (.csv)
"""

from pathlib import Path

from monkey import chunk_file, chunk_directory


def demo_auto_detection():
    """Demonstrate automatic format detection."""
    
    # Sample files (you would use real files)
    sample_files = [
        "document.txt",
        "README.md",
        "data.json",
        "report.pdf",
        "article.html",
        "spreadsheet.csv",
    ]
    
    for filename in sample_files:
        ext = Path(filename).suffix
        print(f"\n{filename} -> Format detected: {ext}")
        
        # In real usage:
        # chunks = chunk_file(filename, max_tokens=1000)


def chunk_mixed_directory():
    """Chunk all supported files in a directory."""
    from monkey import chunk_directory
    
    # Chunk all supported files
    chunks = chunk_directory(
        "./documents",
        extensions=[".txt", ".md", ".pdf", ".docx"],
        max_tokens=1000,
        recursive=True,
    )
    
    # Group by file type
    by_type = {}
    for chunk in chunks:
        source = chunk.metadata.source or ""
        ext = Path(source).suffix.lower()
        if ext not in by_type:
            by_type[ext] = []
        by_type[ext].append(chunk)
    
    print("\nChunks by file type:")
    for ext, type_chunks in sorted(by_type.items()):
        print(f"  {ext}: {len(type_chunks)} chunks")
    
    return chunks


def chunk_json_data():
    """Example: Chunk JSON data."""
    from monkey.parsers import JSONParser
    from monkey import ChunkyMonkey
    
    # Sample JSON data (e.g., API response)
    json_data = '''
    [
        {
            "id": 1,
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of AI...",
            "tags": ["ml", "ai", "tutorial"]
        },
        {
            "id": 2,
            "title": "Deep Learning Fundamentals",
            "content": "Neural networks are composed of layers...",
            "tags": ["dl", "neural-networks"]
        }
    ]
    '''
    
    parser = JSONParser(flatten=True)
    doc = parser.parse(json_data)
    
    chunker = ChunkyMonkey(max_tokens=200, parser=parser)
    chunks = chunker.chunk(doc.get_text())
    
    print(f"\nJSON parsed into {len(doc.elements)} elements")
    print(f"Chunked into {len(chunks)} chunks")
    
    return chunks


def chunk_csv_data():
    """Example: Chunk CSV data."""
    from monkey.parsers import CSVParser
    from monkey import ChunkyMonkey
    
    # Sample CSV
    csv_data = """name,email,department,notes
John Doe,john@example.com,Engineering,Senior developer with 10 years experience
Jane Smith,jane@example.com,Marketing,Marketing manager leading digital campaigns
Bob Wilson,bob@example.com,Sales,Top performer Q4 2024"""
    
    parser = CSVParser(
        has_header=True,
        row_format="key_value",  # Format as "name: John, email: john@..."
    )
    doc = parser.parse(csv_data)
    
    chunker = ChunkyMonkey(max_tokens=100, parser=parser)
    chunks = chunker.chunk(doc.get_text())
    
    print(f"\nCSV: {doc.metadata['total_rows']} rows")
    print(f"Headers: {doc.metadata['headers']}")
    print(f"Chunked into {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}: {chunk.text[:100]}...")
    
    return chunks


def chunk_html_content():
    """Example: Chunk HTML content."""
    from monkey.parsers import HTMLParser
    from monkey import ChunkyMonkey
    
    html = """
    <html>
    <head><title>Sample Article</title></head>
    <body>
        <h1>Getting Started with Python</h1>
        <p>Python is a versatile programming language.</p>
        
        <h2>Installation</h2>
        <p>Download Python from python.org and follow the installer.</p>
        
        <h2>First Program</h2>
        <pre><code>print("Hello, World!")</code></pre>
        
        <h2>Key Features</h2>
        <ul>
            <li>Easy to learn</li>
            <li>Large ecosystem</li>
            <li>Great for AI/ML</li>
        </ul>
    </body>
    </html>
    """
    
    parser = HTMLParser()
    doc = parser.parse(html)
    
    print(f"\nHTML Title: {doc.title}")
    print(f"Elements extracted: {len(doc.elements)}")
    
    chunker = ChunkyMonkey(max_tokens=200, parser=parser)
    chunks = chunker.chunk(doc.get_text())
    
    print(f"Chunked into {len(chunks)} chunks")
    
    return chunks


if __name__ == "__main__":
    print("=" * 60)
    print("Chunky Monkey - Multi-Format Chunking Demo")
    print("=" * 60)
    
    print("\n1. JSON Data Chunking:")
    chunk_json_data()
    
    print("\n\n2. CSV Data Chunking:")
    chunk_csv_data()
    
    print("\n\n3. HTML Content Chunking:")
    chunk_html_content()
    
    print("\n\nFor PDF and DOCX support, install:")
    print("  pip install chunky-monkey[pdf]")
    print("  pip install chunky-monkey[docx]")
