"""
Markdown Chunking Example - Chunky Monkey
==========================================

This example demonstrates how to chunk Markdown documents while
preserving structure (headers, code blocks, lists) in metadata.

No extra dependencies required - just the core library.
"""

from monkey import ChunkyMonkey
from monkey.parsers import MarkdownParser

# A sample Markdown document
markdown_document = """
# Getting Started with Python

Python is a versatile programming language that's perfect for beginners 
and experts alike. This guide will help you get started.

## Installation

### Windows

Download the installer from python.org and run it. Make sure to check 
"Add Python to PATH" during installation.

```powershell
# Verify installation
python --version
```

### macOS

Use Homebrew for the easiest installation experience:

```bash
brew install python3
python3 --version
```

### Linux

Most distributions include Python. If not:

```bash
sudo apt update
sudo apt install python3 python3-pip
```

## Your First Program

Create a file called `hello.py` with the following content:

```python
# hello.py - Your first Python program

def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    message = greet("World")
    print(message)
```

Run it with `python hello.py` and you should see "Hello, World!" printed.

## Next Steps

Now that you have Python installed, you can:

- Explore the standard library
- Learn about virtual environments
- Start building projects

Happy coding!
"""

print("=" * 60)
print("MARKDOWN CHUNKING EXAMPLE")
print("=" * 60)

# Create parser and chunker
parser = MarkdownParser()

# First, let's see what the parser extracts
print("\n--- Document Structure ---")
doc = parser.parse(markdown_document)

print(f"Title: {doc.title}")
print(f"Elements: {len(doc.elements)}")
print("\nOutline:")
for level, text in parser.get_outline(markdown_document):
    indent = "  " * (level - 1)
    print(f"{indent}{'#' * level} {text}")

print("\nCode blocks found:", len(doc.get_code_blocks()))
for cb in doc.get_code_blocks():
    print(f"  - {cb.language or 'plain'}: {cb.content[:30]}...")

# Now chunk with structure awareness
print("\n--- Chunking with Structure ---")

chunker = ChunkyMonkey(
    parser=parser,
    max_tokens=200,
    overlap_tokens=30,
    preserve_code_blocks=True,
    validate=True
)

chunks = chunker.chunk(markdown_document)

print(f"\nCreated {len(chunks)} chunks:")
print()

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(f"  Section: {chunk.metadata.section_header or '(no section)'}")
    print(f"  Position: {chunk.metadata.position}")
    print(f"  Tokens: {chunk.token_counts}")
    
    # Check if contains code
    if "```" in chunk.text or "def " in chunk.text:
        print("  Contains: code")
    
    preview = chunk.text[:80].replace('\n', ' ').strip()
    print(f"  Preview: {preview}...")
    print()

# Export for RAG pipeline
print("=" * 60)
print("EXPORT FOR RAG")
print("=" * 60)

import json

# Convert all chunks to JSON-serializable format
export_data = {
    "source": "getting_started.md",
    "title": doc.title,
    "chunks": [chunk.to_dict() for chunk in chunks]
}

print("\nJSON export (truncated):")
print(json.dumps(export_data, indent=2)[:500] + "...")

print("\n✓ Markdown chunking example complete!")
