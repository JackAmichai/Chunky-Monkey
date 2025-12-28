"""
Basic Usage Example - Chunky Monkey
===================================

This example demonstrates the simplest way to use Chunky Monkey
for text chunking with zero configuration.

No extra dependencies required - just the core library.
"""

from monkey import chunk, Chunk

# Sample document text
document = """
The Python programming language was created by Guido van Rossum in 1991. 
It emphasizes code readability and simplicity. Python has become one of 
the most popular programming languages in the world.

Python is widely used in web development, data science, artificial 
intelligence, and automation. Major companies like Google, Netflix, 
and Instagram use Python extensively in their technology stacks.

The language continues to evolve with new features being added regularly. 
Python 3.12 introduced several performance improvements and new syntax 
features that make the language even more powerful and expressive.
"""

# Simple one-liner chunking
# Uses default settings: CharacterTokenizer, 500 max tokens, no overlap
chunks = chunk(document)

print("=" * 60)
print("BASIC CHUNKING EXAMPLE")
print("=" * 60)
print(f"\nOriginal document: {len(document)} characters")
print(f"Number of chunks: {len(chunks)}")
print()

for i, c in enumerate(chunks):
    print(f"--- Chunk {i + 1} ---")
    print(f"ID: {c.chunk_id}")
    print(f"Length: {len(c.text)} chars")
    print(f"Token counts: {c.token_counts}")
    print(f"Preview: {c.text[:100]}...")
    print()

# Customizing parameters
print("=" * 60)
print("CUSTOM PARAMETERS")
print("=" * 60)

# Smaller chunks with overlap for context preservation
small_chunks = chunk(
    document,
    max_tokens=100,      # Smaller chunks
    overlap_tokens=20,   # Repeat last 20 tokens in next chunk
    source="python_article.txt"  # Track source in metadata
)

print(f"\nWith max_tokens=100, overlap=20:")
print(f"Number of chunks: {len(small_chunks)}")

for i, c in enumerate(small_chunks):
    print(f"\nChunk {i + 1}:")
    print(f"  Source: {c.metadata.source}")
    print(f"  Position: {c.metadata.position}")
    print(f"  Index: {c.metadata.chunk_index} of {c.metadata.total_chunks}")

# Converting to different formats
print("\n" + "=" * 60)
print("OUTPUT FORMATS")
print("=" * 60)

first_chunk = chunks[0]

# As dictionary (JSON-serializable)
print("\nAs dict:")
print(first_chunk.to_dict())

# As JSON string
print("\nAs JSON:")
print(first_chunk.to_json(indent=2))

print("\n✓ Basic usage complete!")
