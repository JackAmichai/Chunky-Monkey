"""
Semantic Chunking Example - Chunky Monkey
==========================================

This example demonstrates semantic (meaning-based) chunking that
detects topic shifts using embedding similarity.

Requirements:
    pip install chunky-monkey[semantic]
    
This installs numpy and sentence-transformers.
"""

try:
    import numpy as np
    from monkey.semantic import SemanticChunker, find_semantic_boundaries
    from monkey.semantic.boundaries import create_sentence_transformer_embedder
    from monkey.tokenizers import CharacterTokenizer
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install with: pip install chunky-monkey[semantic]")
    exit(1)

# Document with clear topic shifts
document = """
The solar system consists of the Sun and the objects that orbit it. 
These include eight planets, their moons, dwarf planets, and countless 
smaller objects like asteroids and comets. The Sun contains 99.86% of 
the system's mass. Jupiter is the largest planet, while Mercury is the 
smallest of the eight planets.

Artificial intelligence has revolutionized many industries. Machine 
learning algorithms can now recognize images, understand speech, and 
generate human-like text. Companies are investing billions in AI 
research. Deep learning models require massive amounts of data and 
computational resources to train effectively.

The Renaissance was a period of cultural rebirth in Europe spanning 
roughly from the 14th to 17th century. It began in Italy and spread 
throughout Europe. Artists like Leonardo da Vinci and Michelangelo 
created masterpieces during this period. The printing press, invented 
by Gutenberg, helped spread new ideas rapidly.

Climate change is affecting ecosystems worldwide. Rising temperatures 
are causing glaciers to melt and sea levels to rise. Many species are 
migrating to new habitats. Scientists predict more extreme weather 
events in the coming decades.
"""

print("=" * 60)
print("SEMANTIC CHUNKING EXAMPLE")
print("=" * 60)

# Create embedding function using sentence-transformers
print("\nLoading embedding model...")
print("(This may take a moment on first run)")

try:
    embedder = create_sentence_transformer_embedder("all-MiniLM-L6-v2")
    print("✓ Model loaded: all-MiniLM-L6-v2")
except Exception as e:
    print(f"Could not load sentence-transformers: {e}")
    print("\nUsing mock embeddings for demonstration...")
    
    # Mock embedder that creates distinct clusters
    def embedder(texts):
        embeddings = []
        for text in texts:
            # Create pseudo-embeddings based on content keywords
            vec = np.random.rand(384)
            if "solar" in text.lower() or "planet" in text.lower():
                vec[0:50] += 1
            elif "artificial" in text.lower() or "learning" in text.lower():
                vec[50:100] += 1
            elif "renaissance" in text.lower() or "art" in text.lower():
                vec[100:150] += 1
            elif "climate" in text.lower() or "temperature" in text.lower():
                vec[150:200] += 1
            embeddings.append(vec / np.linalg.norm(vec))
        return embeddings

# Create semantic chunker
print("\nCreating semantic chunker...")
chunker = SemanticChunker(
    embedding_function=embedder,
    similarity_threshold=0.5,  # Lower = more splits
    max_tokens=500,
    min_sentences_per_chunk=2
)

# Chunk the document
print("\nChunking document based on semantic boundaries...")
chunks = chunker.chunk(document)

print(f"\nDocument has {len(document)} characters")
print(f"Created {len(chunks)} semantic chunks")
print()

for i, chunk in enumerate(chunks):
    print(f"--- Semantic Chunk {i + 1} ---")
    print(f"Tokens: {chunk.token_counts}")
    
    # Detect likely topic
    text_lower = chunk.text.lower()
    if "solar" in text_lower or "planet" in text_lower:
        topic = "🌍 Astronomy"
    elif "artificial" in text_lower or "learning" in text_lower:
        topic = "🤖 AI/ML"
    elif "renaissance" in text_lower or "art" in text_lower:
        topic = "🎨 History/Art"
    elif "climate" in text_lower or "temperature" in text_lower:
        topic = "🌡️ Climate"
    else:
        topic = "📄 General"
    
    print(f"Detected topic: {topic}")
    print(f"Preview: {chunk.text[:100].strip()}...")
    print()

# Compare with naive chunking
print("=" * 60)
print("COMPARISON: SEMANTIC vs NAIVE CHUNKING")
print("=" * 60)

from monkey import chunk as naive_chunk

naive_chunks = naive_chunk(document, max_tokens=200)

print(f"\nNaive chunking (200 tokens): {len(naive_chunks)} chunks")
print(f"Semantic chunking: {len(chunks)} chunks")

print("\nNaive chunks may cut across topics:")
for i, c in enumerate(naive_chunks[:2]):
    print(f"\n  Naive Chunk {i+1}: {c.text[:80]}...")

print("\nSemantic chunks group by topic:")
for i, c in enumerate(chunks[:2]):
    print(f"\n  Semantic Chunk {i+1}: {c.text[:80]}...")

print("\n✓ Semantic chunking example complete!")
