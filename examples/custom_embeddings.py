"""
Custom Embeddings Example - Chunky Monkey
==========================================

This example shows how to use your own embedding function for semantic
chunking - useful when you want to use OpenAI embeddings, Cohere,
or any other embedding provider.

Requirements:
    pip install chunky-monkey[semantic]
    
For OpenAI: pip install openai
For Cohere: pip install cohere
"""

import numpy as np
from typing import List

try:
    from monkey.semantic import SemanticChunker
    from monkey.tokenizers import CharacterTokenizer
except ImportError:
    print("Please install with: pip install chunky-monkey[semantic]")
    exit(1)


# =============================================================================
# CUSTOM EMBEDDING FUNCTION EXAMPLES
# =============================================================================

def mock_openai_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Mock OpenAI-style embedding function.
    
    Replace this with actual OpenAI API calls:
    
        from openai import OpenAI
        client = OpenAI()
        
        def openai_embeddings(texts):
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [np.array(e.embedding) for e in response.data]
    """
    # Mock: return random vectors (dimension 1536 like OpenAI)
    return [np.random.rand(1536) for _ in texts]


def mock_cohere_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Mock Cohere-style embedding function.
    
    Replace this with actual Cohere API calls:
    
        import cohere
        co = cohere.Client("YOUR_API_KEY")
        
        def cohere_embeddings(texts):
            response = co.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return [np.array(e) for e in response.embeddings]
    """
    # Mock: return random vectors (dimension 1024 like Cohere)
    return [np.random.rand(1024) for _ in texts]


def simple_tfidf_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Simple TF-IDF-like embeddings without external dependencies.
    
    This is a basic example - not suitable for production,
    but demonstrates the embedding function interface.
    """
    # Build vocabulary from all texts
    vocab = {}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Create sparse vectors
    embeddings = []
    for text in texts:
        vec = np.zeros(len(vocab) + 1)  # +1 for unknown words
        words = text.lower().split()
        for word in words:
            if word in vocab:
                vec[vocab[word]] += 1
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec)
    
    return embeddings


# =============================================================================
# DEMONSTRATION
# =============================================================================

document = """
Neural networks are computational models inspired by biological brains.
They consist of layers of interconnected nodes or neurons. Deep learning 
uses neural networks with many layers to learn complex patterns.

The stock market fluctuated wildly this week. Investors are concerned 
about inflation and rising interest rates. Technology stocks led the 
decline while energy sector showed gains.

Photosynthesis is the process by which plants convert sunlight into 
energy. Chlorophyll in plant cells absorbs light energy. This process 
produces oxygen as a byproduct.

The latest smartphone features include improved cameras and longer 
battery life. Manufacturers are focusing on AI-powered photography.
Foldable phones are gaining popularity in the premium segment.
"""

print("=" * 60)
print("CUSTOM EMBEDDINGS EXAMPLE")
print("=" * 60)

# Example 1: Using mock OpenAI embeddings
print("\n--- Using Mock OpenAI Embeddings ---")

chunker_openai = SemanticChunker(
    embedding_function=mock_openai_embeddings,
    similarity_threshold=0.6,
    max_tokens=300
)

chunks = chunker_openai.chunk(document)
print(f"Chunks with 'OpenAI' embeddings: {len(chunks)}")

# Example 2: Using simple TF-IDF embeddings
print("\n--- Using Simple TF-IDF Embeddings ---")

chunker_tfidf = SemanticChunker(
    embedding_function=simple_tfidf_embeddings,
    similarity_threshold=0.3,  # Lower threshold for sparse vectors
    max_tokens=300
)

chunks_tfidf = chunker_tfidf.chunk(document)
print(f"Chunks with TF-IDF embeddings: {len(chunks_tfidf)}")

for i, chunk in enumerate(chunks_tfidf):
    print(f"\nChunk {i + 1}:")
    print(f"  {chunk.text[:80]}...")

# =============================================================================
# CREATING YOUR OWN EMBEDDING FUNCTION
# =============================================================================

print("\n" + "=" * 60)
print("HOW TO CREATE YOUR OWN EMBEDDING FUNCTION")
print("=" * 60)

print("""
Your embedding function must:

1. Accept: List[str] - a list of text strings
2. Return: List[np.ndarray] - a list of numpy arrays (same length)

Example with OpenAI:

    from openai import OpenAI
    import numpy as np
    
    client = OpenAI()  # Uses OPENAI_API_KEY env var
    
    def embed_with_openai(texts: list[str]) -> list[np.ndarray]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [np.array(item.embedding) for item in response.data]
    
    # Use it
    chunker = SemanticChunker(
        embedding_function=embed_with_openai,
        similarity_threshold=0.5
    )

Example with HuggingFace:

    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-mpnet-base-v2')
    
    def embed_with_hf(texts: list[str]) -> list[np.ndarray]:
        return list(model.encode(texts))
    
    chunker = SemanticChunker(embedding_function=embed_with_hf)
""")

print("\n✓ Custom embeddings example complete!")
