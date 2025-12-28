"""
Chunky Monkey - Universal semantic text chunking for LLM/RAG applications.

Don't split text. Encapsulate meaning.
"""

__version__ = "0.1.0"

# Core exports - will be populated as modules are implemented
__all__ = [
    "__version__",
    "chunk",
    "ChunkyMonkey",
    "Chunk",
    "ChunkMetadata",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading all modules on import."""
    if name == "Chunk":
        from monkey.core.schema import Chunk
        return Chunk
    if name == "ChunkMetadata":
        from monkey.core.schema import ChunkMetadata
        return ChunkMetadata
    if name == "chunk":
        from monkey.core.chunker import chunk
        return chunk
    if name == "ChunkyMonkey":
        from monkey.core.chunker import ChunkyMonkey
        return ChunkyMonkey
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
