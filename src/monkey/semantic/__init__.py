"""Optional semantic chunking features (requires numpy/sentence-transformers)."""

__all__ = [
    "SemanticChunker",
    "find_semantic_boundaries",
]


def __getattr__(name: str):
    """Lazy import to avoid loading numpy unless needed."""
    if name in ("SemanticChunker", "find_semantic_boundaries"):
        from monkey.semantic.boundaries import SemanticChunker, find_semantic_boundaries
        if name == "SemanticChunker":
            return SemanticChunker
        return find_semantic_boundaries
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
