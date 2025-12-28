"""
Core data structures for Chunky Monkey.

Defines Chunk and ChunkMetadata as immutable dataclasses with
deterministic IDs and optional Pydantic conversion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class ChunkMetadata:
    """
    Metadata attached to each chunk for provenance tracking.
    
    Attributes:
        source: Original document path/name/identifier
        position: (start, end) character offsets in original document
        section_header: Inherited header hierarchy (e.g., "Chapter 1 > Section 1.1")
        parent_chunk_id: ID of parent chunk if this is a sub-chunk
        chunk_index: Sequential index of this chunk in the document
        total_chunks: Total number of chunks in the document (if known)
        custom: User-defined metadata dict for extensibility
    """
    source: str | None = None
    position: tuple[int, int] | None = None
    section_header: str | None = None
    parent_chunk_id: str | None = None
    chunk_index: int | None = None
    total_chunks: int | None = None
    custom: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "source": self.source,
            "position": list(self.position) if self.position else None,
            "section_header": self.section_header,
            "parent_chunk_id": self.parent_chunk_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "custom": self.custom,
        }
    
    def with_update(self, **kwargs: Any) -> ChunkMetadata:
        """Create a new ChunkMetadata with updated fields (immutable update)."""
        current = asdict(self)
        current.update(kwargs)
        # Handle tuple conversion for position
        if "position" in current and isinstance(current["position"], list):
            current["position"] = tuple(current["position"])
        return ChunkMetadata(**current)


def _generate_chunk_id(text: str, position: tuple[int, int] | None) -> str:
    """
    Generate a deterministic chunk ID using SHA-256.
    
    Same text + position always produces the same ID.
    This is critical for:
    - Reproducible embeddings (no vector store drift)
    - Caching and deduplication
    - Debugging and testing
    """
    content = f"{text}|{position}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    """
    A single chunk of text with metadata and token counts.
    
    Attributes:
        text: The chunk content
        chunk_id: Deterministic hash-based identifier
        metadata: Provenance and structural metadata
        token_counts: Dict of tokenizer_name -> count (e.g., {"tiktoken": 150, "chars": 600})
    """
    text: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    token_counts: dict[str, int] = field(default_factory=dict)
    chunk_id: str = field(default="")
    
    def __post_init__(self) -> None:
        """Generate chunk_id if not provided."""
        if not self.chunk_id:
            # Use object.__setattr__ because dataclass is frozen
            object.__setattr__(
                self, 
                "chunk_id", 
                _generate_chunk_id(self.text, self.metadata.position)
            )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "token_counts": self.token_counts,
        }
    
    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_pydantic(self) -> Any:
        """
        Convert to Pydantic model (requires pydantic extra).
        
        Returns:
            PydanticChunk model instance
            
        Raises:
            ImportError: If pydantic is not installed
        """
        try:
            from pydantic import BaseModel
        except ImportError:
            raise ImportError(
                "Pydantic is required for to_pydantic(). "
                "Install with: pip install chunky-monkey[pydantic]"
            )
        
        # Dynamically create Pydantic models
        class PydanticChunkMetadata(BaseModel):
            source: str | None = None
            position: tuple[int, int] | None = None
            section_header: str | None = None
            parent_chunk_id: str | None = None
            chunk_index: int | None = None
            total_chunks: int | None = None
            custom: dict[str, Any] = {}
            
            class Config:
                frozen = True
        
        class PydanticChunk(BaseModel):
            chunk_id: str
            text: str
            metadata: PydanticChunkMetadata
            token_counts: dict[str, int] = {}
            
            class Config:
                frozen = True
        
        return PydanticChunk(
            chunk_id=self.chunk_id,
            text=self.text,
            metadata=PydanticChunkMetadata(**self.metadata.to_dict()),
            token_counts=self.token_counts,
        )
    
    def __len__(self) -> int:
        """Return character length of text."""
        return len(self.text)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Chunk({self.chunk_id[:8]}): {preview!r}"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"Chunk(chunk_id={self.chunk_id!r}, "
            f"text={self.text[:30]!r}..., "
            f"metadata={self.metadata!r}, "
            f"token_counts={self.token_counts!r})"
        )
