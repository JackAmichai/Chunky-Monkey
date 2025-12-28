"""Tests for core schema (Chunk, ChunkMetadata)."""

import pytest
from monkey.core.schema import Chunk, ChunkMetadata, _generate_chunk_id


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""
    
    def test_default_values(self):
        """Test that defaults are sensible."""
        meta = ChunkMetadata()
        assert meta.source is None
        assert meta.position is None
        assert meta.section_header is None
        assert meta.parent_chunk_id is None
        assert meta.chunk_index is None
        assert meta.total_chunks is None
        assert meta.custom == {}
    
    def test_with_values(self):
        """Test creating metadata with values."""
        meta = ChunkMetadata(
            source="test.md",
            position=(100, 200),
            section_header="Chapter 1",
            chunk_index=0,
            total_chunks=10,
        )
        assert meta.source == "test.md"
        assert meta.position == (100, 200)
        assert meta.section_header == "Chapter 1"
        assert meta.chunk_index == 0
        assert meta.total_chunks == 10
    
    def test_to_dict(self):
        """Test JSON-serializable dict conversion."""
        meta = ChunkMetadata(
            source="test.md",
            position=(100, 200),
        )
        d = meta.to_dict()
        assert d["source"] == "test.md"
        assert d["position"] == [100, 200]  # Tuple becomes list for JSON
    
    def test_with_update(self):
        """Test immutable update pattern."""
        meta = ChunkMetadata(source="test.md")
        updated = meta.with_update(chunk_index=5)
        
        # Original unchanged
        assert meta.chunk_index is None
        # New has update
        assert updated.chunk_index == 5
        assert updated.source == "test.md"
    
    def test_frozen(self):
        """Test that metadata is immutable."""
        meta = ChunkMetadata()
        with pytest.raises(AttributeError):
            meta.source = "new_source"


class TestChunk:
    """Tests for Chunk dataclass."""
    
    def test_basic_creation(self):
        """Test creating a simple chunk."""
        chunk = Chunk(text="Hello world")
        assert chunk.text == "Hello world"
        assert chunk.chunk_id  # ID is generated
        assert len(chunk.chunk_id) == 16  # SHA256 truncated
    
    def test_deterministic_id(self):
        """Test that same text+position produces same ID."""
        meta = ChunkMetadata(position=(0, 10))
        chunk1 = Chunk(text="Hello", metadata=meta)
        chunk2 = Chunk(text="Hello", metadata=meta)
        assert chunk1.chunk_id == chunk2.chunk_id
    
    def test_different_text_different_id(self):
        """Test that different text produces different ID."""
        chunk1 = Chunk(text="Hello")
        chunk2 = Chunk(text="World")
        assert chunk1.chunk_id != chunk2.chunk_id
    
    def test_to_dict(self):
        """Test JSON-serializable dict conversion."""
        chunk = Chunk(
            text="Test content",
            metadata=ChunkMetadata(source="test.md"),
            token_counts={"chars": 12},
        )
        d = chunk.to_dict()
        assert d["text"] == "Test content"
        assert d["metadata"]["source"] == "test.md"
        assert d["token_counts"]["chars"] == 12
        assert "chunk_id" in d
    
    def test_to_json(self):
        """Test JSON serialization."""
        chunk = Chunk(text="Test")
        json_str = chunk.to_json()
        assert '"text": "Test"' in json_str
        assert '"chunk_id":' in json_str
    
    def test_len(self):
        """Test __len__ returns text length."""
        chunk = Chunk(text="Hello")
        assert len(chunk) == 5
    
    def test_str(self):
        """Test string representation."""
        chunk = Chunk(text="Hello world, this is a test")
        s = str(chunk)
        assert "Chunk(" in s
        assert "Hello world" in s
    
    def test_repr(self):
        """Test detailed representation."""
        chunk = Chunk(text="Test")
        r = repr(chunk)
        assert "Chunk(" in r
        assert "chunk_id=" in r
        assert "metadata=" in r


class TestGenerateChunkId:
    """Tests for deterministic ID generation."""
    
    def test_consistency(self):
        """Test ID is consistent for same input."""
        id1 = _generate_chunk_id("test", (0, 4))
        id2 = _generate_chunk_id("test", (0, 4))
        assert id1 == id2
    
    def test_different_text(self):
        """Test different text produces different ID."""
        id1 = _generate_chunk_id("test1", (0, 5))
        id2 = _generate_chunk_id("test2", (0, 5))
        assert id1 != id2
    
    def test_different_position(self):
        """Test different position produces different ID."""
        id1 = _generate_chunk_id("test", (0, 4))
        id2 = _generate_chunk_id("test", (100, 104))
        assert id1 != id2
    
    def test_length(self):
        """Test ID length is 16 characters."""
        chunk_id = _generate_chunk_id("test", None)
        assert len(chunk_id) == 16
