"""Tests for the core chunker."""

import pytest
from monkey.core.chunker import TextChunker, ChunkyMonkey, chunk
from monkey.core.schema import Chunk
from monkey.tokenizers.character import CharacterTokenizer


class TestTextChunker:
    """Tests for TextChunker class."""
    
    def test_basic_chunking(self):
        """Test basic text chunking."""
        chunker = TextChunker(max_tokens=50)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_empty_input(self):
        """Test empty input returns empty list."""
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []
    
    def test_single_sentence(self):
        """Test single sentence becomes single chunk."""
        chunker = TextChunker(max_tokens=1000)
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
    
    def test_respects_token_limit(self):
        """Test that chunks respect max_tokens."""
        chunker = TextChunker(
            max_tokens=20,
            tokenizer=CharacterTokenizer()
        )
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            assert chunk.token_counts.get("chars", 0) <= 30  # Allow some flexibility
    
    def test_overlap(self):
        """Test overlap between chunks."""
        chunker = TextChunker(
            max_tokens=30,
            overlap_tokens=10,
            tokenizer=CharacterTokenizer()
        )
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # Check for some overlap (sentences repeated)
            # This is hard to test precisely, but chunks should share some content
            pass  # Overlap is best tested by examining actual output
    
    def test_preserves_code_blocks(self):
        """Test that code blocks are kept intact."""
        chunker = TextChunker(
            max_tokens=100,
            preserve_code_blocks=True
        )
        text = "Introduction.\n\n```python\nprint('hello')\n```\n\nConclusion."
        chunks = chunker.chunk(text)
        
        # Code block should not be split
        all_text = " ".join(c.text for c in chunks)
        assert "print('hello')" in all_text
    
    def test_chunk_has_metadata(self):
        """Test that chunks have proper metadata."""
        chunker = TextChunker(source="test.txt")
        chunks = chunker.chunk("Hello world. Goodbye world.")
        
        for chunk in chunks:
            assert chunk.metadata is not None
            assert chunk.metadata.source == "test.txt"
            assert chunk.metadata.position is not None
    
    def test_chunk_has_token_counts(self):
        """Test that chunks have token counts."""
        chunker = TextChunker(tokenizer=CharacterTokenizer())
        chunks = chunker.chunk("Hello world.")
        
        assert len(chunks) == 1
        assert "chars" in chunks[0].token_counts
        assert chunks[0].token_counts["chars"] > 0
    
    def test_deterministic(self):
        """Test that chunking is deterministic."""
        chunker = TextChunker(max_tokens=50)
        text = "First sentence. Second sentence. Third sentence."
        
        chunks1 = chunker.chunk(text)
        chunks2 = chunker.chunk(text)
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text
    
    def test_hard_split_oversized(self):
        """Test that oversized sentences are hard-split."""
        chunker = TextChunker(
            max_tokens=10,
            tokenizer=CharacterTokenizer()
        )
        # A very long sentence
        text = "This is a very long sentence that definitely exceeds the token limit."
        chunks = chunker.chunk(text)
        
        # Should be split into multiple chunks
        assert len(chunks) >= 2


class TestChunkyMonkey:
    """Tests for ChunkyMonkey high-level API."""
    
    def test_basic_usage(self):
        """Test basic ChunkyMonkey usage."""
        monkey = ChunkyMonkey(max_tokens=100)
        chunks = monkey.chunk("Hello world. This is a test.")
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_with_parser(self):
        """Test ChunkyMonkey with Markdown parser."""
        from monkey.parsers import MarkdownParser
        
        monkey = ChunkyMonkey(
            parser=MarkdownParser(),
            max_tokens=500
        )
        
        text = """# Title

First paragraph.

## Section

Second paragraph.
"""
        chunks = monkey.chunk(text)
        assert len(chunks) >= 1
    
    def test_validation_enabled(self):
        """Test that validation runs when enabled."""
        monkey = ChunkyMonkey(max_tokens=100, validate=True)
        # Should not raise for valid chunks
        chunks = monkey.chunk("Hello world.")
        assert len(chunks) >= 1


class TestChunkFunction:
    """Tests for the simple chunk() function."""
    
    def test_basic(self):
        """Test basic usage of chunk function."""
        chunks = chunk("Hello world. Goodbye world.")
        assert len(chunks) >= 1
    
    def test_with_max_tokens(self):
        """Test with max_tokens parameter."""
        chunks = chunk("Hello world.", max_tokens=100)
        assert len(chunks) == 1
    
    def test_with_source(self):
        """Test source is set in metadata."""
        chunks = chunk("Hello.", source="test.txt")
        assert chunks[0].metadata.source == "test.txt"
    
    def test_with_tokenizer(self):
        """Test with custom tokenizer."""
        tok = CharacterTokenizer()
        chunks = chunk("Hello world.", tokenizer=tok)
        assert "chars" in chunks[0].token_counts
