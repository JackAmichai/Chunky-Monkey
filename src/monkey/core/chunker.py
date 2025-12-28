"""
Core text chunking logic.

Implements the TextChunker class that groups sentences into chunks
while respecting token budgets, preserving structure, and maintaining overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Any

from monkey.core.schema import Chunk, ChunkMetadata
from monkey.core.segmentation import Sentence, segment_sentences
from monkey.tokenizers.base import Tokenizer
from monkey.tokenizers.character import CharacterTokenizer


# Pattern to detect code blocks (fenced with ```)
CODE_BLOCK_PATTERN = re.compile(
    r'^```[\w]*\n.*?^```',
    re.MULTILINE | re.DOTALL
)

# Pattern to detect list items
LIST_ITEM_PATTERN = re.compile(
    r'^(?:[-*+]|\d+\.)\s+.+$',
    re.MULTILINE
)


@dataclass
class ChunkingConfig:
    """Configuration for the chunking process."""
    max_tokens: int = 500
    overlap_tokens: int = 0
    preserve_code_blocks: bool = True
    preserve_lists: bool = True
    min_chunk_tokens: int = 50  # Minimum tokens per chunk (avoid tiny chunks)
    hard_split_threshold: float = 0.5  # Hard split if sentence uses > 50% of budget


@dataclass
class TextChunker:
    """
    Main chunking engine that splits text into semantic chunks.
    
    Key features:
    - Never breaks mid-sentence (unless sentence exceeds max_tokens)
    - Preserves code blocks as atomic units
    - Supports configurable overlap between chunks
    - Tracks positions for source mapping
    
    Example:
        chunker = TextChunker(max_tokens=500)
        chunks = chunker.chunk("Long document text...")
        
        # With custom tokenizer
        from monkey.tokenizers import TiktokenTokenizer
        chunker = TextChunker(
            max_tokens=500,
            tokenizer=TiktokenTokenizer(model="gpt-4")
        )
    """
    max_tokens: int = 500
    overlap_tokens: int = 0
    tokenizer: Tokenizer = field(default_factory=CharacterTokenizer)
    preserve_code_blocks: bool = True
    preserve_lists: bool = True
    min_chunk_tokens: int = 50
    source: str | None = None
    
    def _extract_code_blocks(self, text: str) -> list[tuple[str, int, int, str]]:
        """
        Extract code blocks from text.
        
        Returns:
            List of (code_block_text, start, end, placeholder) tuples
        """
        blocks = []
        for i, match in enumerate(CODE_BLOCK_PATTERN.finditer(text)):
            placeholder = f"__CODE_BLOCK_{i}__"
            blocks.append((match.group(), match.start(), match.end(), placeholder))
        return blocks
    
    def _protect_code_blocks(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace code blocks with placeholders to prevent splitting.
        
        Returns:
            (text_with_placeholders, placeholder_to_code_map)
        """
        if not self.preserve_code_blocks:
            return text, {}
        
        blocks = self._extract_code_blocks(text)
        protected_text = text
        replacements = {}
        
        # Replace in reverse order to maintain positions
        for code, start, end, placeholder in reversed(blocks):
            protected_text = protected_text[:start] + placeholder + protected_text[end:]
            replacements[placeholder] = code
        
        return protected_text, replacements
    
    def _restore_code_blocks(self, text: str, replacements: dict[str, str]) -> str:
        """Restore code blocks from placeholders."""
        result = text
        for placeholder, code in replacements.items():
            result = result.replace(placeholder, code)
        return result
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured tokenizer."""
        return self.tokenizer.count_tokens(text)
    
    def _hard_split(self, text: str, max_tokens: int) -> list[str]:
        """
        Last resort: split oversized text at token boundaries.
        
        Used when a single sentence exceeds max_tokens.
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= max_tokens:
            return [text]
        
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        
        return chunks
    
    def _group_sentences_into_chunks(
        self, 
        sentences: list[Sentence],
        section_header: str | None = None
    ) -> list[Chunk]:
        """
        Group sentences into chunks respecting token budget.
        
        Algorithm:
        1. Add sentences to current chunk until budget exceeded
        2. When budget exceeded, close chunk and start new one
        3. Apply overlap by repeating ending sentences
        """
        if not sentences:
            return []
        
        chunks: list[Chunk] = []
        current_sentences: list[Sentence] = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence.text)
            
            # Handle oversized sentences
            if sentence_tokens > self.max_tokens:
                # Close current chunk if not empty
                if current_sentences:
                    chunks.append(self._create_chunk(
                        current_sentences, 
                        section_header,
                        len(chunks)
                    ))
                    current_sentences = []
                    current_tokens = 0
                
                # Hard split the oversized sentence
                hard_chunks = self._hard_split(sentence.text, self.max_tokens)
                for i, hard_chunk in enumerate(hard_chunks):
                    chunks.append(Chunk(
                        text=hard_chunk,
                        metadata=ChunkMetadata(
                            source=self.source,
                            position=(sentence.start, sentence.end),
                            section_header=section_header,
                            chunk_index=len(chunks),
                        ),
                        token_counts={self.tokenizer.name: self._count_tokens(hard_chunk)}
                    ))
                continue
            
            # Check if adding this sentence exceeds budget
            if current_tokens + sentence_tokens > self.max_tokens and current_sentences:
                # Close current chunk
                chunks.append(self._create_chunk(
                    current_sentences,
                    section_header,
                    len(chunks)
                ))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_sentences)
                current_sentences = overlap_sentences
                current_tokens = sum(self._count_tokens(s.text) for s in current_sentences)
            
            # Add sentence to current chunk
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_sentences:
            chunks.append(self._create_chunk(
                current_sentences,
                section_header,
                len(chunks)
            ))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: list[Sentence]) -> list[Sentence]:
        """Get sentences to repeat for overlap."""
        if self.overlap_tokens <= 0 or not sentences:
            return []
        
        overlap: list[Sentence] = []
        overlap_tokens = 0
        
        # Walk backwards through sentences
        for sentence in reversed(sentences):
            sentence_tokens = self._count_tokens(sentence.text)
            if overlap_tokens + sentence_tokens > self.overlap_tokens:
                break
            overlap.insert(0, sentence)
            overlap_tokens += sentence_tokens
        
        return overlap
    
    def _create_chunk(
        self, 
        sentences: list[Sentence],
        section_header: str | None,
        index: int
    ) -> Chunk:
        """Create a Chunk from a list of sentences."""
        text = " ".join(s.text for s in sentences)
        
        # Calculate position span
        start = sentences[0].start if sentences else 0
        end = sentences[-1].end if sentences else 0
        
        return Chunk(
            text=text,
            metadata=ChunkMetadata(
                source=self.source,
                position=(start, end),
                section_header=section_header,
                chunk_index=index,
            ),
            token_counts={self.tokenizer.name: self._count_tokens(text)}
        )
    
    def chunk(
        self, 
        text: str, 
        section_header: str | None = None
    ) -> list[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            section_header: Optional header to attach to all chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        # Protect code blocks from splitting
        protected_text, code_replacements = self._protect_code_blocks(text)
        
        # Segment into sentences
        sentences = segment_sentences(protected_text)
        
        # Restore code blocks in sentence text
        if code_replacements:
            restored_sentences = []
            for s in sentences:
                restored_text = self._restore_code_blocks(s.text, code_replacements)
                restored_sentences.append(Sentence(
                    text=restored_text,
                    start=s.start,
                    end=s.end
                ))
            sentences = restored_sentences
        
        # Group sentences into chunks
        chunks = self._group_sentences_into_chunks(sentences, section_header)
        
        # Update total_chunks count
        final_chunks = []
        for chunk in chunks:
            updated_metadata = chunk.metadata.with_update(total_chunks=len(chunks))
            final_chunks.append(Chunk(
                text=chunk.text,
                metadata=updated_metadata,
                token_counts=chunk.token_counts
            ))
        
        return final_chunks
    
    def chunk_file(self, path: str) -> list[Chunk]:
        """
        Chunk a file.
        
        Args:
            path: Path to file
            
        Returns:
            List of Chunk objects
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        self.source = path
        return self.chunk(text)


class ChunkyMonkey:
    """
    High-level chunking interface with full configuration.
    
    Example:
        from monkey import ChunkyMonkey
        from monkey.tokenizers import TiktokenTokenizer
        from monkey.parsers import MarkdownParser
        
        chunker = ChunkyMonkey(
            tokenizer=TiktokenTokenizer(model="gpt-4"),
            parser=MarkdownParser(),
            max_tokens=500,
            overlap_tokens=50,
            validate=True
        )
        
        chunks = chunker.chunk(document)
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        parser: Any | None = None,  # Parser type, but avoid circular import
        max_tokens: int = 500,
        overlap_tokens: int = 0,
        preserve_code_blocks: bool = True,
        validate: bool = True,
        source: str | None = None,
    ) -> None:
        """
        Initialize ChunkyMonkey.
        
        Args:
            tokenizer: Tokenizer to use (default: CharacterTokenizer)
            parser: Document parser (default: None, plain text)
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Tokens to repeat between chunks
            preserve_code_blocks: Keep code blocks as atomic units
            validate: Run validation after chunking
            source: Source identifier for metadata
        """
        self.tokenizer = tokenizer or CharacterTokenizer()
        self.parser = parser
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.preserve_code_blocks = preserve_code_blocks
        self.validate = validate
        self.source = source
        
        self._chunker = TextChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer=self.tokenizer,
            preserve_code_blocks=preserve_code_blocks,
            source=source,
        )
    
    def chunk(self, text: str) -> list[Chunk]:
        """
        Chunk text with optional parsing and validation.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of validated Chunk objects
        """
        # If we have a parser, use it to extract structure
        if self.parser is not None:
            return self._chunk_with_parser(text)
        
        # Simple text chunking
        chunks = self._chunker.chunk(text)
        
        # Validate if enabled
        if self.validate:
            self._validate_chunks(chunks)
        
        return chunks
    
    def _chunk_with_parser(self, text: str) -> list[Chunk]:
        """Chunk using parser for structure extraction."""
        parsed = self.parser.parse(text)
        all_chunks: list[Chunk] = []
        
        for element in parsed.elements:
            # Get section context
            section_header = getattr(element, 'section_header', None)
            
            # Chunk element content
            element_chunks = self._chunker.chunk(element.content, section_header)
            all_chunks.extend(element_chunks)
        
        # Update indices
        final_chunks = []
        for i, chunk in enumerate(all_chunks):
            updated_metadata = chunk.metadata.with_update(
                chunk_index=i,
                total_chunks=len(all_chunks),
                source=self.source
            )
            final_chunks.append(Chunk(
                text=chunk.text,
                metadata=updated_metadata,
                token_counts=chunk.token_counts
            ))
        
        if self.validate:
            self._validate_chunks(final_chunks)
        
        return final_chunks
    
    def _validate_chunks(self, chunks: list[Chunk]) -> None:
        """Run validation on chunks."""
        from monkey.validators import validate_token_limits
        
        violations = validate_token_limits(chunks, self.max_tokens, self.tokenizer)
        if violations:
            # Log warning but don't fail
            import warnings
            warnings.warn(f"Chunks exceed token limit: {violations}")
    
    def chunk_file(self, path: str) -> list[Chunk]:
        """
        Chunk a file.
        
        Args:
            path: Path to file
            
        Returns:
            List of Chunk objects
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        self.source = path
        self._chunker.source = path
        return self.chunk(text)


def chunk(
    text: str,
    max_tokens: int = 500,
    tokenizer: Tokenizer | None = None,
    overlap_tokens: int = 0,
    source: str | None = None,
) -> list[Chunk]:
    """
    Simple one-liner chunking function.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        tokenizer: Tokenizer to use (default: CharacterTokenizer)
        overlap_tokens: Tokens to repeat between chunks
        source: Source identifier for metadata
        
    Returns:
        List of Chunk objects
        
    Example:
        from monkey import chunk
        
        chunks = chunk("Long document...", max_tokens=500)
        for c in chunks:
            print(c.text)
    """
    chunker = TextChunker(
        max_tokens=max_tokens,
        tokenizer=tokenizer or CharacterTokenizer(),
        overlap_tokens=overlap_tokens,
        source=source,
    )
    return chunker.chunk(text)
