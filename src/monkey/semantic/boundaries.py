"""
Semantic chunking with embedding-based boundary detection.

Uses cosine similarity between sentence embeddings to detect
topic shifts and create semantically coherent chunks.

Requires: pip install chunky-monkey[semantic]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from monkey.core.schema import Chunk, ChunkMetadata
from monkey.core.segmentation import Sentence, segment_sentences
from monkey.tokenizers.base import Tokenizer
from monkey.tokenizers.character import CharacterTokenizer


# Type alias for embedding functions
EmbeddingFunction = Callable[[list[str]], list["np.ndarray"]]


def _ensure_numpy():
    """Ensure numpy is available."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError(
            "numpy is required for semantic chunking. "
            "Install with: pip install chunky-monkey[semantic]"
        )


def cosine_similarity(vec1: "np.ndarray", vec2: "np.ndarray") -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    np = _ensure_numpy()
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def find_semantic_boundaries(
    sentences: list[str],
    embeddings: list["np.ndarray"],
    threshold: float = 0.5,
    min_chunk_size: int = 2,
) -> list[int]:
    """
    Find indices where topic shifts occur.
    
    Compares consecutive sentence embeddings and marks boundaries
    where similarity drops below threshold.
    
    Args:
        sentences: List of sentence texts
        embeddings: Corresponding embeddings (same length as sentences)
        threshold: Similarity threshold (0-1). Lower = more splits.
        min_chunk_size: Minimum sentences per chunk
        
    Returns:
        List of indices where splits should occur
    """
    if len(sentences) != len(embeddings):
        raise ValueError("sentences and embeddings must have same length")
    
    if len(sentences) <= 1:
        return []
    
    boundaries: list[int] = []
    last_boundary = 0
    
    for i in range(1, len(sentences)):
        # Check minimum chunk size
        if i - last_boundary < min_chunk_size:
            continue
        
        similarity = cosine_similarity(embeddings[i - 1], embeddings[i])
        
        if similarity < threshold:
            boundaries.append(i)
            last_boundary = i
    
    return boundaries


@dataclass
class SemanticChunker:
    """
    Chunker that uses semantic similarity for boundary detection.
    
    Instead of splitting at fixed token counts, this chunker:
    1. Embeds each sentence
    2. Calculates similarity between consecutive sentences
    3. Splits where similarity drops (topic shift)
    4. Respects max_tokens as a hard limit
    
    Example:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        chunker = SemanticChunker(
            embedding_function=model.encode,
            similarity_threshold=0.5,
            max_tokens=500
        )
        
        chunks = chunker.chunk(text)
    """
    embedding_function: EmbeddingFunction
    similarity_threshold: float = 0.5
    max_tokens: int = 500
    tokenizer: Tokenizer = field(default_factory=CharacterTokenizer)
    min_sentences_per_chunk: int = 2
    source: str | None = None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.tokenizer.count_tokens(text)
    
    def _embed_sentences(self, sentences: list[str]) -> list["np.ndarray"]:
        """Embed sentences using the provided function."""
        return self.embedding_function(sentences)
    
    def _create_chunk(
        self,
        sentences: list[Sentence],
        section_header: str | None,
        index: int,
    ) -> Chunk:
        """Create a Chunk from sentences."""
        text = " ".join(s.text for s in sentences)
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
        section_header: str | None = None,
    ) -> list[Chunk]:
        """
        Chunk text using semantic boundaries.
        
        Args:
            text: Text to chunk
            section_header: Optional header for metadata
            
        Returns:
            List of semantically coherent Chunk objects
        """
        np = _ensure_numpy()
        
        if not text or not text.strip():
            return []
        
        # Segment into sentences
        sentences = segment_sentences(text)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [self._create_chunk(sentences, section_header, 0)]
        
        # Embed all sentences
        sentence_texts = [s.text for s in sentences]
        embeddings = self._embed_sentences(sentence_texts)
        
        # Find semantic boundaries
        boundaries = find_semantic_boundaries(
            sentence_texts,
            embeddings,
            threshold=self.similarity_threshold,
            min_chunk_size=self.min_sentences_per_chunk,
        )
        
        # Create chunks based on boundaries
        chunks: list[Chunk] = []
        start_idx = 0
        
        for boundary in boundaries:
            chunk_sentences = sentences[start_idx:boundary]
            
            # Check token limit - may need to split further
            chunk_text = " ".join(s.text for s in chunk_sentences)
            if self._count_tokens(chunk_text) > self.max_tokens:
                # Split this chunk further at token boundaries
                sub_chunks = self._split_at_token_limit(chunk_sentences, section_header, len(chunks))
                chunks.extend(sub_chunks)
            else:
                chunks.append(self._create_chunk(chunk_sentences, section_header, len(chunks)))
            
            start_idx = boundary
        
        # Handle remaining sentences
        if start_idx < len(sentences):
            remaining = sentences[start_idx:]
            chunk_text = " ".join(s.text for s in remaining)
            
            if self._count_tokens(chunk_text) > self.max_tokens:
                sub_chunks = self._split_at_token_limit(remaining, section_header, len(chunks))
                chunks.extend(sub_chunks)
            else:
                chunks.append(self._create_chunk(remaining, section_header, len(chunks)))
        
        # Update total_chunks
        final_chunks = []
        for i, chunk in enumerate(chunks):
            updated_metadata = chunk.metadata.with_update(
                chunk_index=i,
                total_chunks=len(chunks)
            )
            final_chunks.append(Chunk(
                text=chunk.text,
                metadata=updated_metadata,
                token_counts=chunk.token_counts
            ))
        
        return final_chunks
    
    def _split_at_token_limit(
        self,
        sentences: list[Sentence],
        section_header: str | None,
        start_index: int,
    ) -> list[Chunk]:
        """Split sentences into chunks respecting token limit."""
        chunks: list[Chunk] = []
        current: list[Sentence] = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence.text)
            
            if current_tokens + sentence_tokens > self.max_tokens and current:
                chunks.append(self._create_chunk(current, section_header, start_index + len(chunks)))
                current = []
                current_tokens = 0
            
            current.append(sentence)
            current_tokens += sentence_tokens
        
        if current:
            chunks.append(self._create_chunk(current, section_header, start_index + len(chunks)))
        
        return chunks


def create_sentence_transformer_embedder(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingFunction:
    """
    Create an embedding function using sentence-transformers.
    
    Args:
        model_name: Name of the sentence-transformers model
        
    Returns:
        Embedding function compatible with SemanticChunker
        
    Example:
        embedder = create_sentence_transformer_embedder()
        chunker = SemanticChunker(embedding_function=embedder)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. "
            "Install with: pip install chunky-monkey[semantic]"
        )
    
    model = SentenceTransformer(model_name)
    
    def embed(texts: list[str]) -> list["np.ndarray"]:
        return list(model.encode(texts))
    
    return embed
