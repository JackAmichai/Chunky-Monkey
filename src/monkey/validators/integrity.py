"""
Chunk validation and integrity checking.

Provides functions to validate chunks after creation:
- Token limit enforcement
- Roundtrip integrity (no content loss)
- Dangling reference detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from monkey.core.schema import Chunk
    from monkey.tokenizers.base import Tokenizer


@dataclass
class ValidationError:
    """
    Represents a validation error.
    
    Attributes:
        chunk_id: ID of the problematic chunk
        message: Description of the error
        severity: "error" or "warning"
        details: Additional error details
    """
    chunk_id: str
    message: str
    severity: str = "error"
    details: dict | None = None
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] Chunk {self.chunk_id[:8]}: {self.message}"


@dataclass
class ValidationWarning(ValidationError):
    """A non-fatal validation issue."""
    severity: str = "warning"


# Common pronouns that might indicate dangling references
DANGLING_PRONOUNS = {
    # Personal pronouns (subject)
    "he", "she", "it", "they", "we",
    # Personal pronouns (object)
    "him", "her", "them", "us",
    # Possessive pronouns
    "his", "her", "hers", "its", "their", "theirs", "our", "ours",
    # Demonstrative pronouns
    "this", "that", "these", "those",
    # Relative pronouns that might dangle
    "which", "who", "whom",
}

# Pattern to find sentences starting with pronouns
PRONOUN_START_PATTERN = re.compile(
    r'^(?:' + '|'.join(DANGLING_PRONOUNS) + r')\b',
    re.IGNORECASE
)

# Pattern for reference phrases
REFERENCE_PATTERNS = [
    re.compile(r'\b(?:as mentioned|as noted|as described|as shown|as stated)\b', re.I),
    re.compile(r'\b(?:see above|mentioned above|described above|noted above)\b', re.I),
    re.compile(r'\b(?:the (?:above|previous|aforementioned|following))\b', re.I),
    re.compile(r'\b(?:in the previous|in the last|in the next)\b', re.I),
]


def validate_token_limits(
    chunks: Sequence["Chunk"],
    max_tokens: int,
    tokenizer: "Tokenizer",
) -> list[ValidationError]:
    """
    Validate that no chunk exceeds the token limit.
    
    Args:
        chunks: Chunks to validate
        max_tokens: Maximum allowed tokens per chunk
        tokenizer: Tokenizer to use for counting
        
    Returns:
        List of ValidationError for chunks exceeding the limit
    """
    errors: list[ValidationError] = []
    
    for chunk in chunks:
        token_count = tokenizer.count_tokens(chunk.text)
        
        if token_count > max_tokens:
            errors.append(ValidationError(
                chunk_id=chunk.chunk_id,
                message=f"Exceeds token limit: {token_count} > {max_tokens}",
                details={
                    "token_count": token_count,
                    "max_tokens": max_tokens,
                    "excess": token_count - max_tokens,
                }
            ))
    
    return errors


def validate_roundtrip(
    original: str,
    chunks: Sequence["Chunk"],
    tolerance: float = 0.95,
    overlap_aware: bool = True,
) -> list[ValidationError]:
    """
    Validate that chunks can reconstruct (approximately) the original text.
    
    This catches content loss during chunking.
    
    Args:
        original: Original text before chunking
        chunks: Chunks to validate
        tolerance: Minimum ratio of original characters that must be present (0.0-1.0)
        overlap_aware: If True, account for repeated content from overlap
        
    Returns:
        List of ValidationError if content loss exceeds tolerance
    """
    errors: list[ValidationError] = []
    
    if not chunks:
        if original.strip():
            errors.append(ValidationError(
                chunk_id="N/A",
                message="No chunks produced from non-empty input",
                details={"original_length": len(original)}
            ))
        return errors
    
    # Join all chunks
    reconstructed = " ".join(chunk.text for chunk in chunks)
    
    # Normalize whitespace for comparison
    original_normalized = " ".join(original.split())
    reconstructed_normalized = " ".join(reconstructed.split())
    
    # Calculate coverage
    original_chars = set(enumerate(original_normalized))
    
    # Check what percentage of original is in reconstructed
    # Simple heuristic: compare lengths (overlap will inflate reconstructed)
    original_len = len(original_normalized)
    reconstructed_len = len(reconstructed_normalized)
    
    if original_len == 0:
        return errors
    
    # For overlap-aware comparison, reconstructed should be >= original
    # For non-overlap, they should be similar
    if overlap_aware:
        if reconstructed_len < original_len * tolerance:
            errors.append(ValidationError(
                chunk_id="roundtrip",
                message=f"Content loss detected: reconstructed has {reconstructed_len} chars vs original {original_len}",
                details={
                    "original_length": original_len,
                    "reconstructed_length": reconstructed_len,
                    "ratio": reconstructed_len / original_len,
                    "tolerance": tolerance,
                }
            ))
    else:
        ratio = min(reconstructed_len, original_len) / max(reconstructed_len, original_len)
        if ratio < tolerance:
            errors.append(ValidationError(
                chunk_id="roundtrip",
                message=f"Content mismatch: ratio {ratio:.2%} below tolerance {tolerance:.2%}",
                details={
                    "original_length": original_len,
                    "reconstructed_length": reconstructed_len,
                    "ratio": ratio,
                    "tolerance": tolerance,
                }
            ))
    
    return errors


def flag_dangling_references(
    chunks: Sequence["Chunk"],
) -> list[ValidationWarning]:
    """
    Flag chunks that may have dangling references.
    
    Detects:
    - Chunks starting with pronouns (he, she, it, this, etc.)
    - Reference phrases like "as mentioned above"
    
    These indicate potential context loss where the referent
    might be in a different chunk.
    
    Args:
        chunks: Chunks to analyze
        
    Returns:
        List of ValidationWarning for potential issues
    """
    warnings: list[ValidationWarning] = []
    
    for i, chunk in enumerate(chunks):
        text = chunk.text.strip()
        if not text:
            continue
        
        # Skip first chunk (nothing to reference)
        if i == 0:
            continue
        
        # Check for pronoun at start
        first_sentence = text.split('.')[0] if '.' in text else text
        first_word = first_sentence.split()[0] if first_sentence.split() else ""
        
        if first_word.lower() in DANGLING_PRONOUNS:
            warnings.append(ValidationWarning(
                chunk_id=chunk.chunk_id,
                message=f"Starts with pronoun '{first_word}' - possible dangling reference",
                details={
                    "pronoun": first_word,
                    "first_sentence": first_sentence[:100],
                    "chunk_index": i,
                }
            ))
        
        # Check for reference phrases
        for pattern in REFERENCE_PATTERNS:
            match = pattern.search(text[:200])  # Check beginning of chunk
            if match:
                warnings.append(ValidationWarning(
                    chunk_id=chunk.chunk_id,
                    message=f"Contains reference phrase '{match.group()}' - may need context",
                    details={
                        "phrase": match.group(),
                        "position": match.start(),
                        "chunk_index": i,
                    }
                ))
                break  # One warning per chunk for reference phrases
    
    return warnings


def validate_chunks(
    chunks: Sequence["Chunk"],
    original: str | None = None,
    max_tokens: int | None = None,
    tokenizer: "Tokenizer | None" = None,
    check_references: bool = True,
) -> tuple[list[ValidationError], list[ValidationWarning]]:
    """
    Run all validations on chunks.
    
    Args:
        chunks: Chunks to validate
        original: Original text (for roundtrip check)
        max_tokens: Token limit (for limit check)
        tokenizer: Tokenizer to use
        check_references: Whether to check for dangling references
        
    Returns:
        Tuple of (errors, warnings)
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []
    
    # Token limit validation
    if max_tokens is not None and tokenizer is not None:
        errors.extend(validate_token_limits(chunks, max_tokens, tokenizer))
    
    # Roundtrip validation
    if original is not None:
        errors.extend(validate_roundtrip(original, chunks))
    
    # Dangling reference check
    if check_references:
        warnings.extend(flag_dangling_references(chunks))
    
    return errors, warnings
