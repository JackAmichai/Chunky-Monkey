"""
Unicode-aware sentence segmentation.

Handles edge cases like abbreviations, numbers, URLs, and ellipsis
to avoid breaking sentences at incorrect boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Sentence:
    """
    A single sentence with position information.
    
    Attributes:
        text: The sentence content (whitespace-trimmed)
        start: Character offset in original text (inclusive)
        end: Character offset in original text (exclusive)
    """
    text: str
    start: int
    end: int
    
    def __len__(self) -> int:
        return len(self.text)
    
    def __str__(self) -> str:
        return self.text


# Common abbreviations that shouldn't trigger sentence breaks
ABBREVIATIONS = {
    # Titles
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "rev", "hon",
    # Academic
    "ph", "m", "b", "d",  # Ph.D., M.A., B.S., D.Phil.
    # Common
    "vs", "etc", "al", "eg", "ie", "cf", "viz", "approx",
    "inc", "ltd", "corp", "co", "llc",
    # Months (abbreviated)
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    # Measurements
    "ft", "in", "lb", "oz", "pt", "qt", "gal", "mi", "km", "cm", "mm", "kg", "mg",
    # Other
    "no", "nos", "vol", "vols", "dept", "est", "fig", "figs", "govt",
    "min", "max", "avg", "approx", "ca", "c",
    # Technology
    "www", "http", "https", "ftp",
}

# Build abbreviation pattern
ABBREV_PATTERN = "|".join(re.escape(a) for a in sorted(ABBREVIATIONS, key=len, reverse=True))

# Patterns that look like sentence endings but aren't
FALSE_ENDINGS = [
    # Abbreviations followed by period - must be word-bounded
    rf"(?:^|(?<=\s))(?:{ABBREV_PATTERN})\.",
    # Single UPPERCASE letters followed by period (initials like "J. Smith")
    # Must be preceded by word boundary or start of string
    r"(?:^|\s)[A-Z]\.",
    # Numbers with decimals (3.14, 100.00)
    r"\d+\.\d+",
    # URLs - exclude trailing sentence punctuation (.!?) when followed by space
    r"https?://[^\s]*[^\s.!?]",
    r"www\.[^\s]*[^\s.!?]",
    # File extensions - must be attached to filename (no space before dot)
    r"[a-zA-Z0-9_-]+\.[a-z]{2,4}(?=\s|$)",
    # Ellipsis
    r"\.{2,}",
]

# Compile the false endings pattern - case insensitive for abbreviations
FALSE_ENDING_PATTERN = re.compile(
    "|".join(f"({p})" for p in FALSE_ENDINGS),
    re.MULTILINE | re.IGNORECASE
)

# Sentence-ending punctuation
SENTENCE_ENDERS = ".!?"

# Pattern to find potential sentence boundaries
# Looks for .!? followed by space and uppercase letter
SENTENCE_BOUNDARY_PATTERN = re.compile(
    r'([.!?])'                    # Capture the punctuation
    r'(\s+)'                      # Followed by whitespace
    r'(?=[A-Z"\'\(\[0-9])',       # Lookahead for uppercase letter, quote, bracket, or number
    re.MULTILINE
)


def _is_false_ending(text: str, pos: int) -> bool:
    """
    Check if a period at position `pos` is a false sentence ending.
    
    Args:
        text: Full text
        pos: Position of the period/punctuation
        
    Returns:
        True if this is likely NOT a real sentence ending
    """
    # Get context around the position
    start = max(0, pos - 50)
    end = min(len(text), pos + 10)
    context = text[start:end]
    rel_pos = pos - start
    
    # Check for common false endings
    for match in FALSE_ENDING_PATTERN.finditer(context):
        if match.start() <= rel_pos < match.end():
            return True
    
    # Check for abbreviations (case-insensitive)
    # Look back from the period to find the word
    word_start = rel_pos
    while word_start > 0 and context[word_start - 1].isalpha():
        word_start -= 1
    
    word = context[word_start:rel_pos].lower()
    if word in ABBREVIATIONS:
        return True
    
    return False


def segment_sentences(text: str) -> list[Sentence]:
    """
    Split text into sentences with position tracking.
    
    Handles:
    - Abbreviations (Mr., Dr., etc.)
    - Decimal numbers (3.14159)
    - URLs (https://example.com)
    - Ellipsis (Wait... what?)
    - Initials (J. F. Kennedy)
    
    Args:
        text: Text to segment
        
    Returns:
        List of Sentence objects with text and positions
        
    Example:
        >>> sentences = segment_sentences("Mr. Smith went home. He was tired.")
        >>> [s.text for s in sentences]
        ['Mr. Smith went home.', 'He was tired.']
    """
    if not text or not text.strip():
        return []
    
    sentences: list[Sentence] = []
    current_start = 0
    
    # Find all potential sentence boundaries
    for match in SENTENCE_BOUNDARY_PATTERN.finditer(text):
        punct_pos = match.start()
        
        # Check if this is a false ending
        if _is_false_ending(text, punct_pos):
            continue
        
        # This looks like a real sentence boundary
        # End position is right after the punctuation (include it in sentence)
        end_pos = match.start() + 1  # Include the punctuation mark
        
        # Extract sentence text
        sentence_text = text[current_start:end_pos].strip()
        
        if sentence_text:
            # Find actual start (skip leading whitespace in original)
            actual_start = current_start
            while actual_start < len(text) and text[actual_start].isspace():
                actual_start += 1
            
            sentences.append(Sentence(
                text=sentence_text,
                start=actual_start,
                end=end_pos
            ))
        
        # Move past the whitespace to start of next sentence
        current_start = match.end()
    
    # Handle remaining text (last sentence without ending punctuation)
    remaining = text[current_start:].strip()
    if remaining:
        actual_start = current_start
        while actual_start < len(text) and text[actual_start].isspace():
            actual_start += 1
        
        sentences.append(Sentence(
            text=remaining,
            start=actual_start,
            end=len(text)
        ))
    
    return sentences


def segment_sentences_iter(text: str) -> Iterator[Sentence]:
    """
    Generator version of segment_sentences for memory efficiency.
    
    Args:
        text: Text to segment
        
    Yields:
        Sentence objects one at a time
    """
    yield from segment_sentences(text)


def split_into_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into paragraphs (double newline separated).
    
    Args:
        text: Text to split
        
    Returns:
        List of (paragraph_text, start, end) tuples
    """
    paragraphs: list[tuple[str, int, int]] = []
    
    # Split on double newlines
    pattern = re.compile(r'\n\s*\n')
    
    current_start = 0
    for match in pattern.finditer(text):
        para_text = text[current_start:match.start()].strip()
        if para_text:
            paragraphs.append((para_text, current_start, match.start()))
        current_start = match.end()
    
    # Handle last paragraph
    remaining = text[current_start:].strip()
    if remaining:
        paragraphs.append((remaining, current_start, len(text)))
    
    return paragraphs


def count_words(text: str) -> int:
    """
    Count words in text (simple whitespace split).
    
    Args:
        text: Text to count
        
    Returns:
        Number of words
    """
    return len(text.split())
