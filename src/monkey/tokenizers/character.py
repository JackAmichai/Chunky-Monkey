"""
Character-based tokenizer (fallback with zero dependencies).

Approximates tokens as characters. Not accurate for real LLMs,
but useful for testing and when no tokenizer is available.
"""

from __future__ import annotations

from monkey.tokenizers.base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """
    Simple character-based tokenizer.
    
    1 character ≈ 1 token (rough approximation).
    
    Use cases:
    - Testing without dependencies
    - Quick prototyping
    - When exact token counts don't matter
    
    Note: Real LLMs use subword tokenization (BPE, SentencePiece).
    A word like "tokenization" might be 3 tokens in GPT-4 but 12 characters.
    For production, use TiktokenTokenizer or HuggingFaceTokenizer.
    
    Example:
        tok = CharacterTokenizer()
        tok.count_tokens("Hello world")  # Returns 11
    """
    
    _name: str = "chars"
    
    def __init__(self, chars_per_token: float = 1.0) -> None:
        """
        Initialize character tokenizer.
        
        Args:
            chars_per_token: Characters per token ratio.
                Default 1.0 (1 char = 1 token).
                Use ~4.0 to approximate English word tokenization.
        """
        self.chars_per_token = chars_per_token
    
    def count_tokens(self, text: str) -> int:
        """Count tokens as characters / chars_per_token."""
        return max(1, int(len(text) / self.chars_per_token))
    
    def encode(self, text: str) -> list[int]:
        """Encode each character as its Unicode code point."""
        return [ord(c) for c in text]
    
    def decode(self, tokens: list[int]) -> str:
        """Decode Unicode code points back to string."""
        return "".join(chr(t) for t in tokens)


class WordTokenizer(BaseTokenizer):
    """
    Simple whitespace-based word tokenizer.
    
    Splits on whitespace and counts words as tokens.
    More accurate than character counting for English text,
    but still an approximation.
    
    Example:
        tok = WordTokenizer()
        tok.count_tokens("Hello world")  # Returns 2
    """
    
    _name: str = "words"
    
    def count_tokens(self, text: str) -> int:
        """Count whitespace-separated words."""
        words = text.split()
        return len(words) if words else 0
    
    def encode(self, text: str) -> list[int]:
        """
        Encode words as sequential indices.
        
        Note: This is a stateless encoding - not suitable for
        actual NLP tasks, just for counting/splitting.
        """
        words = text.split()
        return list(range(len(words)))
    
    def decode(self, tokens: list[int]) -> str:
        """
        Decode is not fully reversible for WordTokenizer.
        
        Since we lose the actual words during encoding,
        this returns a placeholder. Use CharacterTokenizer
        if you need reversible encode/decode.
        """
        return f"[{len(tokens)} words]"
