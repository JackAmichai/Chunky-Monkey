"""
Tokenizer protocol and base classes.

Defines the interface that all tokenizers must implement,
enabling pluggable tokenization for different LLM providers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """
    Protocol for tokenizers - any class with these methods works.
    
    Using Protocol (structural subtyping) instead of ABC means
    any compatible class works without inheritance.
    
    Example:
        class MyTokenizer:
            def count_tokens(self, text: str) -> int:
                return len(text.split())
            
            def encode(self, text: str) -> list[int]:
                return [ord(c) for c in text]
            
            def decode(self, tokens: list[int]) -> str:
                return "".join(chr(t) for t in tokens)
        
        # This works even though MyTokenizer doesn't inherit from Tokenizer
        tok: Tokenizer = MyTokenizer()
    """
    
    @property
    def name(self) -> str:
        """Unique identifier for this tokenizer (e.g., 'tiktoken-cl100k', 'chars')."""
        ...
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.
        
        Args:
            text: The text to tokenize
            
        Returns:
            Number of tokens
        """
        ...
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: The text to encode
            
        Returns:
            List of token IDs
        """
        ...
    
    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text string
        """
        ...


class BaseTokenizer:
    """
    Base class with common functionality for tokenizers.
    
    Subclasses should override encode() and decode().
    count_tokens() will work automatically based on encode().
    """
    
    _name: str = "base"
    
    @property
    def name(self) -> str:
        """Unique identifier for this tokenizer."""
        return self._name
    
    def count_tokens(self, text: str) -> int:
        """Count tokens by encoding and measuring length."""
        return len(self.encode(text))
    
    def encode(self, text: str) -> list[int]:
        """Encode text to tokens. Override in subclass."""
        raise NotImplementedError("Subclass must implement encode()")
    
    def decode(self, tokens: list[int]) -> str:
        """Decode tokens to text. Override in subclass."""
        raise NotImplementedError("Subclass must implement decode()")
    
    def truncate(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within max_tokens.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])
    
    def split_at_token_boundary(self, text: str, max_tokens: int) -> tuple[str, str]:
        """
        Split text at a token boundary.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens for first part
            
        Returns:
            Tuple of (first_part, remainder)
        """
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text, ""
        
        first_part = self.decode(tokens[:max_tokens])
        remainder = self.decode(tokens[max_tokens:])
        return first_part, remainder
