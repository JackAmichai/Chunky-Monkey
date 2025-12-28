"""
OpenAI tiktoken tokenizer integration.

Provides accurate token counting for OpenAI models (GPT-4, GPT-3.5, etc.).
Requires: pip install chunky-monkey[tiktoken]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monkey.tokenizers.base import BaseTokenizer

if TYPE_CHECKING:
    import tiktoken


# Model to encoding mapping
MODEL_ENCODINGS = {
    # GPT-4 and GPT-4 Turbo
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-vision-preview": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    # GPT-3.5
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-35-turbo": "cl100k_base",  # Azure naming
    # Embeddings
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # Legacy
    "text-davinci-003": "p50k_base",
    "davinci": "r50k_base",
}


class TiktokenTokenizer(BaseTokenizer):
    """
    OpenAI tiktoken tokenizer for accurate GPT token counting.
    
    Supports all OpenAI models with their correct encodings.
    
    Example:
        tok = TiktokenTokenizer(model="gpt-4")
        count = tok.count_tokens("Hello, how are you?")  # Accurate count
        
        # Or specify encoding directly
        tok = TiktokenTokenizer(encoding="cl100k_base")
    
    Raises:
        ImportError: If tiktoken is not installed
    """
    
    def __init__(
        self, 
        model: str | None = None, 
        encoding: str | None = None
    ) -> None:
        """
        Initialize tiktoken tokenizer.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
            encoding: Direct encoding name (e.g., "cl100k_base")
            
        Note: If both are provided, encoding takes precedence.
              If neither is provided, defaults to cl100k_base (GPT-4).
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer. "
                "Install with: pip install chunky-monkey[tiktoken]"
            )
        
        self._tiktoken = tiktoken
        
        # Determine encoding
        if encoding:
            self._encoding_name = encoding
        elif model:
            self._encoding_name = MODEL_ENCODINGS.get(model, "cl100k_base")
        else:
            self._encoding_name = "cl100k_base"
        
        self._model = model
        self._encoder: tiktoken.Encoding = tiktoken.get_encoding(self._encoding_name)
    
    @property
    def name(self) -> str:
        """Return tokenizer identifier."""
        if self._model:
            return f"tiktoken-{self._model}"
        return f"tiktoken-{self._encoding_name}"
    
    @property
    def encoding_name(self) -> str:
        """Return the tiktoken encoding name."""
        return self._encoding_name
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken encoding."""
        return len(self._encoder.encode(text))
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._encoder.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._encoder.decode(tokens)
    
    def encode_with_special(
        self, 
        text: str, 
        allowed_special: set[str] | str = "all"
    ) -> list[int]:
        """
        Encode text, allowing special tokens.
        
        Args:
            text: Text to encode
            allowed_special: Set of special tokens to allow, or "all"
            
        Returns:
            List of token IDs including special tokens
        """
        return self._encoder.encode(text, allowed_special=allowed_special)
