"""
HuggingFace transformers tokenizer integration.

Provides token counting for open-source models (Llama, Mistral, etc.).
Requires: pip install chunky-monkey[huggingface]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monkey.tokenizers.base import BaseTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class HuggingFaceTokenizer(BaseTokenizer):
    """
    HuggingFace transformers tokenizer for open-source models.
    
    Supports any model on HuggingFace Hub with AutoTokenizer.
    
    Example:
        # From model name (downloads tokenizer)
        tok = HuggingFaceTokenizer(model="meta-llama/Llama-2-7b-hf")
        count = tok.count_tokens("Hello, how are you?")
        
        # From existing tokenizer instance
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tok = HuggingFaceTokenizer(tokenizer=hf_tok)
    
    Raises:
        ImportError: If transformers is not installed
    """
    
    def __init__(
        self, 
        model: str | None = None,
        tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast | None" = None,
        trust_remote_code: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize HuggingFace tokenizer.
        
        Args:
            model: HuggingFace model name/path (e.g., "meta-llama/Llama-2-7b-hf")
            tokenizer: Existing tokenizer instance (overrides model)
            trust_remote_code: Whether to trust remote code for custom tokenizers
            **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained
            
        Note: Either model or tokenizer must be provided.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFaceTokenizer. "
                "Install with: pip install chunky-monkey[huggingface]"
            )
        
        if tokenizer is not None:
            self._tokenizer = tokenizer
            self._model_name = getattr(tokenizer, "name_or_path", "custom")
        elif model is not None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model, 
                trust_remote_code=trust_remote_code,
                **kwargs
            )
            self._model_name = model
        else:
            raise ValueError("Either 'model' or 'tokenizer' must be provided")
    
    @property
    def name(self) -> str:
        """Return tokenizer identifier."""
        # Clean up model name for display
        model_short = self._model_name.split("/")[-1] if "/" in self._model_name else self._model_name
        return f"hf-{model_short}"
    
    @property
    def model_name(self) -> str:
        """Return the full model name."""
        return self._model_name
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._tokenizer.vocab_size
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using the HuggingFace tokenizer."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))
    
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add model-specific special tokens
            
        Returns:
            List of token IDs
        """
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            tokens: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into subword strings.
        
        Useful for debugging and understanding tokenization.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token strings
        """
        return self._tokenizer.tokenize(text)
