"""Tokenizer backends for different LLM providers."""

from monkey.tokenizers.base import Tokenizer
from monkey.tokenizers.character import CharacterTokenizer

__all__ = [
    "Tokenizer",
    "CharacterTokenizer",
]

# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "TiktokenTokenizer":
        from monkey.tokenizers.tiktoken_ import TiktokenTokenizer
        return TiktokenTokenizer
    if name == "HuggingFaceTokenizer":
        from monkey.tokenizers.huggingface_ import HuggingFaceTokenizer
        return HuggingFaceTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
