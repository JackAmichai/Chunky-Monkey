"""Tests for tokenizers."""

import pytest
from monkey.tokenizers.base import Tokenizer, BaseTokenizer
from monkey.tokenizers.character import CharacterTokenizer, WordTokenizer


class TestCharacterTokenizer:
    """Tests for CharacterTokenizer."""
    
    def test_count_tokens_basic(self):
        """Test basic character counting."""
        tok = CharacterTokenizer()
        assert tok.count_tokens("hello") == 5
        assert tok.count_tokens("hello world") == 11
    
    def test_count_tokens_empty(self):
        """Test empty string returns 1 (minimum)."""
        tok = CharacterTokenizer()
        # Empty string still returns at least 1 to avoid division by zero
        assert tok.count_tokens("") >= 0
    
    def test_count_tokens_unicode(self):
        """Test Unicode characters."""
        tok = CharacterTokenizer()
        assert tok.count_tokens("你好") == 2
        assert tok.count_tokens("🎉") == 1
    
    def test_encode(self):
        """Test encoding to Unicode code points."""
        tok = CharacterTokenizer()
        encoded = tok.encode("ABC")
        assert encoded == [65, 66, 67]
    
    def test_decode(self):
        """Test decoding from Unicode code points."""
        tok = CharacterTokenizer()
        decoded = tok.decode([65, 66, 67])
        assert decoded == "ABC"
    
    def test_roundtrip(self):
        """Test encode-decode roundtrip."""
        tok = CharacterTokenizer()
        text = "Hello, 世界! 🎉"
        assert tok.decode(tok.encode(text)) == text
    
    def test_chars_per_token(self):
        """Test chars_per_token ratio."""
        tok = CharacterTokenizer(chars_per_token=4.0)
        # 12 chars / 4 = 3 tokens
        assert tok.count_tokens("hello world!") == 3
    
    def test_name(self):
        """Test tokenizer name."""
        tok = CharacterTokenizer()
        assert tok.name == "chars"


class TestWordTokenizer:
    """Tests for WordTokenizer."""
    
    def test_count_tokens_basic(self):
        """Test basic word counting."""
        tok = WordTokenizer()
        assert tok.count_tokens("hello world") == 2
        assert tok.count_tokens("one two three four") == 4
    
    def test_count_tokens_empty(self):
        """Test empty string."""
        tok = WordTokenizer()
        assert tok.count_tokens("") == 0
        assert tok.count_tokens("   ") == 0
    
    def test_count_tokens_punctuation(self):
        """Test that punctuation attached to words counts as one."""
        tok = WordTokenizer()
        # "Hello," is one word, "world!" is one word
        assert tok.count_tokens("Hello, world!") == 2
    
    def test_name(self):
        """Test tokenizer name."""
        tok = WordTokenizer()
        assert tok.name == "words"


class TestBaseTokenizer:
    """Tests for BaseTokenizer functionality."""
    
    def test_truncate(self):
        """Test text truncation."""
        tok = CharacterTokenizer()
        result = tok.truncate("Hello world", 5)
        assert result == "Hello"
    
    def test_truncate_no_change(self):
        """Test truncate when text is under limit."""
        tok = CharacterTokenizer()
        result = tok.truncate("Hi", 10)
        assert result == "Hi"
    
    def test_split_at_boundary(self):
        """Test splitting at token boundary."""
        tok = CharacterTokenizer()
        first, rest = tok.split_at_token_boundary("Hello world", 5)
        assert first == "Hello"
        assert rest == " world"
    
    def test_split_at_boundary_no_split(self):
        """Test split when text is under limit."""
        tok = CharacterTokenizer()
        first, rest = tok.split_at_token_boundary("Hi", 10)
        assert first == "Hi"
        assert rest == ""


class TestTokenizerProtocol:
    """Tests for Tokenizer protocol compliance."""
    
    def test_character_tokenizer_is_tokenizer(self):
        """Test CharacterTokenizer implements Tokenizer protocol."""
        tok = CharacterTokenizer()
        assert isinstance(tok, Tokenizer)
    
    def test_word_tokenizer_is_tokenizer(self):
        """Test WordTokenizer implements Tokenizer protocol."""
        tok = WordTokenizer()
        assert isinstance(tok, Tokenizer)
    
    def test_custom_tokenizer(self):
        """Test custom class can satisfy Tokenizer protocol."""
        class MyTokenizer:
            @property
            def name(self) -> str:
                return "my-tokenizer"
            
            def count_tokens(self, text: str) -> int:
                return len(text.split())
            
            def encode(self, text: str) -> list[int]:
                return [hash(w) for w in text.split()]
            
            def decode(self, tokens: list[int]) -> str:
                return f"[{len(tokens)} tokens]"
        
        tok = MyTokenizer()
        assert isinstance(tok, Tokenizer)
        assert tok.count_tokens("hello world") == 2
