"""Tests for sentence segmentation."""

import pytest
from monkey.core.segmentation import (
    segment_sentences,
    Sentence,
    split_into_paragraphs,
    count_words,
)


class TestSegmentSentences:
    """Tests for segment_sentences function."""
    
    def test_simple_sentences(self):
        """Test basic sentence splitting."""
        text = "Hello world. This is a test."
        sentences = segment_sentences(text)
        assert len(sentences) == 2
        assert sentences[0].text == "Hello world."
        assert sentences[1].text == "This is a test."
    
    def test_question_and_exclamation(self):
        """Test splitting on ? and !"""
        text = "Is this working? Yes! It works."
        sentences = segment_sentences(text)
        assert len(sentences) == 3
    
    def test_abbreviation_mr(self):
        """Test that Mr. doesn't cause split."""
        text = "Mr. Smith went home. He was tired."
        sentences = segment_sentences(text)
        assert len(sentences) == 2
        assert "Mr. Smith" in sentences[0].text
    
    def test_abbreviation_dr(self):
        """Test that Dr. doesn't cause split."""
        text = "Dr. Jones is here. She will see you now."
        sentences = segment_sentences(text)
        assert len(sentences) == 2
        assert "Dr. Jones" in sentences[0].text
    
    def test_decimal_numbers(self):
        """Test that decimal numbers don't cause split."""
        text = "The value is 3.14159. This is pi."
        sentences = segment_sentences(text)
        assert len(sentences) == 2
        assert "3.14159" in sentences[0].text
    
    def test_multiple_decimals(self):
        """Test multiple decimal numbers."""
        text = "Prices are $10.99 and $20.50. Very affordable."
        sentences = segment_sentences(text)
        assert len(sentences) == 2
    
    def test_ellipsis(self):
        """Test ellipsis handling."""
        text = "Wait... What happened? I don't know."
        sentences = segment_sentences(text)
        # Ellipsis should not cause multiple splits
        assert len(sentences) >= 2
    
    def test_empty_string(self):
        """Test empty input."""
        assert segment_sentences("") == []
        assert segment_sentences("   ") == []
    
    def test_single_sentence(self):
        """Test single sentence without ending punctuation."""
        text = "Hello world"
        sentences = segment_sentences(text)
        assert len(sentences) == 1
        assert sentences[0].text == "Hello world"
    
    def test_position_tracking(self):
        """Test that positions are tracked correctly."""
        text = "First. Second."
        sentences = segment_sentences(text)
        
        # First sentence should start at beginning
        assert sentences[0].start == 0
        # Second sentence should start after "First. "
        assert sentences[1].start > sentences[0].end
    
    def test_unicode(self):
        """Test Unicode text."""
        text = "你好世界。这是测试。"
        sentences = segment_sentences(text)
        # Chinese uses 。 for period, may not split correctly
        # but shouldn't crash
        assert len(sentences) >= 1
    
    def test_urls(self):
        """Test that URLs don't cause incorrect splits."""
        text = "Visit https://example.com/page. It's great."
        sentences = segment_sentences(text)
        assert len(sentences) == 2
        assert "https://example.com" in sentences[0].text


class TestSentence:
    """Tests for Sentence dataclass."""
    
    def test_len(self):
        """Test __len__ returns text length."""
        s = Sentence(text="Hello", start=0, end=5)
        assert len(s) == 5
    
    def test_str(self):
        """Test __str__ returns text."""
        s = Sentence(text="Hello", start=0, end=5)
        assert str(s) == "Hello"


class TestSplitIntoParagraphs:
    """Tests for paragraph splitting."""
    
    def test_basic_paragraphs(self):
        """Test splitting on double newlines."""
        text = "First paragraph.\n\nSecond paragraph."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 2
        assert paragraphs[0][0] == "First paragraph."
        assert paragraphs[1][0] == "Second paragraph."
    
    def test_multiple_newlines(self):
        """Test handling of multiple blank lines."""
        text = "First.\n\n\n\nSecond."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 2
    
    def test_single_paragraph(self):
        """Test single paragraph."""
        text = "Just one paragraph."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 1


class TestCountWords:
    """Tests for word counting."""
    
    def test_basic(self):
        """Test basic word count."""
        assert count_words("hello world") == 2
    
    def test_empty(self):
        """Test empty string."""
        assert count_words("") == 0
    
    def test_punctuation(self):
        """Test that punctuation doesn't affect count."""
        assert count_words("Hello, world!") == 2
