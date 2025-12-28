"""Tests for Markdown parser."""

import pytest
from monkey.parsers.markdown import MarkdownParser
from monkey.parsers.base import DocumentElement


class TestMarkdownParser:
    """Tests for MarkdownParser."""
    
    def test_basic_parsing(self):
        """Test basic Markdown parsing."""
        parser = MarkdownParser()
        doc = parser.parse("# Title\n\nSome content.")
        
        assert len(doc.elements) >= 2
        assert doc.title == "Title"
    
    def test_header_extraction(self):
        """Test header extraction at all levels."""
        parser = MarkdownParser()
        text = """# H1
## H2
### H3
#### H4
##### H5
###### H6"""
        doc = parser.parse(text)
        
        headers = doc.get_headers()
        assert len(headers) == 6
        
        levels = [h.level for h in headers]
        assert levels == [1, 2, 3, 4, 5, 6]
    
    def test_code_block_extraction(self):
        """Test fenced code block extraction."""
        parser = MarkdownParser()
        text = """# Code Example

```python
print("hello")
```

More text.
"""
        doc = parser.parse(text)
        
        code_blocks = doc.get_code_blocks()
        assert len(code_blocks) == 1
        assert code_blocks[0].language == "python"
        assert "print" in code_blocks[0].content
    
    def test_code_block_no_language(self):
        """Test code block without language specified."""
        parser = MarkdownParser()
        text = """```
plain code
```"""
        doc = parser.parse(text)
        
        code_blocks = doc.get_code_blocks()
        assert len(code_blocks) == 1
        assert code_blocks[0].language is None or code_blocks[0].language == ""
    
    def test_list_extraction(self):
        """Test list item extraction."""
        parser = MarkdownParser()
        text = """# List

- Item 1
- Item 2
- Item 3
"""
        doc = parser.parse(text)
        
        list_items = [e for e in doc.elements if e.type == "list_item"]
        assert len(list_items) == 3
    
    def test_numbered_list(self):
        """Test numbered list extraction."""
        parser = MarkdownParser()
        text = """1. First
2. Second
3. Third
"""
        doc = parser.parse(text)
        
        list_items = [e for e in doc.elements if e.type == "list_item"]
        assert len(list_items) == 3
    
    def test_section_header_propagation(self):
        """Test that section headers propagate to content."""
        parser = MarkdownParser()
        text = """# Chapter 1

First paragraph.

## Section 1.1

Second paragraph.
"""
        doc = parser.parse(text)
        
        # Find paragraphs
        paragraphs = [e for e in doc.elements if e.type == "paragraph"]
        
        # First paragraph should have "Chapter 1" in section
        if paragraphs:
            assert paragraphs[0].section_header is not None
    
    def test_empty_input(self):
        """Test empty input."""
        parser = MarkdownParser()
        doc = parser.parse("")
        assert doc.elements == []
    
    def test_blockquote(self):
        """Test blockquote extraction."""
        parser = MarkdownParser()
        text = "> This is a quote."
        doc = parser.parse(text)
        
        quotes = [e for e in doc.elements if e.type == "blockquote"]
        assert len(quotes) == 1
        assert "This is a quote" in quotes[0].content
    
    def test_get_outline(self):
        """Test outline extraction."""
        parser = MarkdownParser()
        text = """# Title
## Chapter 1
### Section 1.1
## Chapter 2
"""
        outline = parser.get_outline(text)
        
        assert len(outline) == 4
        assert outline[0] == (1, "Title")
        assert outline[1] == (2, "Chapter 1")


class TestParsedDocument:
    """Tests for ParsedDocument."""
    
    def test_get_text(self):
        """Test get_text concatenates all content."""
        parser = MarkdownParser()
        doc = parser.parse("# Title\n\nContent here.")
        
        text = doc.get_text()
        assert "Title" in text
        assert "Content" in text
    
    def test_get_headers(self):
        """Test get_headers filters correctly."""
        parser = MarkdownParser()
        doc = parser.parse("# Header\n\nParagraph\n\n## Another")
        
        headers = doc.get_headers()
        assert all(h.type == "header" for h in headers)
    
    def test_metadata(self):
        """Test document metadata."""
        parser = MarkdownParser()
        doc = parser.parse("# Test")
        
        assert doc.metadata.get("format") == "markdown"
