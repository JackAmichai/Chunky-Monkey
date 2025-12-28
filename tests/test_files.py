"""Tests for file chunking utilities."""

import pytest
import tempfile
from pathlib import Path

from monkey.files import (
    get_parser_for_file,
    chunk_file,
    is_supported,
    SUPPORTED_EXTENSIONS,
)
from monkey.parsers.plaintext import PlainTextParser
from monkey.parsers.markdown import MarkdownParser


class TestGetParserForFile:
    """Test parser detection."""
    
    def test_txt_file(self):
        """Test plaintext parser for .txt files."""
        parser = get_parser_for_file("document.txt")
        assert isinstance(parser, PlainTextParser)
    
    def test_md_file(self):
        """Test markdown parser for .md files."""
        parser = get_parser_for_file("README.md")
        assert isinstance(parser, MarkdownParser)
    
    def test_markdown_extension(self):
        """Test markdown parser for .markdown files."""
        parser = get_parser_for_file("doc.markdown")
        assert isinstance(parser, MarkdownParser)
    
    def test_unknown_extension(self):
        """Test fallback to plaintext for unknown extensions."""
        parser = get_parser_for_file("file.xyz")
        assert isinstance(parser, PlainTextParser)
    
    def test_code_files(self):
        """Test code files use plaintext parser."""
        for ext in [".py", ".js", ".java", ".cpp"]:
            parser = get_parser_for_file(f"code{ext}")
            assert isinstance(parser, PlainTextParser)


class TestChunkFile:
    """Test file chunking."""
    
    def test_chunk_txt_file(self):
        """Test chunking a text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is sentence one. This is sentence two. This is sentence three.")
            f.flush()
            
            chunks = chunk_file(f.name, max_tokens=50)
            
            assert len(chunks) >= 1
            assert all(c.text for c in chunks)
        
        Path(f.name).unlink()
    
    def test_chunk_md_file(self):
        """Test chunking a markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nThis is a paragraph.\n\n## Section\n\nAnother paragraph.")
            f.flush()
            
            chunks = chunk_file(f.name, max_tokens=100)
            
            assert len(chunks) >= 1
        
        Path(f.name).unlink()
    
    def test_chunk_json_file(self):
        """Test chunking a JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('[{"name": "John"}, {"name": "Jane"}]')
            f.flush()
            
            chunks = chunk_file(f.name, max_tokens=100)
            
            assert len(chunks) >= 1
        
        Path(f.name).unlink()
    
    def test_chunk_csv_file(self):
        """Test chunking a CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nJohn,30\nJane,25")
            f.flush()
            
            chunks = chunk_file(f.name, max_tokens=100)
            
            assert len(chunks) >= 1
        
        Path(f.name).unlink()
    
    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            chunk_file("nonexistent_file.txt")
    
    def test_custom_max_tokens(self):
        """Test custom max_tokens parameter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Word " * 100)  # 100 words
            f.flush()
            
            chunks = chunk_file(f.name, max_tokens=20)
            
            # Should create multiple chunks with small token limit
            assert len(chunks) > 1
        
        Path(f.name).unlink()
    
    def test_source_in_metadata(self):
        """Test that file source is preserved in chunk metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Some content here.")
            f.flush()
            
            chunks = chunk_file(f.name)
            
            assert len(chunks) >= 1
            assert chunks[0].metadata.source is not None
        
        Path(f.name).unlink()


class TestIsSupported:
    """Test file support detection."""
    
    def test_supported_extensions(self):
        """Test known supported extensions."""
        assert is_supported("doc.txt")
        assert is_supported("doc.md")
        assert is_supported("doc.json")
        assert is_supported("doc.csv")
        assert is_supported("doc.html")
        assert is_supported("doc.pdf")
        assert is_supported("doc.docx")
    
    def test_unsupported_extensions(self):
        """Test unknown extensions."""
        # Unknown extensions default to supported (plaintext)
        assert is_supported("file.xyz") is False  # Not in EXTENSION_PARSERS
    
    def test_case_insensitive(self):
        """Test case-insensitive extension matching."""
        assert is_supported("DOC.TXT")
        assert is_supported("Doc.Md")


class TestSupportedExtensions:
    """Test supported extensions list."""
    
    def test_common_extensions_supported(self):
        """Test that common extensions are in the list."""
        common = [".txt", ".md", ".json", ".csv", ".html", ".pdf", ".docx", ".py"]
        for ext in common:
            assert ext in SUPPORTED_EXTENSIONS
