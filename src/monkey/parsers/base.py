"""
Parser protocol and base classes.

Defines the interface for document parsers that extract structure
from different file formats (Markdown, HTML, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Literal, Any, runtime_checkable


@dataclass
class DocumentElement:
    """
    A structural element extracted from a document.
    
    Attributes:
        type: Element type (header, paragraph, code, list, etc.)
        content: The text content of the element
        level: For headers, the level (1-6). None for other types.
        language: For code blocks, the language. None for other types.
        position: (start, end) character offsets in original document
        section_header: The header hierarchy this element belongs to
        metadata: Additional element-specific metadata
    """
    type: Literal["header", "paragraph", "code", "list", "list_item", "blockquote", "table"]
    content: str
    level: int | None = None
    language: str | None = None
    position: tuple[int, int] = (0, 0)
    section_header: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.content)


@dataclass
class ParsedDocument:
    """
    A parsed document containing structural elements.
    
    Attributes:
        elements: List of DocumentElement objects
        source: Original document source/path
        title: Document title (first H1 if available)
        metadata: Document-level metadata
    """
    elements: list[DocumentElement] = field(default_factory=list)
    source: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_text(self) -> str:
        """Get all content as plain text."""
        return "\n\n".join(e.content for e in self.elements)
    
    def get_headers(self) -> list[DocumentElement]:
        """Get all header elements."""
        return [e for e in self.elements if e.type == "header"]
    
    def get_code_blocks(self) -> list[DocumentElement]:
        """Get all code block elements."""
        return [e for e in self.elements if e.type == "code"]


@runtime_checkable
class Parser(Protocol):
    """
    Protocol for document parsers.
    
    Any class with a parse() method returning ParsedDocument works.
    """
    
    def parse(self, text: str) -> ParsedDocument:
        """
        Parse text into structured elements.
        
        Args:
            text: Raw document text
            
        Returns:
            ParsedDocument containing structural elements
        """
        ...


class BaseParser:
    """
    Base class for parsers with common functionality.
    
    Subclasses should override parse().
    """
    
    def parse(self, text: str) -> ParsedDocument:
        """Parse text into structured elements. Override in subclass."""
        raise NotImplementedError("Subclass must implement parse()")
    
    def parse_file(self, path: str) -> ParsedDocument:
        """
        Parse a file.
        
        Args:
            path: Path to file
            
        Returns:
            ParsedDocument with source set to path
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        doc = self.parse(text)
        doc.source = path
        return doc
