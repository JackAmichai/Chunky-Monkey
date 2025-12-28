"""
Plain text parser (passthrough).

Simply wraps text in a single paragraph element.
Used as the default when no parser is specified.
"""

from __future__ import annotations

from monkey.parsers.base import BaseParser, ParsedDocument, DocumentElement


class PlainTextParser(BaseParser):
    """
    Simple parser for plain text (no structure extraction).
    
    Wraps the entire text as a single paragraph element.
    Useful as a baseline or when document structure doesn't matter.
    
    Example:
        parser = PlainTextParser()
        doc = parser.parse("Hello world. This is text.")
        # doc.elements = [DocumentElement(type="paragraph", content="Hello world...")]
    """
    
    def __init__(self, split_paragraphs: bool = True) -> None:
        """
        Initialize plain text parser.
        
        Args:
            split_paragraphs: If True, split on double newlines into separate paragraphs.
                             If False, treat entire text as one paragraph.
        """
        self.split_paragraphs = split_paragraphs
    
    def parse(self, text: str) -> ParsedDocument:
        """
        Parse plain text.
        
        Args:
            text: Raw text content
            
        Returns:
            ParsedDocument with paragraph elements
        """
        if not text or not text.strip():
            return ParsedDocument(elements=[])
        
        elements: list[DocumentElement] = []
        
        if self.split_paragraphs:
            # Split on double newlines
            paragraphs = text.split("\n\n")
            position = 0
            
            for para in paragraphs:
                para = para.strip()
                if para:
                    # Find actual position in original text
                    start = text.find(para, position)
                    if start == -1:
                        start = position
                    end = start + len(para)
                    
                    elements.append(DocumentElement(
                        type="paragraph",
                        content=para,
                        position=(start, end),
                    ))
                    position = end
        else:
            # Entire text as one paragraph
            elements.append(DocumentElement(
                type="paragraph",
                content=text.strip(),
                position=(0, len(text)),
            ))
        
        return ParsedDocument(elements=elements)
