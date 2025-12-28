"""
Markdown parser with structure extraction.

Extracts headers, code blocks, lists, and paragraphs from Markdown text.
Propagates section headers to all elements for context preservation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from monkey.parsers.base import BaseParser, ParsedDocument, DocumentElement


# Regex patterns for Markdown elements
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#+)?$', re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r'^```(\w*)\n(.*?)^```', re.MULTILINE | re.DOTALL)
FENCED_CODE_ALT = re.compile(r'^~~~(\w*)\n(.*?)^~~~', re.MULTILINE | re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')
LIST_PATTERN = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', re.MULTILINE)
BLOCKQUOTE_PATTERN = re.compile(r'^>\s+(.+)$', re.MULTILINE)
HORIZONTAL_RULE = re.compile(r'^(?:[-*_]){3,}\s*$', re.MULTILINE)


@dataclass
class MarkdownSection:
    """Tracks the current section hierarchy."""
    headers: list[tuple[int, str]]  # List of (level, text) tuples
    
    def __init__(self) -> None:
        self.headers = []
    
    def update(self, level: int, text: str) -> None:
        """Update section hierarchy with new header."""
        # Remove headers at same or deeper level
        self.headers = [(l, t) for l, t in self.headers if l < level]
        self.headers.append((level, text))
    
    def get_path(self) -> str | None:
        """Get current section path as string."""
        if not self.headers:
            return None
        return " > ".join(t for _, t in self.headers)


class MarkdownParser(BaseParser):
    """
    Parser for Markdown documents.
    
    Extracts:
    - Headers (H1-H6) with hierarchy tracking
    - Fenced code blocks (``` or ~~~) with language detection
    - Lists (bullet and numbered)
    - Blockquotes
    - Paragraphs
    
    Each element gets the current section header path attached,
    enabling chunks to know their document context.
    
    Example:
        parser = MarkdownParser()
        doc = parser.parse('''
        # Chapter 1
        
        Introduction paragraph.
        
        ## Section 1.1
        
        ```python
        print("Hello")
        ```
        ''')
        
        for elem in doc.elements:
            print(f"{elem.type}: {elem.section_header}")
        # header: Chapter 1
        # paragraph: Chapter 1
        # header: Chapter 1 > Section 1.1
        # code: Chapter 1 > Section 1.1
    """
    
    def __init__(
        self, 
        preserve_code_blocks: bool = True,
        extract_lists: bool = True,
    ) -> None:
        """
        Initialize Markdown parser.
        
        Args:
            preserve_code_blocks: Keep code blocks as atomic units
            extract_lists: Extract lists as separate elements
        """
        self.preserve_code_blocks = preserve_code_blocks
        self.extract_lists = extract_lists
    
    def parse(self, text: str) -> ParsedDocument:
        """
        Parse Markdown text into structural elements.
        
        Args:
            text: Markdown content
            
        Returns:
            ParsedDocument with elements and metadata
        """
        if not text or not text.strip():
            return ParsedDocument(elements=[])
        
        elements: list[DocumentElement] = []
        section = MarkdownSection()
        title: str | None = None
        
        # First pass: extract code blocks and replace with placeholders
        code_blocks: dict[str, tuple[str, str, int, int]] = {}
        protected_text = text
        
        for i, match in enumerate(CODE_BLOCK_PATTERN.finditer(text)):
            placeholder = f"__MARKDOWN_CODE_{i}__"
            language = match.group(1) or ""
            code_content = match.group(2)
            code_blocks[placeholder] = (
                match.group(0),  # Full match including fences
                language,
                match.start(),
                match.end()
            )
        
        # Also check for ~~~ fenced blocks
        for i, match in enumerate(FENCED_CODE_ALT.finditer(text)):
            placeholder = f"__MARKDOWN_CODE_ALT_{i}__"
            language = match.group(1) or ""
            code_content = match.group(2)
            code_blocks[placeholder] = (
                match.group(0),
                language,
                match.start(),
                match.end()
            )
        
        # Replace code blocks with placeholders
        for placeholder, (full_match, _, start, end) in sorted(
            code_blocks.items(), 
            key=lambda x: x[1][2], 
            reverse=True
        ):
            protected_text = protected_text[:start] + placeholder + protected_text[end:]
        
        # Process line by line
        lines = protected_text.split('\n')
        current_paragraph: list[str] = []
        current_para_start: int = 0
        position = 0
        
        def flush_paragraph() -> None:
            """Add accumulated paragraph to elements."""
            nonlocal current_paragraph, current_para_start
            if current_paragraph:
                para_text = '\n'.join(current_paragraph).strip()
                if para_text:
                    # Check if it's a code block placeholder
                    if para_text in code_blocks:
                        full_match, language, orig_start, orig_end = code_blocks[para_text]
                        # Extract just the code content (without fences)
                        code_lines = full_match.split('\n')[1:-1]
                        code_content = '\n'.join(code_lines)
                        
                        elements.append(DocumentElement(
                            type="code",
                            content=code_content,
                            language=language or None,
                            position=(orig_start, orig_end),
                            section_header=section.get_path(),
                        ))
                    else:
                        elements.append(DocumentElement(
                            type="paragraph",
                            content=para_text,
                            position=(current_para_start, position),
                            section_header=section.get_path(),
                        ))
                current_paragraph = []
        
        for line in lines:
            line_start = position
            position += len(line) + 1  # +1 for newline
            
            # Check for horizontal rule
            if HORIZONTAL_RULE.match(line):
                flush_paragraph()
                continue
            
            # Check for header
            header_match = HEADER_PATTERN.match(line)
            if header_match:
                flush_paragraph()
                
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                
                # Update section hierarchy
                section.update(level, header_text)
                
                # Track title (first H1)
                if level == 1 and title is None:
                    title = header_text
                
                elements.append(DocumentElement(
                    type="header",
                    content=header_text,
                    level=level,
                    position=(line_start, position),
                    section_header=section.get_path(),
                ))
                continue
            
            # Check for list item
            list_match = LIST_PATTERN.match(line)
            if list_match and self.extract_lists:
                flush_paragraph()
                
                indent = len(list_match.group(1))
                marker = list_match.group(2)
                item_content = list_match.group(3).strip()
                
                elements.append(DocumentElement(
                    type="list_item",
                    content=item_content,
                    position=(line_start, position),
                    section_header=section.get_path(),
                    metadata={"indent": indent, "marker": marker},
                ))
                continue
            
            # Check for blockquote
            blockquote_match = BLOCKQUOTE_PATTERN.match(line)
            if blockquote_match:
                flush_paragraph()
                
                quote_content = blockquote_match.group(1).strip()
                elements.append(DocumentElement(
                    type="blockquote",
                    content=quote_content,
                    position=(line_start, position),
                    section_header=section.get_path(),
                ))
                continue
            
            # Empty line - paragraph break
            if not line.strip():
                flush_paragraph()
                current_para_start = position
                continue
            
            # Regular text - add to current paragraph
            if not current_paragraph:
                current_para_start = line_start
            current_paragraph.append(line)
        
        # Don't forget the last paragraph
        flush_paragraph()
        
        return ParsedDocument(
            elements=elements,
            title=title,
            metadata={"format": "markdown"},
        )
    
    def get_outline(self, text: str) -> list[tuple[int, str]]:
        """
        Extract document outline (headers only).
        
        Args:
            text: Markdown content
            
        Returns:
            List of (level, header_text) tuples
        """
        outline = []
        for match in HEADER_PATTERN.finditer(text):
            level = len(match.group(1))
            header_text = match.group(2).strip()
            outline.append((level, header_text))
        return outline
