"""
HTML document parser.

Extracts text and structure from HTML documents.
Uses BeautifulSoup when available, falls back to regex-based parsing.
"""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path
from typing import Literal

from monkey.parsers.base import BaseParser, ParsedDocument, DocumentElement


class HTMLParser(BaseParser):
    """
    Parse HTML documents into structured elements.
    
    Extracts headers, paragraphs, code blocks, lists, and tables
    while stripping HTML markup.
    
    Example:
        >>> parser = HTMLParser()
        >>> doc = parser.parse("<h1>Title</h1><p>Hello world</p>")
        >>> doc.elements[0].type
        'header'
        >>> doc.elements[0].content
        'Title'
    """
    
    # HTML tags that indicate headers
    HEADER_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    
    # Tags to treat as code blocks
    CODE_TAGS = {"pre", "code"}
    
    # Tags to skip entirely
    SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}
    
    # Block-level tags
    BLOCK_TAGS = {
        "p", "div", "section", "article", "main", "aside", "nav",
        "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "pre", "blockquote", "table", "tr", "td", "th",
        "form", "fieldset", "figure", "figcaption", "details", "summary"
    }
    
    def __init__(
        self,
        use_beautifulsoup: bool = True,
        extract_links: bool = False,
        extract_images: bool = False,
        preserve_whitespace: bool = False,
    ):
        """
        Initialize HTML parser.
        
        Args:
            use_beautifulsoup: Use BeautifulSoup if available (better parsing)
            extract_links: Include link URLs in output
            extract_images: Include image alt text/src in output
            preserve_whitespace: Preserve original whitespace
        """
        self.use_beautifulsoup = use_beautifulsoup
        self.extract_links = extract_links
        self.extract_images = extract_images
        self.preserve_whitespace = preserve_whitespace
        self._bs4_available = None
    
    def _check_bs4(self) -> bool:
        """Check if BeautifulSoup is available."""
        if self._bs4_available is None:
            try:
                from bs4 import BeautifulSoup
                self._bs4_available = True
            except ImportError:
                self._bs4_available = False
        return self._bs4_available
    
    def parse(self, text: str) -> ParsedDocument:
        """
        Parse HTML text into structured elements.
        
        Args:
            text: HTML content
            
        Returns:
            ParsedDocument with extracted elements
        """
        if self.use_beautifulsoup and self._check_bs4():
            return self._parse_with_bs4(text)
        else:
            return self._parse_with_regex(text)
    
    def parse_file(self, path: str | Path, encoding: str = "utf-8") -> ParsedDocument:
        """
        Parse an HTML file.
        
        Args:
            path: Path to HTML file
            encoding: File encoding
            
        Returns:
            ParsedDocument with extracted elements
        """
        path = Path(path)
        text = path.read_text(encoding=encoding)
        doc = self.parse(text)
        doc.source = str(path)
        return doc
    
    def _parse_with_bs4(self, html: str) -> ParsedDocument:
        """Parse using BeautifulSoup."""
        from bs4 import BeautifulSoup, NavigableString, Comment
        
        soup = BeautifulSoup(html, "html.parser")
        elements = []
        position = 0
        title = None
        
        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Remove script/style tags
        for tag in soup.find_all(self.SKIP_TAGS):
            tag.decompose()
        
        # Process body or entire document
        body = soup.find("body") or soup
        
        current_section = None
        
        for element in body.descendants:
            if isinstance(element, (NavigableString, Comment)):
                continue
            
            if element.name in self.SKIP_TAGS:
                continue
            
            tag_name = element.name.lower() if element.name else ""
            
            # Handle headers
            if tag_name in self.HEADER_TAGS:
                text = element.get_text(strip=True)
                if text:
                    level = int(tag_name[1])
                    current_section = text
                    elements.append(DocumentElement(
                        type="header",
                        content=text,
                        level=level,
                        position=(position, position + len(text)),
                        section_header=current_section,
                    ))
                    position += len(text) + 2
            
            # Handle code blocks
            elif tag_name == "pre":
                text = element.get_text()
                if text.strip():
                    # Try to detect language from class
                    lang = None
                    code_tag = element.find("code")
                    if code_tag and code_tag.get("class"):
                        for cls in code_tag.get("class", []):
                            if cls.startswith("language-"):
                                lang = cls[9:]
                                break
                    
                    elements.append(DocumentElement(
                        type="code",
                        content=text.strip(),
                        language=lang,
                        position=(position, position + len(text)),
                        section_header=current_section,
                    ))
                    position += len(text) + 2
            
            # Handle paragraphs
            elif tag_name == "p":
                text = element.get_text(strip=True)
                if text:
                    elements.append(DocumentElement(
                        type="paragraph",
                        content=text,
                        position=(position, position + len(text)),
                        section_header=current_section,
                    ))
                    position += len(text) + 2
            
            # Handle lists
            elif tag_name in ("ul", "ol"):
                items = []
                for li in element.find_all("li", recursive=False):
                    item_text = li.get_text(strip=True)
                    if item_text:
                        items.append(item_text)
                
                if items:
                    list_text = "\n".join(f"• {item}" for item in items)
                    elements.append(DocumentElement(
                        type="list",
                        content=list_text,
                        position=(position, position + len(list_text)),
                        section_header=current_section,
                        metadata={"list_type": tag_name, "item_count": len(items)}
                    ))
                    position += len(list_text) + 2
            
            # Handle blockquotes
            elif tag_name == "blockquote":
                text = element.get_text(strip=True)
                if text:
                    elements.append(DocumentElement(
                        type="blockquote",
                        content=text,
                        position=(position, position + len(text)),
                        section_header=current_section,
                    ))
                    position += len(text) + 2
            
            # Handle tables
            elif tag_name == "table":
                table_text = self._extract_table_bs4(element)
                if table_text:
                    elements.append(DocumentElement(
                        type="table",
                        content=table_text,
                        position=(position, position + len(table_text)),
                        section_header=current_section,
                    ))
                    position += len(table_text) + 2
        
        return ParsedDocument(
            elements=elements,
            title=title,
        )
    
    def _extract_table_bs4(self, table) -> str:
        """Extract table content as markdown."""
        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for cell in tr.find_all(["td", "th"]):
                cells.append(cell.get_text(strip=True))
            if cells:
                rows.append(cells)
        
        if not rows:
            return ""
        
        # Build markdown table
        lines = []
        for i, row in enumerate(rows):
            lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                lines.append("|" + "|".join(["---"] * len(row)) + "|")
        
        return "\n".join(lines)
    
    def _parse_with_regex(self, html: str) -> ParsedDocument:
        """Parse using regex (fallback when BS4 not available)."""
        elements = []
        position = 0
        
        # Remove script and style tags
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        
        # Extract title
        title = None
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = unescape(title_match.group(1).strip())
        
        current_section = None
        
        # Extract headers
        for match in re.finditer(r"<(h[1-6])[^>]*>(.*?)</\1>", html, re.IGNORECASE | re.DOTALL):
            level = int(match.group(1)[1])
            text = self._strip_tags(match.group(2)).strip()
            if text:
                current_section = text
                elements.append(DocumentElement(
                    type="header",
                    content=text,
                    level=level,
                    position=(match.start(), match.end()),
                    section_header=current_section,
                ))
        
        # Extract paragraphs
        for match in re.finditer(r"<p[^>]*>(.*?)</p>", html, re.IGNORECASE | re.DOTALL):
            text = self._strip_tags(match.group(1)).strip()
            if text:
                elements.append(DocumentElement(
                    type="paragraph",
                    content=text,
                    position=(match.start(), match.end()),
                ))
        
        # Extract code blocks
        for match in re.finditer(r"<pre[^>]*>(.*?)</pre>", html, re.IGNORECASE | re.DOTALL):
            text = self._strip_tags(match.group(1))
            if text.strip():
                elements.append(DocumentElement(
                    type="code",
                    content=unescape(text),
                    position=(match.start(), match.end()),
                ))
        
        # Sort by position
        elements.sort(key=lambda e: e.position[0])
        
        return ParsedDocument(
            elements=elements,
            title=title,
        )
    
    def _strip_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode entities
        text = unescape(text)
        # Normalize whitespace
        if not self.preserve_whitespace:
            text = re.sub(r"\s+", " ", text)
        return text
