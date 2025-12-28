"""
PDF document parser.

Extracts text and structure from PDF files using optional dependencies.
Supports multiple backends: PyMuPDF (fitz), pypdf, and pdfplumber.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import BinaryIO, Literal

from monkey.parsers.base import BaseParser, ParsedDocument, DocumentElement


class PDFParser(BaseParser):
    """
    Parse PDF documents into structured elements.
    
    Supports multiple backends:
    - 'pymupdf' (fitz): Best quality, supports images/tables
    - 'pypdf': Pure Python, lightweight
    - 'pdfplumber': Good for tables
    
    Example:
        >>> parser = PDFParser()
        >>> doc = parser.parse_file("document.pdf")
        >>> for element in doc.elements:
        ...     print(element.type, element.content[:50])
    """
    
    def __init__(
        self,
        backend: Literal["auto", "pymupdf", "pypdf", "pdfplumber"] = "auto",
        extract_images: bool = False,
        extract_tables: bool = True,
        page_separator: str = "\n\n---PAGE BREAK---\n\n",
    ):
        """
        Initialize PDF parser.
        
        Args:
            backend: PDF library to use ('auto' tries pymupdf -> pypdf -> pdfplumber)
            extract_images: Whether to extract image descriptions (requires pymupdf)
            extract_tables: Whether to detect and extract tables
            page_separator: Text to insert between pages
        """
        self.backend = backend
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.page_separator = page_separator
        self._backend_module = None
    
    def _get_backend(self):
        """Detect and return available PDF backend."""
        if self._backend_module is not None:
            return self._backend_module
        
        backends_to_try = []
        
        if self.backend == "auto":
            backends_to_try = ["pymupdf", "pypdf", "pdfplumber"]
        else:
            backends_to_try = [self.backend]
        
        for backend in backends_to_try:
            try:
                if backend == "pymupdf":
                    import fitz
                    self._backend_module = ("pymupdf", fitz)
                    return self._backend_module
                elif backend == "pypdf":
                    import pypdf
                    self._backend_module = ("pypdf", pypdf)
                    return self._backend_module
                elif backend == "pdfplumber":
                    import pdfplumber
                    self._backend_module = ("pdfplumber", pdfplumber)
                    return self._backend_module
            except ImportError:
                continue
        
        raise ImportError(
            "No PDF backend available. Install one of:\n"
            "  pip install pymupdf     # Best quality\n"
            "  pip install pypdf       # Lightweight\n"
            "  pip install pdfplumber  # Good for tables\n"
            "Or: pip install chunky-monkey[pdf]"
        )
    
    def parse(self, text: str) -> ParsedDocument:
        """
        Parse PDF from text (not typical usage - use parse_file or parse_bytes).
        
        For PDFs, this assumes text is already extracted.
        """
        elements = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        position = 0
        for para in paragraphs:
            elements.append(DocumentElement(
                type="paragraph",
                content=para,
                position=(position, position + len(para)),
            ))
            position += len(para) + 2
        
        return ParsedDocument(elements=elements)
    
    def parse_file(self, path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file.
        
        Args:
            path: Path to PDF file
            
        Returns:
            ParsedDocument with extracted elements
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        
        with open(path, "rb") as f:
            return self.parse_bytes(f.read(), source=str(path))
    
    def parse_bytes(self, data: bytes, source: str | None = None) -> ParsedDocument:
        """
        Parse PDF from bytes.
        
        Args:
            data: PDF file content as bytes
            source: Optional source identifier
            
        Returns:
            ParsedDocument with extracted elements
        """
        backend_name, backend = self._get_backend()
        
        if backend_name == "pymupdf":
            return self._parse_pymupdf(data, backend, source)
        elif backend_name == "pypdf":
            return self._parse_pypdf(data, backend, source)
        elif backend_name == "pdfplumber":
            return self._parse_pdfplumber(data, backend, source)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    
    def _parse_pymupdf(self, data: bytes, fitz, source: str | None) -> ParsedDocument:
        """Parse using PyMuPDF (fitz)."""
        elements = []
        position = 0
        title = None
        
        doc = fitz.open(stream=data, filetype="pdf")
        
        try:
            # Try to get document title from metadata
            metadata = doc.metadata
            if metadata and metadata.get("title"):
                title = metadata["title"]
            
            for page_num, page in enumerate(doc):
                # Extract text blocks with position info
                blocks = page.get_text("blocks")
                
                for block in blocks:
                    if block[6] == 0:  # Text block (not image)
                        text = block[4].strip()
                        if not text:
                            continue
                        
                        # Detect if this might be a header (short, possibly bold)
                        is_header = len(text) < 100 and not text.endswith(".")
                        
                        element_type = "header" if is_header else "paragraph"
                        
                        elements.append(DocumentElement(
                            type=element_type,
                            content=text,
                            level=1 if is_header else None,
                            position=(position, position + len(text)),
                            metadata={"page": page_num + 1}
                        ))
                        position += len(text) + 2
                
                # Add page separator
                if page_num < len(doc) - 1:
                    elements.append(DocumentElement(
                        type="paragraph",
                        content=self.page_separator.strip(),
                        position=(position, position + len(self.page_separator)),
                        metadata={"page_break": True}
                    ))
                    position += len(self.page_separator)
        finally:
            doc.close()
        
        return ParsedDocument(
            elements=elements,
            source=source,
            title=title,
            metadata={"page_count": len(doc) if doc else 0}
        )
    
    def _parse_pypdf(self, data: bytes, pypdf, source: str | None) -> ParsedDocument:
        """Parse using pypdf."""
        import io
        
        elements = []
        position = 0
        
        reader = pypdf.PdfReader(io.BytesIO(data))
        
        # Try to get title from metadata
        title = None
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            for para in paragraphs:
                elements.append(DocumentElement(
                    type="paragraph",
                    content=para,
                    position=(position, position + len(para)),
                    metadata={"page": page_num + 1}
                ))
                position += len(para) + 2
            
            # Add page separator
            if page_num < len(reader.pages) - 1:
                elements.append(DocumentElement(
                    type="paragraph",
                    content=self.page_separator.strip(),
                    position=(position, position + len(self.page_separator)),
                    metadata={"page_break": True}
                ))
                position += len(self.page_separator)
        
        return ParsedDocument(
            elements=elements,
            source=source,
            title=title,
            metadata={"page_count": len(reader.pages)}
        )
    
    def _parse_pdfplumber(self, data: bytes, pdfplumber, source: str | None) -> ParsedDocument:
        """Parse using pdfplumber."""
        import io
        
        elements = []
        position = 0
        
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                
                # Extract tables if enabled
                if self.extract_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        # Convert table to markdown-style text
                        table_text = self._table_to_text(table)
                        elements.append(DocumentElement(
                            type="table",
                            content=table_text,
                            position=(position, position + len(table_text)),
                            metadata={"page": page_num + 1}
                        ))
                        position += len(table_text) + 2
                
                # Split remaining text into paragraphs
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                
                for para in paragraphs:
                    elements.append(DocumentElement(
                        type="paragraph",
                        content=para,
                        position=(position, position + len(para)),
                        metadata={"page": page_num + 1}
                    ))
                    position += len(para) + 2
                
                # Add page separator
                if page_num < len(pdf.pages) - 1:
                    elements.append(DocumentElement(
                        type="paragraph",
                        content=self.page_separator.strip(),
                        position=(position, position + len(self.page_separator)),
                        metadata={"page_break": True}
                    ))
                    position += len(self.page_separator)
        
        return ParsedDocument(
            elements=elements,
            source=source,
            metadata={"page_count": len(pdf.pages) if pdf else 0}
        )
    
    def _table_to_text(self, table: list[list]) -> str:
        """Convert a table to markdown-style text."""
        if not table or not table[0]:
            return ""
        
        lines = []
        for i, row in enumerate(table):
            cells = [str(cell) if cell else "" for cell in row]
            lines.append("| " + " | ".join(cells) + " |")
            
            # Add header separator after first row
            if i == 0:
                lines.append("|" + "|".join(["---"] * len(cells)) + "|")
        
        return "\n".join(lines)
