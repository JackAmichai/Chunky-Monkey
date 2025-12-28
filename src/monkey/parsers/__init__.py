"""Document parsers for extracting structure from various file formats."""

from monkey.parsers.base import Parser, ParsedDocument, DocumentElement
from monkey.parsers.plaintext import PlainTextParser
from monkey.parsers.markdown import MarkdownParser

__all__ = [
    # Base classes
    "Parser",
    "ParsedDocument",
    "DocumentElement",
    # Built-in parsers
    "PlainTextParser",
    "MarkdownParser",
    # Optional parsers (lazy imports)
    "PDFParser",
    "HTMLParser",
    "DocxParser",
    "JSONParser",
    "CSVParser",
]


# Lazy imports for optional parsers
def __getattr__(name: str):
    """Lazy import optional parsers."""
    if name == "PDFParser":
        from monkey.parsers.pdf import PDFParser
        return PDFParser
    if name == "HTMLParser":
        from monkey.parsers.html import HTMLParser
        return HTMLParser
    if name == "DocxParser":
        from monkey.parsers.docx import DocxParser
        return DocxParser
    if name == "JSONParser":
        from monkey.parsers.data import JSONParser
        return JSONParser
    if name == "CSVParser":
        from monkey.parsers.data import CSVParser
        return CSVParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
