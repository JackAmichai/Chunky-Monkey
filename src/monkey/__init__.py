"""
Chunky Monkey - Universal semantic text chunking for LLM/RAG applications.

Don't split text. Encapsulate meaning.

Supports multiple file formats:
- Plain text (.txt)
- Markdown (.md)
- PDF (.pdf) - requires: pip install chunky-monkey[pdf]
- HTML (.html) - enhanced with: pip install chunky-monkey[html]
- Word (.docx) - requires: pip install chunky-monkey[docx]
- JSON (.json)
- CSV (.csv)
"""

__version__ = "0.2.0"

# Core exports - will be populated as modules are implemented
__all__ = [
    "__version__",
    # Core API
    "chunk",
    "ChunkyMonkey",
    "Chunk",
    "ChunkMetadata",
    # File utilities
    "chunk_file",
    "chunk_files",
    "chunk_directory",
    # Tokenizers
    "CharacterTokenizer",
    "WordTokenizer",
    # Parsers
    "MarkdownParser",
    "PlainTextParser",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading all modules on import."""
    # Core schema
    if name == "Chunk":
        from monkey.core.schema import Chunk
        return Chunk
    if name == "ChunkMetadata":
        from monkey.core.schema import ChunkMetadata
        return ChunkMetadata
    
    # Core chunker
    if name == "chunk":
        from monkey.core.chunker import chunk
        return chunk
    if name == "ChunkyMonkey":
        from monkey.core.chunker import ChunkyMonkey
        return ChunkyMonkey
    
    # File utilities
    if name == "chunk_file":
        from monkey.files import chunk_file
        return chunk_file
    if name == "chunk_files":
        from monkey.files import chunk_files
        return chunk_files
    if name == "chunk_directory":
        from monkey.files import chunk_directory
        return chunk_directory
    
    # Tokenizers
    if name == "CharacterTokenizer":
        from monkey.tokenizers import CharacterTokenizer
        return CharacterTokenizer
    if name == "WordTokenizer":
        from monkey.tokenizers import WordTokenizer
        return WordTokenizer
    
    # Parsers
    if name == "MarkdownParser":
        from monkey.parsers import MarkdownParser
        return MarkdownParser
    if name == "PlainTextParser":
        from monkey.parsers import PlainTextParser
        return PlainTextParser
    if name == "PDFParser":
        from monkey.parsers import PDFParser
        return PDFParser
    if name == "HTMLParser":
        from monkey.parsers import HTMLParser
        return HTMLParser
    if name == "DocxParser":
        from monkey.parsers import DocxParser
        return DocxParser
    if name == "JSONParser":
        from monkey.parsers import JSONParser
        return JSONParser
    if name == "CSVParser":
        from monkey.parsers import CSVParser
        return CSVParser
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
