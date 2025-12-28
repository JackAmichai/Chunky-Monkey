"""
Unified file chunking utility.

Automatically detects file type and applies appropriate parser and chunker.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from monkey.core.schema import Chunk
from monkey.core.chunker import TextChunker, ChunkyMonkey
from monkey.parsers.base import Parser, ParsedDocument
from monkey.parsers.plaintext import PlainTextParser
from monkey.parsers.markdown import MarkdownParser

if TYPE_CHECKING:
    from monkey.tokenizers.base import Tokenizer


# File extension to parser mapping
EXTENSION_PARSERS = {
    # Plain text
    ".txt": "plaintext",
    ".text": "plaintext",
    ".log": "plaintext",
    
    # Markdown
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdown": "markdown",
    ".mkd": "markdown",
    ".rst": "plaintext",  # TODO: Add RST parser
    
    # Code files (treat as plain text with code awareness)
    ".py": "plaintext",
    ".js": "plaintext",
    ".ts": "plaintext",
    ".java": "plaintext",
    ".c": "plaintext",
    ".cpp": "plaintext",
    ".h": "plaintext",
    ".hpp": "plaintext",
    ".cs": "plaintext",
    ".go": "plaintext",
    ".rs": "plaintext",
    ".rb": "plaintext",
    ".php": "plaintext",
    ".swift": "plaintext",
    ".kt": "plaintext",
    ".scala": "plaintext",
    ".r": "plaintext",
    ".sql": "plaintext",
    ".sh": "plaintext",
    ".bash": "plaintext",
    ".zsh": "plaintext",
    ".ps1": "plaintext",
    ".yaml": "plaintext",
    ".yml": "plaintext",
    ".toml": "plaintext",
    ".ini": "plaintext",
    ".cfg": "plaintext",
    ".conf": "plaintext",
    ".xml": "plaintext",
    
    # HTML
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    
    # PDF
    ".pdf": "pdf",
    
    # Word documents
    ".docx": "docx",
    
    # Data files
    ".json": "json",
    ".jsonl": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
}


def get_parser_for_file(path: str | Path) -> Parser:
    """
    Get appropriate parser for a file based on extension.
    
    Args:
        path: Path to file
        
    Returns:
        Parser instance suitable for the file type
        
    Raises:
        ValueError: If file type is not supported
    """
    path = Path(path)
    ext = path.suffix.lower()
    
    parser_type = EXTENSION_PARSERS.get(ext)
    
    if parser_type is None:
        # Default to plaintext for unknown extensions
        parser_type = "plaintext"
    
    return _create_parser(parser_type)


def _create_parser(parser_type: str) -> Parser:
    """Create a parser instance by type name."""
    if parser_type == "plaintext":
        return PlainTextParser()
    
    elif parser_type == "markdown":
        return MarkdownParser()
    
    elif parser_type == "html":
        from monkey.parsers.html import HTMLParser
        return HTMLParser()
    
    elif parser_type == "pdf":
        from monkey.parsers.pdf import PDFParser
        return PDFParser()
    
    elif parser_type == "docx":
        from monkey.parsers.docx import DocxParser
        return DocxParser()
    
    elif parser_type == "json":
        from monkey.parsers.data import JSONParser
        return JSONParser()
    
    elif parser_type == "jsonl":
        from monkey.parsers.data import JSONParser
        return JSONParser()
    
    elif parser_type == "csv":
        from monkey.parsers.data import CSVParser
        return CSVParser()
    
    elif parser_type == "tsv":
        from monkey.parsers.data import CSVParser
        return CSVParser(delimiter="\t")
    
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")


def chunk_file(
    path: str | Path,
    max_tokens: int = 512,
    overlap_tokens: int = 0,
    tokenizer: "Tokenizer | None" = None,
    parser: Parser | None = None,
) -> list[Chunk]:
    """
    Chunk a file with automatic format detection.
    
    This is the simplest way to chunk any supported file format.
    The function automatically detects the file type and applies
    the appropriate parser.
    
    Args:
        path: Path to file to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        tokenizer: Optional tokenizer (defaults to character-based)
        parser: Optional parser override (auto-detected if None)
        
    Returns:
        List of Chunk objects
        
    Example:
        >>> chunks = chunk_file("document.pdf", max_tokens=1000)
        >>> chunks = chunk_file("data.json", max_tokens=500)
        >>> chunks = chunk_file("README.md")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Get parser
    if parser is None:
        parser = get_parser_for_file(path)
    
    # Create chunker with source set
    monkey = ChunkyMonkey(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer=tokenizer,
        parser=parser,
        source=str(path),
    )
    
    # Read and parse file
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        from monkey.parsers.pdf import PDFParser
        if isinstance(parser, PDFParser):
            doc = parser.parse_file(path)
            text = doc.get_text()
        else:
            with open(path, "rb") as f:
                text = f.read().decode("utf-8", errors="ignore")
    
    elif ext == ".docx":
        from monkey.parsers.docx import DocxParser
        if isinstance(parser, DocxParser):
            doc = parser.parse_file(path)
            text = doc.get_text()
        else:
            # Can't read docx as plain text
            raise ValueError("DOCX files require DocxParser")
    
    else:
        # Text-based files
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
    
    # Chunk
    return monkey.chunk(text)


def chunk_files(
    paths: list[str | Path],
    max_tokens: int = 512,
    overlap_tokens: int = 0,
    tokenizer: "Tokenizer | None" = None,
) -> list[Chunk]:
    """
    Chunk multiple files with automatic format detection.
    
    Args:
        paths: List of file paths to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap
        tokenizer: Optional tokenizer
        
    Returns:
        Combined list of Chunk objects from all files
    """
    all_chunks = []
    
    for path in paths:
        try:
            chunks = chunk_file(
                path,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                tokenizer=tokenizer,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            # Log error but continue with other files
            print(f"Warning: Failed to chunk {path}: {e}")
    
    return all_chunks


def chunk_directory(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = True,
    max_tokens: int = 512,
    overlap_tokens: int = 0,
    tokenizer: "Tokenizer | None" = None,
    extensions: list[str] | None = None,
) -> list[Chunk]:
    """
    Chunk all files in a directory.
    
    Args:
        directory: Directory path
        pattern: Glob pattern for file matching
        recursive: Search subdirectories
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Tokens overlap between chunks
        tokenizer: Optional tokenizer
        extensions: List of extensions to include (e.g., [".pdf", ".md"])
        
    Returns:
        List of Chunk objects from all matching files
        
    Example:
        >>> chunks = chunk_directory("./docs", extensions=[".md", ".txt"])
        >>> chunks = chunk_directory("./data", pattern="*.pdf")
    """
    directory = Path(directory)
    
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    # Find files
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Filter by extension
    if extensions:
        extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" 
                     for ext in extensions]
        files = [f for f in files if f.suffix.lower() in extensions]
    
    # Filter to only supported files
    supported_files = [f for f in files if f.is_file() and f.suffix.lower() in EXTENSION_PARSERS]
    
    return chunk_files(
        supported_files,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer=tokenizer,
    )


# List of supported extensions
SUPPORTED_EXTENSIONS = list(EXTENSION_PARSERS.keys())


def is_supported(path: str | Path) -> bool:
    """Check if a file type is supported for chunking."""
    path = Path(path)
    return path.suffix.lower() in EXTENSION_PARSERS
