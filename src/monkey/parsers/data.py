"""
JSON and CSV data parsers.

Converts structured data into text suitable for chunking.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from monkey.parsers.base import BaseParser, ParsedDocument, DocumentElement


class JSONParser(BaseParser):
    """
    Parse JSON documents into structured text elements.
    
    Converts JSON objects and arrays into readable text format
    suitable for LLM processing.
    
    Example:
        >>> parser = JSONParser()
        >>> doc = parser.parse('{"name": "John", "age": 30}')
        >>> print(doc.get_text())
        name: John
        age: 30
    """
    
    def __init__(
        self,
        flatten: bool = True,
        max_depth: int = 10,
        array_item_separator: str = "\n---\n",
        key_value_format: str = "{key}: {value}",
        include_nulls: bool = False,
    ):
        """
        Initialize JSON parser.
        
        Args:
            flatten: Flatten nested objects into dot-notation keys
            max_depth: Maximum nesting depth to process
            array_item_separator: Separator between array items
            key_value_format: Format string for key-value pairs
            include_nulls: Include null values in output
        """
        self.flatten = flatten
        self.max_depth = max_depth
        self.array_item_separator = array_item_separator
        self.key_value_format = key_value_format
        self.include_nulls = include_nulls
    
    def parse(self, text: str) -> ParsedDocument:
        """
        Parse JSON text into structured elements.
        
        Args:
            text: JSON string
            
        Returns:
            ParsedDocument with extracted elements
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        return self._parse_data(data)
    
    def parse_file(self, path: str | Path, encoding: str = "utf-8") -> ParsedDocument:
        """Parse a JSON file."""
        path = Path(path)
        text = path.read_text(encoding=encoding)
        doc = self.parse(text)
        doc.source = str(path)
        return doc
    
    def _parse_data(self, data: Any, prefix: str = "") -> ParsedDocument:
        """Parse JSON data into document elements."""
        elements = []
        position = 0
        
        if isinstance(data, list):
            # Array of objects - each becomes an element
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    content = self._dict_to_text(item, prefix)
                else:
                    content = str(item)
                
                if content:
                    elements.append(DocumentElement(
                        type="paragraph",
                        content=content,
                        position=(position, position + len(content)),
                        metadata={"array_index": i}
                    ))
                    position += len(content) + len(self.array_item_separator)
        
        elif isinstance(data, dict):
            # Single object
            content = self._dict_to_text(data, prefix)
            if content:
                elements.append(DocumentElement(
                    type="paragraph",
                    content=content,
                    position=(0, len(content)),
                ))
        
        else:
            # Scalar value
            content = str(data)
            elements.append(DocumentElement(
                type="paragraph",
                content=content,
                position=(0, len(content)),
            ))
        
        return ParsedDocument(elements=elements)
    
    def _dict_to_text(self, obj: dict, prefix: str = "", depth: int = 0) -> str:
        """Convert a dictionary to readable text."""
        if depth > self.max_depth:
            return str(obj)
        
        lines = []
        
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if value is None and not self.include_nulls:
                continue
            
            if isinstance(value, dict) and self.flatten:
                nested = self._dict_to_text(value, full_key, depth + 1)
                if nested:
                    lines.append(nested)
            elif isinstance(value, list):
                if all(isinstance(v, (str, int, float, bool)) for v in value):
                    # Simple list - join values
                    lines.append(self.key_value_format.format(
                        key=full_key,
                        value=", ".join(str(v) for v in value)
                    ))
                else:
                    # Complex list - each item on own line
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            nested = self._dict_to_text(item, f"{full_key}[{i}]", depth + 1)
                            if nested:
                                lines.append(nested)
                        else:
                            lines.append(f"{full_key}[{i}]: {item}")
            else:
                lines.append(self.key_value_format.format(key=full_key, value=value))
        
        return "\n".join(lines)


class CSVParser(BaseParser):
    """
    Parse CSV documents into structured text elements.
    
    Converts CSV rows into readable text, with each row becoming
    a separate element for chunking.
    
    Example:
        >>> parser = CSVParser()
        >>> doc = parser.parse("name,age\\nJohn,30\\nJane,25")
        >>> for elem in doc.elements:
        ...     print(elem.content)
        name: John, age: 30
        name: Jane, age: 25
    """
    
    def __init__(
        self,
        has_header: bool = True,
        delimiter: str = ",",
        row_format: str = "key_value",  # "key_value", "table", "raw"
        rows_per_element: int = 1,
        include_row_numbers: bool = False,
    ):
        """
        Initialize CSV parser.
        
        Args:
            has_header: First row contains column names
            delimiter: CSV delimiter character
            row_format: How to format rows ("key_value", "table", "raw")
            rows_per_element: Number of rows per document element
            include_row_numbers: Include row numbers in output
        """
        self.has_header = has_header
        self.delimiter = delimiter
        self.row_format = row_format
        self.rows_per_element = rows_per_element
        self.include_row_numbers = include_row_numbers
    
    def parse(self, text: str) -> ParsedDocument:
        """
        Parse CSV text into structured elements.
        
        Args:
            text: CSV string
            
        Returns:
            ParsedDocument with extracted elements
        """
        reader = csv.reader(StringIO(text), delimiter=self.delimiter)
        rows = list(reader)
        
        if not rows:
            return ParsedDocument(elements=[])
        
        headers = None
        if self.has_header and rows:
            headers = rows[0]
            rows = rows[1:]
        
        return self._parse_rows(rows, headers)
    
    def parse_file(
        self,
        path: str | Path,
        encoding: str = "utf-8"
    ) -> ParsedDocument:
        """Parse a CSV file."""
        path = Path(path)
        text = path.read_text(encoding=encoding)
        doc = self.parse(text)
        doc.source = str(path)
        return doc
    
    def _parse_rows(
        self,
        rows: list[list[str]],
        headers: list[str] | None
    ) -> ParsedDocument:
        """Parse CSV rows into document elements."""
        elements = []
        position = 0
        
        # Group rows
        for i in range(0, len(rows), self.rows_per_element):
            batch = rows[i:i + self.rows_per_element]
            
            if self.row_format == "table" and headers:
                content = self._rows_to_table(batch, headers)
            elif self.row_format == "raw":
                content = "\n".join(self.delimiter.join(row) for row in batch)
            else:  # key_value
                content = self._rows_to_key_value(batch, headers, i)
            
            if content:
                elements.append(DocumentElement(
                    type="paragraph",
                    content=content,
                    position=(position, position + len(content)),
                    metadata={
                        "row_start": i + 1 if self.has_header else i,
                        "row_count": len(batch),
                    }
                ))
                position += len(content) + 2
        
        return ParsedDocument(
            elements=elements,
            metadata={
                "headers": headers,
                "total_rows": len(rows),
            }
        )
    
    def _rows_to_key_value(
        self,
        rows: list[list[str]],
        headers: list[str] | None,
        start_idx: int
    ) -> str:
        """Convert rows to key-value format."""
        lines = []
        
        for row_num, row in enumerate(rows):
            if self.include_row_numbers:
                lines.append(f"Row {start_idx + row_num + 1}:")
            
            if headers:
                pairs = []
                for j, value in enumerate(row):
                    if j < len(headers):
                        pairs.append(f"{headers[j]}: {value}")
                    else:
                        pairs.append(value)
                lines.append(", ".join(pairs))
            else:
                lines.append(", ".join(row))
        
        return "\n".join(lines)
    
    def _rows_to_table(
        self,
        rows: list[list[str]],
        headers: list[str]
    ) -> str:
        """Convert rows to markdown table format."""
        lines = []
        
        # Header row
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        # Data rows
        for row in rows:
            # Pad row to match headers length
            padded = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
        
        return "\n".join(lines)
