"""Tests for JSON and CSV parsers."""

import pytest
from monkey.parsers.data import JSONParser, CSVParser


class TestJSONParser:
    """Test JSON parser functionality."""
    
    def test_simple_object(self):
        """Test parsing simple JSON object."""
        parser = JSONParser()
        doc = parser.parse('{"name": "John", "age": 30}')
        
        assert len(doc.elements) == 1
        assert "name: John" in doc.elements[0].content
        assert "age: 30" in doc.elements[0].content
    
    def test_array_of_objects(self):
        """Test parsing JSON array."""
        parser = JSONParser()
        doc = parser.parse('[{"name": "John"}, {"name": "Jane"}]')
        
        assert len(doc.elements) == 2
        assert "John" in doc.elements[0].content
        assert "Jane" in doc.elements[1].content
    
    def test_nested_object(self):
        """Test parsing nested JSON."""
        parser = JSONParser(flatten=True)
        doc = parser.parse('{"person": {"name": "John", "age": 30}}')
        
        text = doc.get_text()
        assert "person.name: John" in text
        assert "person.age: 30" in text
    
    def test_array_values(self):
        """Test parsing object with array values."""
        parser = JSONParser()
        doc = parser.parse('{"tags": ["python", "ai", "ml"]}')
        
        text = doc.get_text()
        assert "python" in text
        assert "ai" in text
    
    def test_null_handling(self):
        """Test null value handling."""
        parser = JSONParser(include_nulls=False)
        doc = parser.parse('{"name": "John", "email": null}')
        
        text = doc.get_text()
        assert "name: John" in text
        assert "email" not in text
    
    def test_include_nulls(self):
        """Test including null values."""
        parser = JSONParser(include_nulls=True)
        doc = parser.parse('{"name": "John", "email": null}')
        
        text = doc.get_text()
        assert "email: None" in text
    
    def test_invalid_json(self):
        """Test error on invalid JSON."""
        parser = JSONParser()
        
        with pytest.raises(ValueError):
            parser.parse('{"invalid": }')
    
    def test_scalar_value(self):
        """Test parsing scalar JSON value."""
        parser = JSONParser()
        doc = parser.parse('"hello world"')
        
        assert len(doc.elements) == 1
        assert "hello world" in doc.elements[0].content


class TestCSVParser:
    """Test CSV parser functionality."""
    
    def test_basic_csv(self):
        """Test parsing basic CSV."""
        parser = CSVParser()
        doc = parser.parse("name,age\nJohn,30\nJane,25")
        
        assert len(doc.elements) == 2
        assert "name: John" in doc.elements[0].content
        assert "age: 30" in doc.elements[0].content
        assert "name: Jane" in doc.elements[1].content
    
    def test_no_header(self):
        """Test CSV without header."""
        parser = CSVParser(has_header=False)
        doc = parser.parse("John,30\nJane,25")
        
        assert len(doc.elements) == 2
        assert "John, 30" in doc.elements[0].content
    
    def test_custom_delimiter(self):
        """Test CSV with custom delimiter."""
        parser = CSVParser(delimiter="\t")
        doc = parser.parse("name\tage\nJohn\t30")
        
        text = doc.get_text()
        assert "name: John" in text
    
    def test_table_format(self):
        """Test table format output."""
        parser = CSVParser(row_format="table")
        doc = parser.parse("name,age\nJohn,30\nJane,25")
        
        text = doc.get_text()
        assert "|" in text
        assert "name" in text
        assert "John" in text
    
    def test_raw_format(self):
        """Test raw format output."""
        parser = CSVParser(row_format="raw")
        doc = parser.parse("name,age\nJohn,30")
        
        text = doc.get_text()
        assert "John,30" in text
    
    def test_rows_per_element(self):
        """Test grouping multiple rows per element."""
        parser = CSVParser(rows_per_element=2)
        doc = parser.parse("name,age\nJohn,30\nJane,25\nBob,35\nAlice,28")
        
        assert len(doc.elements) == 2  # 4 rows / 2 = 2 elements
    
    def test_include_row_numbers(self):
        """Test including row numbers."""
        parser = CSVParser(include_row_numbers=True)
        doc = parser.parse("name,age\nJohn,30\nJane,25")
        
        text = doc.get_text()
        assert "Row" in text
    
    def test_metadata(self):
        """Test metadata extraction."""
        parser = CSVParser()
        doc = parser.parse("name,age\nJohn,30\nJane,25")
        
        assert doc.metadata["headers"] == ["name", "age"]
        assert doc.metadata["total_rows"] == 2
    
    def test_empty_csv(self):
        """Test parsing empty CSV."""
        parser = CSVParser()
        doc = parser.parse("")
        
        assert len(doc.elements) == 0
    
    def test_quoted_values(self):
        """Test CSV with quoted values."""
        parser = CSVParser()
        doc = parser.parse('name,description\nJohn,"A ""great"" person"')
        
        text = doc.get_text()
        assert "great" in text
