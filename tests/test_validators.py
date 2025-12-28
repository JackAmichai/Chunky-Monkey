"""Tests for validators."""

import pytest
from monkey.core.schema import Chunk, ChunkMetadata
from monkey.tokenizers.character import CharacterTokenizer
from monkey.validators.integrity import (
    validate_token_limits,
    validate_roundtrip,
    flag_dangling_references,
    validate_chunks,
    ValidationError,
    ValidationWarning,
)


def make_chunk(text: str, chunk_id: str = "test") -> Chunk:
    """Helper to create test chunks."""
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        metadata=ChunkMetadata(),
        token_counts={}
    )


class TestValidateTokenLimits:
    """Tests for validate_token_limits."""
    
    def test_valid_chunks(self):
        """Test chunks within limit return no errors."""
        tok = CharacterTokenizer()
        chunks = [
            make_chunk("Hello"),  # 5 chars
            make_chunk("World"),  # 5 chars
        ]
        
        errors = validate_token_limits(chunks, max_tokens=10, tokenizer=tok)
        assert errors == []
    
    def test_oversized_chunk(self):
        """Test oversized chunk returns error."""
        tok = CharacterTokenizer()
        chunks = [
            make_chunk("This is way too long"),  # 20 chars
        ]
        
        errors = validate_token_limits(chunks, max_tokens=10, tokenizer=tok)
        assert len(errors) == 1
        assert errors[0].severity == "error"
        assert "Exceeds" in errors[0].message
    
    def test_error_details(self):
        """Test error contains useful details."""
        tok = CharacterTokenizer()
        chunks = [make_chunk("Hello world")]
        
        errors = validate_token_limits(chunks, max_tokens=5, tokenizer=tok)
        assert len(errors) == 1
        assert errors[0].details is not None
        assert "token_count" in errors[0].details
        assert "excess" in errors[0].details


class TestValidateRoundtrip:
    """Tests for validate_roundtrip."""
    
    def test_complete_coverage(self):
        """Test no errors when chunks cover original."""
        original = "Hello world test"
        chunks = [make_chunk("Hello world test")]
        
        errors = validate_roundtrip(original, chunks)
        assert errors == []
    
    def test_content_loss(self):
        """Test error when significant content is lost."""
        original = "A very long piece of text that should all be preserved in the chunks"
        chunks = [make_chunk("Short")]  # Lost most content
        
        errors = validate_roundtrip(original, chunks, tolerance=0.9)
        assert len(errors) >= 1
    
    def test_empty_chunks(self):
        """Test error when no chunks from non-empty input."""
        errors = validate_roundtrip("Hello", [])
        assert len(errors) == 1
        assert "No chunks" in errors[0].message
    
    def test_empty_original(self):
        """Test no error for empty original."""
        errors = validate_roundtrip("", [])
        assert errors == []


class TestFlagDanglingReferences:
    """Tests for flag_dangling_references."""
    
    def test_no_warnings_clean_text(self):
        """Test no warnings for clean text."""
        chunks = [
            make_chunk("The system works well.", "c1"),
            make_chunk("Performance is good.", "c2"),
        ]
        
        warnings = flag_dangling_references(chunks)
        assert warnings == []
    
    def test_warns_pronoun_start(self):
        """Test warning when chunk starts with pronoun."""
        chunks = [
            make_chunk("The machine is ready.", "c1"),
            make_chunk("It works perfectly.", "c2"),  # Starts with "It"
        ]
        
        warnings = flag_dangling_references(chunks)
        assert len(warnings) >= 1
        assert any("pronoun" in w.message.lower() for w in warnings)
    
    def test_warns_reference_phrase(self):
        """Test warning for reference phrases."""
        chunks = [
            make_chunk("First chunk.", "c1"),
            make_chunk("As mentioned above, this is important.", "c2"),
        ]
        
        warnings = flag_dangling_references(chunks)
        assert len(warnings) >= 1
        assert any("reference" in w.message.lower() for w in warnings)
    
    def test_first_chunk_skipped(self):
        """Test that first chunk doesn't get warnings."""
        chunks = [
            make_chunk("It starts here.", "c1"),  # First chunk with "It" is OK
        ]
        
        warnings = flag_dangling_references(chunks)
        assert warnings == []
    
    def test_multiple_issues(self):
        """Test detection of multiple issues."""
        chunks = [
            make_chunk("First.", "c1"),
            make_chunk("He did it.", "c2"),  # Pronoun
            make_chunk("She agreed.", "c3"),  # Pronoun
            make_chunk("As noted above.", "c4"),  # Reference phrase
        ]
        
        warnings = flag_dangling_references(chunks)
        assert len(warnings) >= 3


class TestValidateChunks:
    """Tests for the combined validate_chunks function."""
    
    def test_returns_tuple(self):
        """Test returns (errors, warnings) tuple."""
        chunks = [make_chunk("Hello")]
        result = validate_chunks(chunks)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        errors, warnings = result
        assert isinstance(errors, list)
        assert isinstance(warnings, list)
    
    def test_all_validations(self):
        """Test all validations run when parameters provided."""
        tok = CharacterTokenizer()
        chunks = [make_chunk("Hello")]
        original = "Hello"
        
        errors, warnings = validate_chunks(
            chunks,
            original=original,
            max_tokens=100,
            tokenizer=tok,
            check_references=True
        )
        
        # Should complete without error
        assert isinstance(errors, list)
        assert isinstance(warnings, list)
    
    def test_skips_validations_without_params(self):
        """Test validations are skipped when params not provided."""
        chunks = [make_chunk("Hello")]
        
        # Only reference checking should run (no original, no tokenizer)
        errors, warnings = validate_chunks(chunks, check_references=False)
        assert errors == []
        assert warnings == []


class TestValidationError:
    """Tests for ValidationError class."""
    
    def test_str(self):
        """Test string representation."""
        error = ValidationError(
            chunk_id="abc123",
            message="Something went wrong"
        )
        s = str(error)
        assert "ERROR" in s
        assert "abc123" in s
        assert "Something went wrong" in s
    
    def test_default_severity(self):
        """Test default severity is error."""
        error = ValidationError(chunk_id="x", message="test")
        assert error.severity == "error"


class TestValidationWarning:
    """Tests for ValidationWarning class."""
    
    def test_default_severity(self):
        """Test default severity is warning."""
        warning = ValidationWarning(chunk_id="x", message="test")
        assert warning.severity == "warning"
