# Chunky Monkey - Code Explanation

This document explains the architecture, design decisions, and implementation details of the Chunky Monkey library.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Core Schema](#core-schema)
3. [Tokenizer Abstraction](#tokenizer-abstraction)
4. [Sentence Segmentation](#sentence-segmentation)
5. [Text Chunker](#text-chunker)
6. [Markdown Parser](#markdown-parser)
7. [Validators](#validators)
8. [Public API](#public-api)
9. [Semantic Chunking](#semantic-chunking)

---

## Project Structure

```
chunky-monkey/
├── pyproject.toml          # Modern Python packaging (PEP 621)
├── src/monkey/             # Source code (src-layout for clean imports)
│   ├── __init__.py         # Public API exports
│   ├── core/               # Core chunking logic
│   │   ├── schema.py       # Data structures (Chunk, ChunkMetadata)
│   │   ├── segmentation.py # Sentence splitting
│   │   └── chunker.py      # Main chunking algorithm
│   ├── tokenizers/         # Token counting backends
│   │   ├── base.py         # Protocol definition
│   │   ├── character.py    # Fallback (no dependencies)
│   │   ├── tiktoken_.py    # OpenAI tokenizer
│   │   └── huggingface_.py # HuggingFace transformers
│   ├── parsers/            # Document structure extraction
│   │   ├── base.py         # Parser protocol
│   │   ├── plaintext.py    # Simple text (no structure)
│   │   └── markdown.py     # Markdown headers, code, lists
│   ├── validators/         # Integrity checks
│   │   └── integrity.py    # Token limits, roundtrip, references
│   └── semantic/           # Optional semantic features
│       └── boundaries.py   # Topic-shift detection
├── tests/                  # Test suite
└── examples/               # Usage examples
```

### Why src-layout?

The `src/` layout prevents accidentally importing from the local directory instead of the installed package. It's the recommended structure for publishable Python libraries.

---

## Core Schema

**File:** `src/monkey/core/schema.py`

### Design Decisions

1. **Dataclasses over Pydantic (by default)**
   - Zero dependencies in core
   - Fast instantiation
   - Built-in `asdict()` for JSON serialization
   - Optional Pydantic conversion for users who need validation

2. **Deterministic chunk_id**
   - Uses SHA-256 hash of (text + position)
   - Same input always produces same ID
   - Critical for reproducible embeddings and caching

3. **Rich metadata structure**
   - `source`: Original document path/name
   - `position`: Character offsets in original document
   - `section_header`: Inherited from document structure
   - `parent_chunk_id`: For chunk hierarchies

### Key Code Patterns

```python
@dataclass(frozen=True)  # Immutable for hashability
class ChunkMetadata:
    source: str | None = None
    position: tuple[int, int] | None = None  # (start, end) offsets
    section_header: str | None = None
    parent_chunk_id: str | None = None
```

The `frozen=True` makes chunks immutable—once created, they can't be modified. This prevents bugs from accidental mutation and allows chunks to be used as dict keys or in sets.

---

## Tokenizer Abstraction

**Files:** `src/monkey/tokenizers/`

### Why Abstraction?

Different LLMs use different tokenizers:
- OpenAI: tiktoken (BPE)
- Llama/Mistral: SentencePiece
- HuggingFace models: Various

The same text produces **different token counts** across models. A universal chunker must support pluggable tokenizers.

### Protocol Pattern

```python
class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
```

Using `Protocol` (structural subtyping) instead of ABC means any class with these methods works—no inheritance required.

### Implementations

1. **CharacterTokenizer** (fallback)
   - 1 char ≈ 1 token (rough approximation)
   - Zero dependencies
   - Good enough for testing, not for production

2. **TiktokenTokenizer** (OpenAI)
   - Uses `tiktoken` library
   - Supports all OpenAI models (gpt-4, gpt-3.5-turbo, etc.)
   - Fast C implementation

3. **HuggingFaceTokenizer** (open models)
   - Uses `transformers.AutoTokenizer`
   - Supports Llama, Mistral, etc.
   - Auto-downloads tokenizer files

---

## Sentence Segmentation

**File:** `src/monkey/core/segmentation.py`

### The Challenge

Splitting on `.!?` breaks on:
- Abbreviations: "Mr. Smith went home."
- Numbers: "The value is 3.14159."
- URLs: "Visit https://example.com/page."
- Ellipsis: "Wait... what happened?"

### Solution: Regex with Negative Lookbehinds

```python
SENTENCE_PATTERN = re.compile(
    r'(?<!'          # Not preceded by:
    r'Mr|Mrs|Dr|...' # Common abbreviations
    r')'
    r'[.!?]'         # Sentence-ending punctuation
    r'(?=\s|$)'      # Followed by whitespace or end
)
```

### Output Format

Returns `Sentence` objects with:
- `text`: The sentence content
- `start`: Character offset in original document
- `end`: Character offset (exclusive)

This enables reconstructing the original document and mapping chunks back to source positions.

---

## Text Chunker

**File:** `src/monkey/core/chunker.py`

### Algorithm Overview

1. **Segment** text into sentences
2. **Group** sentences into chunks until token budget is reached
3. **Never break** mid-sentence (except oversized sentences)
4. **Apply overlap** by repeating ending sentences in next chunk
5. **Preserve** atomic units (code blocks, lists)

### Key Parameters

- `max_tokens`: Hard limit per chunk
- `overlap_tokens`: Tokens to repeat between chunks (context continuity)
- `preserve_code_blocks`: Keep ``` blocks as single units
- `preserve_lists`: Keep list items grouped

### Hard-Split Logic

When a single sentence exceeds `max_tokens`:

```python
def _hard_split(self, text: str, max_tokens: int) -> list[str]:
    """Last resort: split oversized text at token boundaries."""
    tokens = self.tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(self.tokenizer.decode(chunk_tokens))
    return chunks
```

This preserves as much context as possible while guaranteeing the token limit.

---

## Markdown Parser

**File:** `src/monkey/parsers/markdown.py`

### What It Extracts

1. **Headers** (H1-H6): Creates section hierarchy
2. **Code blocks**: Fenced ``` blocks as atomic units
3. **Lists**: Bullet and numbered lists grouped
4. **Paragraphs**: Standard text blocks

### Structure Tree

```python
@dataclass
class MarkdownElement:
    type: Literal["header", "code", "list", "paragraph"]
    content: str
    level: int | None = None  # Header level (1-6)
    language: str | None = None  # Code block language
    position: tuple[int, int] = (0, 0)
```

### Header Propagation

The parser tracks "current section" and attaches it to all subsequent elements:

```markdown
# Chapter 1          ← Sets section to "Chapter 1"

Some paragraph.      ← metadata.section_header = "Chapter 1"

## Section 1.1       ← Sets section to "Chapter 1 > Section 1.1"

More text.           ← metadata.section_header = "Chapter 1 > Section 1.1"
```

This context travels with chunks, so the LLM knows where each chunk belongs.

---

## Validators

**File:** `src/monkey/validators/integrity.py`

### Three Validation Functions

1. **`validate_token_limits(chunks, max_tokens, tokenizer)`**
   - Asserts no chunk exceeds the specified limit
   - Returns list of violations

2. **`validate_roundtrip(original, chunks, tolerance=0.95)`**
   - Joins chunks and compares to original
   - Accounts for overlap (not exact match)
   - Ensures no content was lost

3. **`flag_dangling_references(chunks)`**
   - Detects orphan pronouns: "it", "they", "this", "that"
   - Flags chunks that start with references without antecedents
   - Helps identify context loss

### Usage Pattern

```python
chunks = chunker.chunk(text)

# Validate before using
violations = validate_token_limits(chunks, max_tokens=500, tokenizer=tok)
if violations:
    raise ValueError(f"Chunks exceed limit: {violations}")

warnings = flag_dangling_references(chunks)
for w in warnings:
    logger.warning(f"Possible context loss: {w}")
```

---

## Public API

**File:** `src/monkey/__init__.py`

### Simple Function

```python
def chunk(
    text: str,
    max_tokens: int = 500,
    tokenizer: Tokenizer | None = None,
    overlap_tokens: int = 0
) -> list[Chunk]:
    """One-liner chunking for quick use."""
```

### Full Configuration Class

```python
class ChunkyMonkey:
    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        parser: Parser | None = None,
        max_tokens: int = 500,
        overlap_tokens: int = 0,
        preserve_code_blocks: bool = True,
        validate: bool = True
    ):
        ...
    
    def chunk(self, text: str) -> list[Chunk]:
        ...
    
    def chunk_file(self, path: str) -> list[Chunk]:
        ...
```

### Design Philosophy

- **Simple things simple**: `chunk(text)` just works
- **Complex things possible**: `ChunkyMonkey(...)` for full control
- **No surprises**: Deterministic output, same input = same chunks

---

## Semantic Chunking

**File:** `src/monkey/semantic/boundaries.py`

### How It Works

1. **Embed** each sentence using an embedding function
2. **Calculate** cosine similarity between consecutive sentences
3. **Detect** topic shifts where similarity drops below threshold
4. **Split** at semantic boundaries instead of token counts

### Key Algorithm

```python
def find_semantic_boundaries(
    sentences: list[str],
    embeddings: list[np.ndarray],
    threshold: float = 0.5
) -> list[int]:
    """Returns indices where topic shifts occur."""
    boundaries = []
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity < threshold:
            boundaries.append(i)
    return boundaries
```

### Embedding Function Protocol

```python
EmbeddingFunction = Callable[[list[str]], list[np.ndarray]]
```

Users can provide:
- `sentence-transformers` model
- OpenAI embeddings API
- Any custom function

This keeps the core library lightweight—semantic features are opt-in.

---

## Summary

Chunky Monkey is built on these principles:

1. **Zero required dependencies** - Core works with just Python stdlib
2. **Structural integrity** - Never break sentences, code, or lists
3. **Universal compatibility** - Works with any LLM/tokenizer
4. **Rich metadata** - Every chunk knows its origin
5. **Validation built-in** - Catch problems before they cause hallucinations
6. **Deterministic** - Same input always produces same output

The modular architecture allows users to pick exactly the features they need while keeping the core lightweight and fast.
