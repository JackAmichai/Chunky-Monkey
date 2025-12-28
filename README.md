# 🐵 Chunky Monkey

**Universal semantic text chunking for LLM/RAG applications**

> *Don't split text. Encapsulate meaning.*

[![PyPI version](https://badge.fury.io/py/chunky-monkey.svg)](https://badge.fury.io/py/chunky-monkey)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Current text splitters (LangChain, LlamaIndex) are **"dumb"**—they cut text at arbitrary character counts, which:

- ❌ Severs sentences mid-thought → LLM confusion
- ❌ Separates pronouns from their references → "He pressed the button" (who?)
- ❌ Breaks tables, code blocks, and lists → corrupted data
- ❌ Loses document structure → no context for retrieval

**Result: Hallucinations, retrieval failures, and garbage outputs.**

## The Solution

Chunky Monkey is a **structural-integrity-first** chunking library that:

- ✅ Never breaks mid-sentence (Unicode-aware segmentation)
- ✅ Preserves code blocks, tables, and lists as atomic units
- ✅ Propagates document structure (headers → chunk metadata)
- ✅ Works with ANY tokenizer (OpenAI, HuggingFace, Claude, Llama)
- ✅ Validates chunk integrity before output
- ✅ Zero required dependencies (extras for advanced features)

---

## Installation

```bash
# Core library (zero dependencies)
pip install chunky-monkey

# With OpenAI tokenizer support
pip install chunky-monkey[tiktoken]

# With semantic chunking (topic detection)
pip install chunky-monkey[semantic]

# Everything
pip install chunky-monkey[all]
```

---

## Quick Start

```python
from monkey import chunk

text = """
# Introduction

Machine learning is transforming industries. The technology enables 
computers to learn from data without explicit programming.

# Applications

Healthcare uses ML for diagnosis. Finance uses it for fraud detection.
The possibilities are endless.
"""

# Simple usage - chunks text respecting sentence boundaries
chunks = chunk(text, max_tokens=100)

for c in chunks:
    print(f"Chunk {c.chunk_id[:8]}: {c.text[:50]}...")
    print(f"  Tokens: {c.token_counts}")
    print(f"  Section: {c.metadata.section_header}")
```

---

## Advanced Usage

```python
from monkey import ChunkyMonkey
from monkey.tokenizers import TiktokenTokenizer
from monkey.parsers import MarkdownParser

# Full configuration
chunker = ChunkyMonkey(
    tokenizer=TiktokenTokenizer(model="gpt-4"),
    parser=MarkdownParser(),
    max_tokens=500,
    overlap_tokens=50,
    preserve_code_blocks=True,
    validate=True
)

chunks = chunker.chunk(document)

# Each chunk includes rich metadata
for chunk in chunks:
    print(chunk.to_dict())  # JSON-serializable output
```

---

## Why Chunky Monkey?

| Feature | LangChain | LlamaIndex | Unstructured | **Chunky Monkey** |
|---------|-----------|------------|--------------|-------------------|
| Universal tokenizer support | ⚠️ | ⚠️ | ❌ | ✅ |
| Never breaks sentences | ❌ | ❌ | ⚠️ | ✅ |
| Preserves code blocks | ❌ | ⚠️ | ✅ | ✅ |
| Metadata propagation | ❌ | ⚠️ | ✅ | ✅ |
| Integrity validation | ❌ | ❌ | ❌ | ✅ |
| Zero dependencies | ❌ | ❌ | ❌ | ✅ |
| Deterministic output | ❌ | ❌ | ❌ | ✅ |

---

## Documentation

- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Examples](examples/)

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.
