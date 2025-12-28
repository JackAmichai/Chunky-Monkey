"""
End-to-End RAG Pipeline Example - Chunky Monkey
================================================

This example demonstrates a complete workflow:
1. Load a Markdown document
2. Parse structure (headers, code blocks)
3. Chunk with token limits
4. Validate integrity
5. Export to JSON for vector database ingestion

This produces output compatible with LangChain, LlamaIndex,
and direct vector DB ingestion (Pinecone, Weaviate, Chroma, etc.).
"""

import json
from pathlib import Path

from monkey import ChunkyMonkey, Chunk
from monkey.parsers import MarkdownParser
from monkey.tokenizers import CharacterTokenizer
from monkey.validators import (
    validate_token_limits,
    validate_roundtrip,
    flag_dangling_references,
)


# =============================================================================
# SAMPLE DOCUMENT
# =============================================================================

SAMPLE_DOCUMENT = """
# API Reference: User Authentication

This document describes the authentication endpoints for our REST API.

## Overview

Our API uses JWT (JSON Web Tokens) for authentication. All protected 
endpoints require a valid token in the Authorization header.

### Token Lifecycle

Tokens are valid for 24 hours after issuance. Refresh tokens can be 
used to obtain new access tokens without re-authentication.

## Endpoints

### POST /auth/login

Authenticates a user and returns access and refresh tokens.

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "secretpassword"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGc...",
  "refresh_token": "dGhpcyB...",
  "expires_in": 86400
}
```

### POST /auth/refresh

Obtains a new access token using a refresh token.

**Request Body:**

```json
{
  "refresh_token": "dGhpcyB..."
}
```

**Response:**

```json
{
  "access_token": "eyJhbGc...",
  "expires_in": 86400
}
```

### POST /auth/logout

Invalidates the current session and refresh token.

**Headers:**
- `Authorization: Bearer <access_token>`

**Response:** 204 No Content

## Error Handling

All authentication errors return standard error responses:

```json
{
  "error": "invalid_credentials",
  "message": "Email or password is incorrect",
  "status": 401
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| 401  | Invalid credentials or expired token |
| 403  | Insufficient permissions |
| 429  | Too many requests (rate limited) |

## Rate Limiting

Authentication endpoints are rate limited to prevent brute force attacks:

- Login: 5 attempts per minute per IP
- Refresh: 10 requests per minute per user
- Other endpoints: 100 requests per minute per user

Exceeding these limits returns a 429 status code.

## Security Best Practices

1. Always use HTTPS in production
2. Store tokens securely (not in localStorage)
3. Implement proper CORS policies
4. Use short-lived access tokens
5. Rotate refresh tokens on use
"""


def main():
    print("=" * 70)
    print("END-TO-END RAG PIPELINE EXAMPLE")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: CONFIGURE THE CHUNKER
    # =========================================================================
    print("\n📋 Step 1: Configure chunker")
    
    # Use CharacterTokenizer for this example
    # In production, use TiktokenTokenizer for OpenAI or appropriate tokenizer
    tokenizer = CharacterTokenizer(chars_per_token=4)  # Approximate
    
    chunker = ChunkyMonkey(
        tokenizer=tokenizer,
        parser=MarkdownParser(),
        max_tokens=200,      # ~800 characters
        overlap_tokens=30,   # ~120 characters overlap
        preserve_code_blocks=True,
        validate=True,
        source="api_auth_docs.md"
    )
    
    print(f"  Tokenizer: {tokenizer.name}")
    print(f"  Max tokens: 200")
    print(f"  Overlap: 30 tokens")
    
    # =========================================================================
    # STEP 2: CHUNK THE DOCUMENT
    # =========================================================================
    print("\n📄 Step 2: Chunk document")
    
    chunks = chunker.chunk(SAMPLE_DOCUMENT)
    
    print(f"  Input: {len(SAMPLE_DOCUMENT)} characters")
    print(f"  Output: {len(chunks)} chunks")
    
    # =========================================================================
    # STEP 3: VALIDATE CHUNKS
    # =========================================================================
    print("\n✅ Step 3: Validate chunks")
    
    # Check token limits
    token_errors = validate_token_limits(chunks, max_tokens=200, tokenizer=tokenizer)
    print(f"  Token limit violations: {len(token_errors)}")
    
    # Check for content loss
    roundtrip_errors = validate_roundtrip(SAMPLE_DOCUMENT, chunks, tolerance=0.8)
    print(f"  Roundtrip errors: {len(roundtrip_errors)}")
    
    # Check for dangling references
    reference_warnings = flag_dangling_references(chunks)
    print(f"  Dangling reference warnings: {len(reference_warnings)}")
    
    if reference_warnings:
        print("\n  ⚠️ Dangling references detected:")
        for w in reference_warnings[:3]:
            print(f"    - Chunk {w.chunk_id[:8]}: {w.message}")
    
    # =========================================================================
    # STEP 4: PREPARE FOR VECTOR DATABASE
    # =========================================================================
    print("\n📦 Step 4: Prepare for vector database")
    
    # Format for common vector databases
    vector_db_records = []
    
    for chunk in chunks:
        record = {
            # Unique identifier
            "id": chunk.chunk_id,
            
            # The text to embed
            "text": chunk.text,
            
            # Metadata for filtering and display
            "metadata": {
                "source": chunk.metadata.source,
                "section": chunk.metadata.section_header,
                "chunk_index": chunk.metadata.chunk_index,
                "total_chunks": chunk.metadata.total_chunks,
                "char_start": chunk.metadata.position[0] if chunk.metadata.position else None,
                "char_end": chunk.metadata.position[1] if chunk.metadata.position else None,
            }
        }
        vector_db_records.append(record)
    
    print(f"  Records prepared: {len(vector_db_records)}")
    
    # =========================================================================
    # STEP 5: EXPORT
    # =========================================================================
    print("\n💾 Step 5: Export")
    
    # Export as JSON (for Chroma, Weaviate, etc.)
    export_data = {
        "document": {
            "source": "api_auth_docs.md",
            "title": "API Reference: User Authentication",
            "total_chunks": len(chunks),
        },
        "chunks": vector_db_records
    }
    
    # Pretty print sample
    print("\n  Sample record (first chunk):")
    sample = json.dumps(vector_db_records[0], indent=4)
    for line in sample.split('\n')[:15]:
        print(f"    {line}")
    print("    ...")
    
    # =========================================================================
    # STEP 6: USAGE EXAMPLES
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTEGRATION EXAMPLES")
    print("=" * 70)
    
    print("""
📌 LangChain Integration:

    from langchain.schema import Document
    
    langchain_docs = [
        Document(
            page_content=record["text"],
            metadata=record["metadata"]
        )
        for record in vector_db_records
    ]
    
    # Then use with any LangChain vector store
    vectorstore = Chroma.from_documents(langchain_docs, embeddings)

📌 LlamaIndex Integration:

    from llama_index.core import Document
    
    llama_docs = [
        Document(
            text=record["text"],
            metadata=record["metadata"]
        )
        for record in vector_db_records
    ]
    
    index = VectorStoreIndex.from_documents(llama_docs)

📌 Direct Pinecone:

    import pinecone
    
    vectors = [
        {
            "id": record["id"],
            "values": embed(record["text"]),  # Your embedding function
            "metadata": record["metadata"]
        }
        for record in vector_db_records
    ]
    
    index.upsert(vectors=vectors)
""")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
✓ Document chunked: {len(SAMPLE_DOCUMENT)} chars → {len(chunks)} chunks
✓ Validation passed: {len(token_errors)} errors, {len(reference_warnings)} warnings  
✓ Export ready: {len(vector_db_records)} records for vector DB

Each chunk contains:
  - Unique deterministic ID
  - Text content
  - Source document reference
  - Section header context
  - Position in original document
  - Chunk sequence info
""")
    
    return export_data


if __name__ == "__main__":
    main()
