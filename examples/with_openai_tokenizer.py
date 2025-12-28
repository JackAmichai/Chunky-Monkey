"""
OpenAI Tokenizer Example - Chunky Monkey
=========================================

This example shows how to use the TiktokenTokenizer for accurate
token counting with OpenAI models (GPT-4, GPT-3.5, etc.).

Requirements:
    pip install chunky-monkey[tiktoken]
"""

try:
    from monkey import ChunkyMonkey
    from monkey.tokenizers import TiktokenTokenizer
except ImportError:
    print("Please install with: pip install chunky-monkey[tiktoken]")
    exit(1)

# Sample document
document = """
# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed. The field has seen explosive growth in recent years.

## Supervised Learning

In supervised learning, algorithms learn from labeled training data. 
The model makes predictions based on input features and is corrected 
when predictions are wrong. Common algorithms include:

- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines

## Unsupervised Learning

Unsupervised learning works with unlabeled data. The algorithm tries 
to find hidden patterns or structures in the data. Clustering and 
dimensionality reduction are common applications.

## Deep Learning

Deep learning uses neural networks with many layers. These models can 
learn complex patterns and have achieved remarkable results in image 
recognition, natural language processing, and game playing.
"""

print("=" * 60)
print("OPENAI TOKENIZER EXAMPLE")
print("=" * 60)

# Create tokenizer for GPT-4
gpt4_tokenizer = TiktokenTokenizer(model="gpt-4")

print(f"\nTokenizer: {gpt4_tokenizer.name}")
print(f"Encoding: {gpt4_tokenizer.encoding_name}")

# Show token count difference vs characters
char_count = len(document)
token_count = gpt4_tokenizer.count_tokens(document)
print(f"\nDocument stats:")
print(f"  Characters: {char_count}")
print(f"  GPT-4 tokens: {token_count}")
print(f"  Ratio: {char_count / token_count:.2f} chars/token")

# Chunk with accurate GPT-4 token limits
chunker = ChunkyMonkey(
    tokenizer=gpt4_tokenizer,
    max_tokens=150,  # Accurate GPT-4 token limit
    overlap_tokens=25,
    validate=True
)

chunks = chunker.chunk(document)

print(f"\nChunks created: {len(chunks)}")
print()

for i, chunk in enumerate(chunks):
    tokens = gpt4_tokenizer.count_tokens(chunk.text)
    print(f"Chunk {i + 1}:")
    print(f"  Tokens: {tokens}")
    print(f"  Preview: {chunk.text[:60].strip()}...")
    print()

# Compare different models
print("=" * 60)
print("COMPARING TOKENIZERS")
print("=" * 60)

sample = "Hello, how are you today? I hope you're doing well!"

models = [
    ("gpt-4", "cl100k_base"),
    ("gpt-4o", "o200k_base"),
    ("gpt-3.5-turbo", "cl100k_base"),
]

print(f"\nSample text: \"{sample}\"")
print()

for model, expected_encoding in models:
    try:
        tok = TiktokenTokenizer(model=model)
        count = tok.count_tokens(sample)
        print(f"  {model:20} ({tok.encoding_name}): {count} tokens")
    except Exception as e:
        print(f"  {model:20}: {e}")

print("\n✓ OpenAI tokenizer example complete!")
