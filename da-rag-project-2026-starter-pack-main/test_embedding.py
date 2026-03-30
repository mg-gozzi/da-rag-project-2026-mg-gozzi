#!/usr/bin/env python3
"""Quick test of embedding functionality."""

print('Testing embedding functionality...')
from embeddings import create_embedding, get_embedding_info

# Test basic embedding info
info = get_embedding_info()
print('Embedding info:', info)

# Test with a simple text
try:
    print('Creating embedding for test text...')
    embedding = create_embedding('Hello world')
    print(f'Embedding created successfully! Length: {len(embedding)}')
    print(f'First 5 values: {embedding[:5]}')
except Exception as e:
    print(f'Error creating embedding: {e}')
    import traceback
    traceback.print_exc()