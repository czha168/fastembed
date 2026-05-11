#!/usr/bin/env python3
"""
Basic usage example for fastembed_mlx.
"""

from fastembed_mlx import MLXTextEmbedding


def main():
    print("Loading model...")
    embedder = MLXTextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.dim}")
    print(f"Max length: {embedder.max_length}")
    print()

    # Single query embedding
    query = "What is machine learning?"
    query_emb = embedder.query_embed(query)
    print(f'Query: "{query}"')
    print(f"Embedding shape: {query_emb.shape}")
    print(f"Embedding norm: {sum(x**2 for x in query_emb)**0.5:.6f}")
    print()

    # Batch embedding
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
    ]

    print("Batch embedding documents...")
    embeddings = list(embedder.embed(documents))

    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        print(f"  [{i}] {doc[:50]}... -> shape {emb.shape}")

    print()

    # Compute similarities
    import numpy as np
    query_emb = embedder.query_embed(query)
    doc_embeddings = np.array(embeddings)

    similarities = np.dot(doc_embeddings, query_emb)
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    print("Ranked by similarity to query:")
    for rank, (idx, score) in enumerate(ranked, 1):
        print(f"  {rank}. [{score:.4f}] {documents[idx][:60]}...")


if __name__ == "__main__":
    main()
