#!/usr/bin/env python3
"""
Benchmark batch embedding throughput.
Compare MLX vs CPU ONNX performance on Apple Silicon.
"""

import time
import numpy as np

from fastembed_mlx import MLXTextEmbedding


def benchmark(embedder, texts, batch_size=32, warmup=2, runs=5):
    for _ in range(warmup):
        list(embedder.embed(texts[:batch_size]))

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        list(embedder.embed(texts, batch_size=batch_size))
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = len(texts) / avg_time

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "throughput": throughput,
        "times": times,
    }


def main():
    n_docs = 1000
    doc_length = 50

    print(f"Generating {n_docs} synthetic documents (~{doc_length} words each)...")
    vocab = [
        "the", "a", "is", "are", "was", "were", "machine", "learning",
        "artificial", "intelligence", "neural", "network", "deep", "model",
        "training", "data", "algorithm", "computer", "vision", "language",
    ]

    import random
    random.seed(42)
    documents = [
        " ".join(random.choices(vocab, k=doc_length))
        for _ in range(n_docs)
    ]

    print("\n=== MLX Backend (Metal GPU) ===")
    mlx_embedder = MLXTextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        compile=True,
    )

    for batch_size in [16, 32, 64, 128]:
        result = benchmark(mlx_embedder, documents, batch_size=batch_size)
        print(
            f"  batch={batch_size:3d}: {result['throughput']:.1f} docs/sec "
            f"(avg {result['avg_time']:.3f}s +/- {result['std_time']:.3f}s)"
        )

    try:
        from fastembed import TextEmbedding
        print("\n=== FastEmbed ONNX (CPU) ===")
        onnx_embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

        for batch_size in [16, 32]:
            result = benchmark(onnx_embedder, documents, batch_size=batch_size)
            print(
                f"  batch={batch_size:3d}: {result['throughput']:.1f} docs/sec "
                f"(avg {result['avg_time']:.3f}s +/- {result['std_time']:.3f}s)"
            )
    except ImportError:
        print("\nFastEmbed not installed. Install with: pip install fastembed")


if __name__ == "__main__":
    main()
