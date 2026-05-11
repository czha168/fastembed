"""
Integration tests for MLXTextEmbedding end-to-end.
"""

import unittest
import numpy as np

from fastembed_mlx import MLXTextEmbedding


class TestMLXTextEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = MLXTextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            compile=False,
        )

    def test_single_embedding(self):
        emb = self.embedder.query_embed("Hello world")
        self.assertEqual(emb.shape, (384,))
        self.assertEqual(emb.dtype, np.float32)

    def test_batch_embedding(self):
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = list(self.embedder.embed(texts))
        self.assertEqual(len(embeddings), 3)
        for emb in embeddings:
            self.assertEqual(emb.shape, (384,))

    def test_embedding_normalization(self):
        emb = self.embedder.query_embed("Test")
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_similarity_consistency(self):
        emb1 = self.embedder.query_embed("I like apples")
        emb2 = self.embedder.query_embed("I like oranges")
        emb3 = self.embedder.query_embed("The weather is nice today")
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)
        self.assertGreater(sim_12, sim_13)

    def test_batch_array_output(self):
        texts = ["A", "B", "C"]
        arr = self.embedder.embed_batch(texts)
        self.assertEqual(arr.shape, (3, 384))


if __name__ == "__main__":
    unittest.main()
