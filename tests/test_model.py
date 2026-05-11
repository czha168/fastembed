"""
Unit tests for MLX BERT model architecture.
"""

import unittest
import mlx.core as mx

from fastembed_mlx.nn import BertModel, BertEmbeddings, BertEncoder
from fastembed_mlx.nn.attention import BertSelfAttention
from fastembed_mlx.nn.feedforward import BertIntermediate, BertOutput
from fastembed_mlx.nn.transformer import BertLayer


class TestBertEmbeddings(unittest.TestCase):
    def test_shape(self):
        emb = BertEmbeddings(vocab_size=100, hidden_size=32, max_position_embeddings=16)
        input_ids = mx.zeros((2, 8), dtype=mx.int32)
        out = emb(input_ids)
        self.assertEqual(out.shape, (2, 8, 32))


class TestBertSelfAttention(unittest.TestCase):
    def test_shape(self):
        attn = BertSelfAttention(hidden_size=384, num_attention_heads=12)
        hidden = mx.random.normal((2, 10, 384))
        out = attn(hidden)
        self.assertEqual(out.shape, (2, 10, 384))

    def test_with_mask(self):
        attn = BertSelfAttention(hidden_size=384, num_attention_heads=12)
        hidden = mx.random.normal((2, 10, 384))
        mask = mx.zeros((2, 1, 1, 10))
        out = attn(hidden, mask)
        self.assertEqual(out.shape, (2, 10, 384))


class TestBertLayer(unittest.TestCase):
    def test_shape(self):
        layer = BertLayer(hidden_size=384, num_attention_heads=12, intermediate_size=1536)
        hidden = mx.random.normal((2, 10, 384))
        out = layer(hidden)
        self.assertEqual(out.shape, (2, 10, 384))


class TestBertModel(unittest.TestCase):
    def test_full_forward(self):
        model = BertModel(
            vocab_size=30522, hidden_size=384, num_hidden_layers=6,
            num_attention_heads=12, intermediate_size=1536, max_position_embeddings=512,
        )
        input_ids = mx.zeros((2, 10), dtype=mx.int32)
        attention_mask = mx.ones((2, 10), dtype=mx.int32)
        out = model(input_ids, attention_mask)
        self.assertEqual(out.shape, (2, 10, 384))

    def test_attention_mask_effect(self):
        model = BertModel(
            vocab_size=30522, hidden_size=384, num_hidden_layers=6,
            num_attention_heads=12, intermediate_size=1536,
        )
        input_ids = mx.zeros((1, 5), dtype=mx.int32)
        mask_full = mx.ones((1, 5), dtype=mx.int32)
        out_full = model(input_ids, mask_full)
        mask_partial = mx.array([[1, 1, 1, 0, 0]], dtype=mx.int32)
        out_partial = model(input_ids, mask_partial)
        self.assertFalse(mx.allclose(out_full, out_partial))


if __name__ == "__main__":
    unittest.main()
