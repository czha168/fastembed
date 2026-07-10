"""
BERT embedding layer: token + position + segment embeddings.
Architecture: Devlin et al. (2018)
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Optional

class BertEmbeddings(nn.Module):
    def __init__(
        self, vocab_size: int = 30522, hidden_size: int = 384,
        max_position_embeddings: int = 512, type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12, hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def __call__(
        self, input_ids: mx.array, token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_length = input_ids.shape
        words = self.word_embeddings(input_ids)
        if position_ids is None:
            position_ids = mx.arange(seq_length, dtype=mx.int32)
            position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))
        positions = self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)
        token_types = self.token_type_embeddings(token_type_ids)
        embeddings = words + positions + token_types
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
