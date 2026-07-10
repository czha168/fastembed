"""
BERT multi-head self-attention.
Architecture: hidden_size=384, num_heads=12 -> head_dim=32
"""
import math
from typing import Optional
import mlx.core as mx
import mlx.nn as nn

class BertSelfAttention(nn.Module):
    def __init__(
        self, hidden_size: int = 384, num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: mx.array) -> mx.array:
        batch_size, seq_length = x.shape[:2]
        x = x.reshape(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return mx.transpose(x, (0, 2, 1, 3))

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        batch_size, seq_length = hidden_states.shape[:2]
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)
        attention_scores = mx.matmul(Q, mx.transpose(K, (0, 1, 3, 2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)
        context = mx.matmul(attention_probs, V)
        context = mx.transpose(context, (0, 2, 1, 3))
        context = context.reshape(batch_size, seq_length, self.all_head_size)
        return context

class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size: int = 384, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)

class BertAttention(nn.Module):
    def __init__(
        self, hidden_size: int = 384, num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1, layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size, layer_norm_eps)

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        return self.output(self.self(hidden_states, attention_mask), hidden_states)
