"""
BERT feed-forward: expansion (H->4H) with GELU, projection (4H->H).
"""
import mlx.core as mx
import mlx.nn as nn

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size: int = 384, intermediate_size: int = 1536):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.activation(self.dense(hidden_states))

class BertOutput(nn.Module):
    def __init__(
        self, hidden_size: int = 384, intermediate_size: int = 1536,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, intermediate: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(intermediate)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)
