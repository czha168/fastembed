"""
BERT transformer block and encoder stack.
"""
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from .attention import BertAttention
from .feedforward import BertIntermediate, BertOutput

class BertLayer(nn.Module):
    def __init__(
        self, hidden_size: int = 384, num_attention_heads: int = 12,
        intermediate_size: int = 1536, attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, layer_norm_eps)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, layer_norm_eps)

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)

class BertEncoder(nn.Module):
    def __init__(
        self, num_hidden_layers: int = 6, hidden_size: int = 384,
        num_attention_heads: int = 12, intermediate_size: int = 1536,
        attention_probs_dropout_prob: float = 0.1, layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.layer = [
            BertLayer(hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob, layer_norm_eps)
            for _ in range(num_hidden_layers)
        ]

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
