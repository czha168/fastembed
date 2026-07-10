"""
Quantized BERT model using MLX QuantizedLinear and QuantizedEmbedding.
For 4-bit quantized models from HuggingFace.
"""
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from .embeddings import BertEmbeddings


class QuantizedBertEmbeddings(nn.Module):
    """BERT embeddings with all quantized embeddings."""
    def __init__(
        self, vocab_size: int = 30522, hidden_size: int = 384,
        max_position_embeddings: int = 512, type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12, hidden_dropout_prob: float = 0.1,
        group_size: int = 64, bits: int = 4,
    ):
        super().__init__()
        # All embeddings are quantized
        self.word_embeddings = nn.QuantizedEmbedding(
            vocab_size, hidden_size, group_size=group_size, bits=bits
        )
        self.position_embeddings = nn.QuantizedEmbedding(
            max_position_embeddings, hidden_size, group_size=group_size, bits=bits
        )
        self.token_type_embeddings = nn.QuantizedEmbedding(
            type_vocab_size, hidden_size, group_size=group_size, bits=bits
        )
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


class QuantizedBertSelfAttention(nn.Module):
    """Self-attention with quantized QKV projections."""
    def __init__(
        self, hidden_size: int = 384, num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1,
        group_size: int = 64, bits: int = 4,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        # QKV projections are quantized
        self.query = nn.QuantizedLinear(hidden_size, hidden_size, group_size=group_size, bits=bits)
        self.key = nn.QuantizedLinear(hidden_size, hidden_size, group_size=group_size, bits=bits)
        self.value = nn.QuantizedLinear(hidden_size, hidden_size, group_size=group_size, bits=bits)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: mx.array) -> mx.array:
        batch_size, seq_length = x.shape[:2]
        x = x.reshape(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return mx.transpose(x, (0, 2, 1, 3))

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        import math
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


class QuantizedBertSelfOutput(nn.Module):
    """Self-output with quantized dense layer."""
    def __init__(
        self, hidden_size: int = 384, layer_norm_eps: float = 1e-12,
        group_size: int = 64, bits: int = 4,
    ):
        super().__init__()
        self.dense = nn.QuantizedLinear(hidden_size, hidden_size, group_size=group_size, bits=bits)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class QuantizedBertAttention(nn.Module):
    """Attention module with quantized layers."""
    def __init__(
        self, hidden_size: int = 384, num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1, layer_norm_eps: float = 1e-12,
        group_size: int = 64, bits: int = 4,
    ):
        super().__init__()
        self.self = QuantizedBertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, group_size, bits)
        self.output = QuantizedBertSelfOutput(hidden_size, layer_norm_eps, group_size, bits)

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        return self.output(self.self(hidden_states, attention_mask), hidden_states)


class QuantizedBertIntermediate(nn.Module):
    """Feed-forward intermediate with quantized expansion."""
    def __init__(
        self, hidden_size: int = 384, intermediate_size: int = 1536,
        group_size: int = 64, bits: int = 4,
    ):
        super().__init__()
        self.dense = nn.QuantizedLinear(hidden_size, intermediate_size, group_size=group_size, bits=bits)
        self.activation = nn.GELU()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.activation(self.dense(hidden_states))


class QuantizedBertOutput(nn.Module):
    """Feed-forward output with quantized projection."""
    def __init__(
        self, hidden_size: int = 384, intermediate_size: int = 1536,
        layer_norm_eps: float = 1e-12,
        group_size: int = 64, bits: int = 4,
    ):
        super().__init__()
        self.dense = nn.QuantizedLinear(intermediate_size, hidden_size, group_size=group_size, bits=bits)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, intermediate: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(intermediate)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class QuantizedBertLayer(nn.Module):
    """BERT transformer layer with quantized linear layers."""
    def __init__(
        self, hidden_size: int = 384, num_attention_heads: int = 12,
        intermediate_size: int = 1536, attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        group_size: int = 64, bits: int = 4,
    ):
        super().__init__()
        self.attention = QuantizedBertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, layer_norm_eps, group_size, bits)
        self.intermediate = QuantizedBertIntermediate(hidden_size, intermediate_size, group_size, bits)
        self.output = QuantizedBertOutput(hidden_size, intermediate_size, layer_norm_eps, group_size, bits)

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class QuantizedBertEncoder(nn.Module):
    """BERT encoder stack with quantized layers."""
    def __init__(
        self, num_hidden_layers: int = 6, hidden_size: int = 384,
        num_attention_heads: int = 12, intermediate_size: int = 1536,
        attention_probs_dropout_prob: float = 0.1, layer_norm_eps: float = 1e-12,
        group_size: int = 64, bits: int = 4,
    ):
        super().__init__()
        self.layer = [
            QuantizedBertLayer(hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob, layer_norm_eps, group_size, bits)
            for _ in range(num_hidden_layers)
        ]

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class QuantizedBertModel(nn.Module):
    """Full BERT model with quantized layers for 4-bit weights."""
    def __init__(
        self, vocab_size: int = 30522, hidden_size: int = 384,
        num_hidden_layers: int = 6, num_attention_heads: int = 12,
        intermediate_size: int = 1536, max_position_embeddings: int = 512,
        type_vocab_size: int = 2, layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1, attention_probs_dropout_prob: float = 0.1,
        group_size: int = 64, bits: int = 4,
    ):
        super().__init__()
        self.embeddings = QuantizedBertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
            layer_norm_eps, hidden_dropout_prob, group_size, bits
        )
        self.encoder = QuantizedBertEncoder(
            num_hidden_layers, hidden_size, num_attention_heads, intermediate_size,
            attention_probs_dropout_prob, layer_norm_eps, group_size, bits
        )

    def _build_attention_mask(self, attention_mask: Optional[mx.array]) -> Optional[mx.array]:
        if attention_mask is None:
            return None
        mask = (1.0 - attention_mask.astype(mx.float32)) * -1e9
        return mask[:, None, None, :]

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        extended_mask = self._build_attention_mask(attention_mask)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        return self.encoder(embedding_output, extended_mask)
