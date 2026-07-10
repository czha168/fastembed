"""
Full BERT model: Embeddings -> Encoder.
"""
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from .embeddings import BertEmbeddings
from .transformer import BertEncoder

class BertModel(nn.Module):
    def __init__(
        self, vocab_size: int = 30522, hidden_size: int = 384,
        num_hidden_layers: int = 6, num_attention_heads: int = 12,
        intermediate_size: int = 1536, max_position_embeddings: int = 512,
        type_vocab_size: int = 2, layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1, attention_probs_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, layer_norm_eps, hidden_dropout_prob)
        self.encoder = BertEncoder(num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob, layer_norm_eps)

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
