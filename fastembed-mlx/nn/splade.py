# fastembed-mlx/nn/splade.py
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from .bert import BertModel

class MlxSpladeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def __call__(
        self, 
        input_ids: mx.array, 
        attention_mask: Optional[mx.array] = None, 
        token_type_ids: Optional[mx.array] = None
    ) -> mx.array:
        sequence_output = self.bert(input_ids, attention_mask, token_type_ids)
        
        x = self.transform(sequence_output)
        x = mx.gelu(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        
        relu_log = mx.log(1.0 + mx.maximum(logits, 0.0))
        
        if attention_mask is not None:
            expanded_mask = mx.expand_dims(attention_mask, -1)
            weighted_log = relu_log * expanded_mask
        else:
            weighted_log = relu_log
        
        return mx.max(weighted_log, axis=1)
