"""
Model configurations from HuggingFace specs.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    dim: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    pooling: str
    normalization: bool
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    type_vocab_size: int = 2
    quantization: Optional[Dict[str, Any]] = None

SUPPORTED_MODELS: Dict[str, ModelConfig] = {
    "mlx-community/all-MiniLM-L6-v2-4bit": ModelConfig(
        model_id="mlx-community/all-MiniLM-L6-v2-4bit",
        dim=384,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=512,
        vocab_size=30522,
        pooling="mean",
        normalization=True,
        quantization={"bits": 4, "group_size": 64},
    ),
    "sentence-transformers/all-MiniLM-L6-v2": ModelConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=512,
        vocab_size=30522,
        pooling="mean",
        normalization=True,
    ),
    # Append the BGE-Small-en-v1.5 4-bit configuration here
    "mlx-community/bge-small-en-v1.5-4bit": ModelConfig(
        model_id="mlx-community/bge-small-en-v1.5-4bit",
        dim=384,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=512,
        vocab_size=30522,
        pooling="first",  # BGE uses CLS token pooling ("first") instead of mean pooling
        normalization=True,
        quantization={"bits": 4, "group_size": 64},
    ),
}

def get_config(model_name: str) -> ModelConfig:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Available: {list(SUPPORTED_MODELS.keys())}")
    return SUPPORTED_MODELS[model_name]
