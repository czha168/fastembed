"""
Weight loading from HuggingFace safetensors with MLX quantization support.
"""
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import mlx.core as mx
from huggingface_hub import hf_hub_download
from .config import ModelConfig, get_config


def _download_files(repo_id: str, local_dir: Path) -> Tuple[Path, Path]:
    """Download model config and weights from HuggingFace."""
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=str(local_dir))
    try:
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=str(local_dir))
    except Exception:
        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", local_dir=str(local_dir))
    return Path(config_path), Path(weights_path)


def _load_safetensors(path: Path) -> Dict[str, mx.array]:
    """Load safetensors file - handles quantized weights natively."""
    return mx.load(str(path))


def _load_pytorch_bin(path: Path) -> Dict[str, mx.array]:
    """Load PyTorch bin file and convert to MLX arrays."""
    import torch
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k: mx.array(v.numpy()) for k, v in state_dict.items()}


def _map_hf_to_mlx_weights(hf_weights: Dict[str, mx.array], config: ModelConfig) -> Dict[str, mx.array]:
    """
    Map HuggingFace weight names to MLX model weight names.

    For quantized models, MLX's QuantizedLinear and QuantizedEmbedding expect
    the quantized weights directly with their scales and biases.
    """
    mapped = {}

    for hf_name, weight in hf_weights.items():
        # Remove "bert." prefix if present
        mlx_name = hf_name[5:] if hf_name.startswith("bert.") else hf_name

        # Skip pooler and position_ids (MLX generates positions dynamically)
        if mlx_name.startswith("pooler") or mlx_name == "embeddings.position_ids":
            continue

        mapped[mlx_name] = weight

    return mapped

def load_model(model_name: str, cache_dir: str = "~/.cache/fastembed_mlx"):
    """
    Load a BERT model with support for quantized weights.

    Returns:
        Tuple of (model, config) where model is either QuantizedBertModel or BertModel
        depending on whether the model uses quantization.
    """
    from .nn.bert import BertModel
    from .nn.quantized_bert import QuantizedBertModel
    from .config import resolve_mlx_model_name

    model_name = resolve_mlx_model_name(model_name)
    config = get_config(model_name)
    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)

    # Download model files
    config_path, weights_path = _download_files(model_name, cache_path)

    # Load HF config for validation
    with open(config_path) as f:
        hf_config = json.load(f)
    assert hf_config["hidden_size"] == config.hidden_size
    assert hf_config["num_hidden_layers"] == config.num_hidden_layers

    # Load weights
    if weights_path.suffix == ".safetensors":
        hf_weights = _load_safetensors(weights_path)
    else:
        hf_weights = _load_pytorch_bin(weights_path)

    # Map HF weight names to MLX names
    mlx_weights = _map_hf_to_mlx_weights(hf_weights, config)

    # Create model based on quantization config
    is_quantized = config.quantization is not None
    if is_quantized:
        group_size = config.quantization.get("group_size", 64)
        bits = config.quantization.get("bits", 4)
        model = QuantizedBertModel(
            vocab_size=config.vocab_size, hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size, max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps, hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            group_size=group_size, bits=bits,
        )
    else:
        model = BertModel(
            vocab_size=config.vocab_size, hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size, max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps, hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        )

    # Load weights into model
    model.load_weights(list(mlx_weights.items()))

    return model, config
