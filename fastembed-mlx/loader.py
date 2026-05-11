"""
Weight loading from HuggingFace safetensors with MLX quantization support.
"""
import json
from pathlib import Path
from typing import Dict, Tuple
import mlx.core as mx
from huggingface_hub import hf_hub_download
from .nn import BertModel
from .config import ModelConfig, get_config

def _download_files(repo_id: str, local_dir: Path) -> Tuple[Path, Path]:
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=str(local_dir))
    try:
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=str(local_dir))
    except Exception:
        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", local_dir=str(local_dir))
    return Path(config_path), Path(weights_path)

def _load_safetensors(path: Path) -> Dict[str, mx.array]:
    return mx.load(str(path))

def _load_pytorch_bin(path: Path) -> Dict[str, mx.array]:
    import torch
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k: mx.array(v.numpy()) for k, v in state_dict.items()}

def _map_hf_to_mlx_weights(hf_weights: Dict[str, mx.array], config: ModelConfig) -> Dict[str, mx.array]:
    mapped = {}
    for hf_name, weight in hf_weights.items():
        mlx_name = hf_name[5:] if hf_name.startswith("bert.") else hf_name
        # Skip pooler and position_ids (MLX generates positions dynamically)
        if mlx_name.startswith("pooler") or mlx_name == "embeddings.position_ids":
            continue
        mapped[mlx_name] = weight
    return mapped

def load_model(model_name: str, cache_dir: str = "~/.cache/fastembed_mlx") -> Tuple[BertModel, ModelConfig]:
    config = get_config(model_name)
    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    config_path, weights_path = _download_files(model_name, cache_path)
    with open(config_path) as f:
        hf_config = json.load(f)
    assert hf_config["hidden_size"] == config.hidden_size
    assert hf_config["num_hidden_layers"] == config.num_hidden_layers
    if weights_path.suffix == ".safetensors":
        hf_weights = _load_safetensors(weights_path)
    else:
        hf_weights = _load_pytorch_bin(weights_path)
    mlx_weights = _map_hf_to_mlx_weights(hf_weights, config)
    model = BertModel(
        vocab_size=config.vocab_size, hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size, max_position_embeddings=config.max_position_embeddings,
        layer_norm_eps=config.layer_norm_eps, hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
    )
    model.load_weights(list(mlx_weights.items()))
    return model, config
