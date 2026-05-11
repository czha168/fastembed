"""
Model factory with JIT compilation.
"""
from typing import Tuple
from .nn import BertModel
from .config import ModelConfig
from .loader import load_model

def get_mlx_model(model_name: str, compile: bool = True, cache_dir: str = "~/.cache/fastembed_mlx") -> Tuple[BertModel, ModelConfig]:
    model, config = load_model(model_name, cache_dir)
    if compile:
        import mlx.core as mx
        model = mx.compile(model)
    return model, config
