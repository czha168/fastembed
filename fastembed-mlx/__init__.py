"""
fastembed_mlx: Clean-room MLX embedding backend for Apple Silicon.
"""
from .embedder import MLXTextEmbedding
from .config import ModelConfig, SUPPORTED_MODELS

__version__ = "0.1.0"
__all__ = ["MLXTextEmbedding", "ModelConfig", "SUPPORTED_MODELS"]
