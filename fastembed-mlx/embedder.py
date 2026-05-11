"""
MLXTextEmbedding: FastEmbed-compatible API for Apple Silicon.
"""
from typing import Iterator, List, Optional
import numpy as np
import mlx.core as mx
from .config import ModelConfig, get_config
from .models import get_mlx_model
from .tokenizer import MLXTokenizer
from .pooling import pool_and_normalize

class MLXTextEmbedding:
    def __init__(
        self, model_name: str = "mlx-community/all-MiniLM-L6-v2-4bit",
        batch_size: int = 32, cache_dir: str = "~/.cache/fastembed_mlx", compile: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model, self.config = get_mlx_model(model_name, compile, cache_dir)
        self.tokenizer = MLXTokenizer(model_name, self.config.max_position_embeddings)

    def embed(self, texts: List[str], batch_size: Optional[int] = None) -> Iterator[np.ndarray]:
        batch_size = batch_size or self.batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer.encode(batch)
            input_ids = mx.array(encoded["input_ids"])
            attention_mask = mx.array(encoded["attention_mask"])
            last_hidden_state = self.model(input_ids, attention_mask)
            embeddings = pool_and_normalize(last_hidden_state, attention_mask, self.config.pooling, self.config.normalization)
            embeddings_np = np.array(embeddings)
            for emb in embeddings_np:
                yield emb

    def query_embed(self, query: str) -> np.ndarray:
        return next(self.embed([query]))

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        return np.array(list(self.embed(texts, batch_size)))

    @property
    def dim(self) -> int:
        return self.config.dim

    @property
    def max_length(self) -> int:
        return self.config.max_position_embeddings

    def __repr__(self) -> str:
        return f"MLXTextEmbedding(model=\"{self.model_name}\", dim={self.dim}, backend=mlx)"
