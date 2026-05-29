""" MLXTextEmbedding: FastEmbed-compatible API for Apple Silicon. """
from typing import Iterator, List, Optional
import numpy as np
import mlx.core as mx

from .config import ModelConfig, get_config
from .models import get_mlx_model
from .tokenizer import MLXTokenizer
from .pooling import pool_and_normalize

# Imports required for the Sparse/SPLADE extension
from fastembed.sparse.sparse_embedding_base import SparseTextEmbeddingBase
from fastembed.common.models import SparseEmbedding


class MLXTextEmbedding:
    def __init__(
        self,
        model_name: str = "mlx-community/all-MiniLM-L6-v2-4bit",
        batch_size: int = 32,
        cache_dir: str = "~/.cache/fastembed_mlx",
        compile: bool = True,
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
            embeddings = pool_and_normalize(
                last_hidden_state, attention_mask, self.config.pooling, self.config.normalization
            )
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


# --- Added MLX Sparse Expansion Engine ---

class MLXSparseTextEmbedding(SparseTextEmbeddingBase):
    def __init__(
        self, 
        model_name: str = "prithivida/Splade_PP_en_v1", 
        batch_size: int = 32, 
        cache_dir: str = "~/.cache/fastembed_mlx",
        compile: bool = True,
        **kwargs
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Matches your exact model initialization and tokenizer structures
        self.model, self.config = get_mlx_model(model_name, compile, cache_dir)
        self.tokenizer = MLXTokenizer(model_name, self.config.max_position_embeddings)

    def embed(
        self, 
        documents: str | Iterable[str], 
        batch_size: int = 256, 
        parallel: Optional[int] = None, 
        **kwargs
    ) -> Iterator[SparseEmbedding]:
        if isinstance(documents, str):
            documents = [documents]
            
        # Fallback to local class configuration if defaults are requested
        b_size = batch_size or self.batch_size
        
        # Convert an iterable smoothly to an accessible sequence loop
        docs_list = list(documents) if not isinstance(documents, list) else documents

        for i in range(0, len(docs_list), b_size):
            batch = docs_list[i:i + b_size]
            
            # Use your tokenizer mapping pattern
            encoded = self.tokenizer.encode(batch)
            input_ids = mx.array(encoded["input_ids"])
            attention_mask = mx.array(encoded["attention_mask"])
            
            # Forward passage handles token type ids if exposed by BERT configurations
            token_type_ids = mx.array(encoded["token_type_ids"]) if "token_type_ids" in encoded else None

            # 1. Complete Forward pipeline through your custom MlxSpladeModel pass
            sparse_vectors = self.model(input_ids, attention_mask, token_type_ids)
            
            # 2. Force graph compilation and evaluate tensors in memory
            mx.eval(sparse_vectors)
            
            # 3. Pull output to NumPy to parse non-zero active weights
            np_vectors = np.array(sparse_vectors)
            
            for vec in np_vectors:
                # Isolate dimensions expanded by the Log-ReLU activation layer (> 0.0)
                nonzero_indices = np.nonzero(vec)[0]
                values = vec[nonzero_indices]
                
                # Sort dimensions in descending weight order for downstream retrieval matching
                sort_idx = np.argsort(values)[::-1]
                
                yield SparseEmbedding(
                    indices=nonzero_indices[sort_idx].tolist(),
                    values=values[sort_idx].tolist()
                )

    def query_embed(self, query: str | Iterable[str], **kwargs) -> Iterator[SparseEmbedding]:
        if isinstance(query, str):
            query = [query]
        yield from self.embed(query, batch_size=len(query))

    @property
    def max_length(self) -> int:
        return self.config.max_position_embeddings

    def __repr__(self) -> str:
        return f"MLXSparseTextEmbedding(model=\"{self.model_name}\", backend=mlx)"
