# tests/test_mlx_sparse.py
import pytest
import numpy as np
import mlx.core as mx

from fastembed.sparse import SparseTextEmbedding
from fastembed.common.models import SparseEmbedding
from fastembed_mlx.embedder import MlxSparseTextEmbedding  # Case-aligned import
from fastembed_mlx.nn.splade import MlxSpladeModel


def test_sparse_text_embedding_factory_intercept():
    """
    Ensure that requesting 'prithivida/Splade_PP_en_v1' cleanly intercepts
    the ONNX fallback registry and initializes your custom MLX class on Apple Silicon.
    """
    embedder = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    # Assert it initialized our specialized MLX handler class perfectly
    assert isinstance(embedder.model, MlxSparseTextEmbedding)
    assert embedder.model.model_name == "prithivida/Splade_PP_en_v1"


def test_mlx_splade_model_forward_pass():
    """
    Test the neural network layer operations in splade.py.
    Verifies it maps to vocabulary size and correctly runs max-pooling on axis 1.
    """
    embedder = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    model = embedder.model.model
    
    assert isinstance(model, MlxSpladeModel)
    
    # Mock inputs: Batch size 2, Sequence length 4
    batch_size = 2
    seq_len = 4
    vocab_size = embedder.model.config.vocab_size  # 30522
    
    # Explicitly enforce type to match internal embedding array indexes
    mock_input_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    mock_attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
    
    # Run forward step
    output_vectors = model(mock_input_ids, mock_attention_mask)
    
    # Verify the element-wise max pooling flattened the sequence dimension
    assert output_vectors.shape == (batch_size, vocab_size)


def test_sparse_embedding_output_format():
    """
    Verify that the generator yields standard SparseEmbedding objects,
    strips out zeroed weights, and sorts them by value descending.
    """
    embedder = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    documents = ["Manhattan Project atomic bomb"]
    embeddings_generator = embedder.embed(documents, batch_size=1)
    
    result = list(embeddings_generator)
    
    assert len(result) == 1
    sparse_emb = result[0]
    
    # Verify strict class output matching
    assert isinstance(sparse_emb, SparseEmbedding)
    assert isinstance(sparse_emb.indices, list)
    assert isinstance(sparse_emb.values, list)
    
    # Verify sparsity: dimensions must hold positive contextual weights
    assert len(sparse_emb.indices) == len(sparse_emb.values)
    assert all(val > 0.0 for val in sparse_emb.values)
    
    # Verify sorted ordering: values must be in descending order
    assert sparse_emb.values == sorted(sparse_emb.values, reverse=True)


def test_query_embed_parity():
    """
    Ensure the query wrapper behaves consistently and unpacks the single string array.
    """
    embedder = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    query = "nuclear physics"
    
    query_result = list(embedder.query_embed(query))
    
    assert len(query_result) == 1
    assert isinstance(query_result[0], SparseEmbedding)
