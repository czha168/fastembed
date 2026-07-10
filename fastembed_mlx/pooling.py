"""
Pooling strategies. all-MiniLM-L6-v2 uses MEAN pooling.
"""
import mlx.core as mx

def mean_pooling(last_hidden_state: mx.array, attention_mask: mx.array) -> mx.array:
    mask_expanded = attention_mask[:, :, None].astype(mx.float32)
    sum_embeddings = mx.sum(last_hidden_state * mask_expanded, axis=1)
    token_counts = mx.sum(mask_expanded, axis=1)
    token_counts = mx.maximum(token_counts, 1e-9)
    return sum_embeddings / token_counts

def cls_pooling(last_hidden_state: mx.array) -> mx.array:
    return last_hidden_state[:, 0, :]

def max_pooling(last_hidden_state: mx.array, attention_mask: mx.array) -> mx.array:
    mask_expanded = attention_mask[:, :, None].astype(mx.float32)
    masked_hidden = last_hidden_state * mask_expanded + (1.0 - mask_expanded) * (-1e9)
    return mx.max(masked_hidden, axis=1)

def l2_normalize(embeddings: mx.array, eps: float = 1e-12) -> mx.array:
    norms = mx.sqrt(mx.sum(embeddings ** 2, axis=-1, keepdims=True))
    return embeddings / mx.maximum(norms, eps)

def pool_and_normalize(last_hidden_state: mx.array, attention_mask: mx.array, pooling: str = "mean", normalize: bool = True) -> mx.array:
    if pooling == "mean":
        embeddings = mean_pooling(last_hidden_state, attention_mask)
    elif pooling == "cls" or pooling == "first":  # "first" = CLS token pooling
        embeddings = cls_pooling(last_hidden_state)
    elif pooling == "max":
        embeddings = max_pooling(last_hidden_state, attention_mask)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    if normalize:
        embeddings = l2_normalize(embeddings)
    return embeddings
