"""
Utility functions.
"""
import time
from typing import Callable
from functools import wraps

def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        return result, time.perf_counter() - start
    return wrapper

def cosine_similarity(a, b):
    import mlx.core as mx
    return mx.sum(a * b) / (mx.sqrt(mx.sum(a ** 2)) * mx.sqrt(mx.sum(b ** 2)))
