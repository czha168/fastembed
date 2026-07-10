"""
HuggingFace tokenizer wrapper with MLX-friendly output.
"""
from typing import List, Dict, Union
import numpy as np

class MLXTokenizer:
    def __init__(self, model_name: str, max_length: int = 512):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or ""

    def encode(self, texts: Union[str, List[str]], return_tensors: str = "np", **kwargs) -> Dict[str, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors=return_tensors, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
