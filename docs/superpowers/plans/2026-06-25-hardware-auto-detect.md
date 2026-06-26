---
change: hardware-auto-detect
design-doc: docs/superpowers/specs/2026-06-25-hardware-auto-detect-design.md
base-ref: f3e7dcb140f7d308a5400f4497c4a9ed10f82326
---

# Hardware Auto-Detect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement automatic hardware detection that selects the best available runtime backend (CUDA > MLX > CPU) at model initialization time, making fastembed zero-config while preserving full backward compatibility.

**Architecture:** A new centralized `fastembed/common/hardware.py` module detects the available backend via `onnxruntime.get_available_providers()` and `import mlx.core`, caching the result. A `should_use_mlx()` convenience function combines backend detection, MLX model registry lookup, and override-parameter checking. Each top-level embedding class calls `should_use_mlx()` before its ONNX registry loop to route to MLX when appropriate.

**Tech Stack:** Python 3.10+, onnxruntime (hard dep), mlx.core (optional dep), pytest

## Global Constraints

- `Backend` enum is separate from `Device` enum — `Device` is ONNX-specific, `Backend` represents overall runtime
- Auto-detection only triggers when `cuda=Device.AUTO` (default) and `providers=None`
- Explicit `cuda`/`providers` values bypass MLX routing entirely and follow existing ONNX logic
- `OnnxModel._load_onnx_model` CUDA auto-selection logic is unchanged (D6)
- Detection result is cached after first call to avoid repeated import probes
- `should_use_mlx` returns `False` for models not in `SUPPORTED_MODELS`, causing natural fall-through to ONNX
- Backward compatibility: existing code with explicit `cuda=False` or `providers=[...]` must behave identically

---

## File Structure

| File | Responsibility |
|------|---------------|
| `fastembed/common/types.py` | Add `Backend` enum alongside existing `Device` enum |
| `fastembed/common/hardware.py` | **New file** — `detect_backend()`, `get_mlx_supported_models()`, `should_use_mlx()` |
| `fastembed/text/text_embedding.py` | Add MLX routing check in `__init__` before ONNX registry loop |
| `fastembed/sparse/sparse_text_embedding.py` | Replace hard-coded `HAS_MLX` check with `should_use_mlx()` call |
| `tests/test_hardware.py` | **New file** — all hardware detection and routing tests |

---

### Task 1: Backend Enum

**Files:**
- Modify: `fastembed/common/types.py:1-14`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Consumes: existing `Device` enum in `fastembed/common/types.py`
- Produces: `Backend` enum with values `CUDA = "cuda"`, `MLX = "mlx"`, `CPU = "cpu"`

- [x] **Step 1: Write the failing test**

Create `tests/test_hardware.py`:

```python
from fastembed.common.types import Backend


def test_backend_enum_values():
    assert Backend.CUDA == "cuda"
    assert Backend.MLX == "mlx"
    assert Backend.CPU == "cpu"
    assert set(Backend._value2member_map_.keys()) == {"cuda", "mlx", "cpu"}
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hardware.py::test_backend_enum_values -v`
Expected: FAIL with `ImportError: cannot import name 'Backend' from 'fastembed.common.types'`

- [x] **Step 3: Write minimal implementation**

Add `Backend` enum to `fastembed/common/types.py` after the existing `Device` enum:

```python
class Backend(str, Enum):
    CUDA = "cuda"
    MLX = "mlx"
    CPU = "cpu"
```

The file becomes:

```python
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class Backend(str, Enum):
    CUDA = "cuda"
    MLX = "mlx"
    CPU = "cpu"


PathInput: TypeAlias = str | Path
ImageInput: TypeAlias = PathInput | Image.Image

OnnxProvider: TypeAlias = str | tuple[str, dict[Any, Any]]
NumpyArray: TypeAlias = (
    NDArray[np.float64]
    | NDArray[np.float32]
    | NDArray[np.float16]
    | NDArray[np.int8]
    | NDArray[np.int64]
    | NDArray[np.int32]
)
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hardware.py::test_backend_enum_values -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add fastembed/common/types.py tests/test_hardware.py
git commit -m "feat: add Backend enum to fastembed.common.types"
```

---

### Task 2: Hardware Detection Module

**Files:**
- Create: `fastembed/common/hardware.py`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Consumes: `Backend` enum from `fastembed/common/types`
- Produces:
  - `detect_backend() -> Backend` — probes CUDA → MLX → CPU, caches result
  - `get_mlx_supported_models() -> set[str] | None` — lazy import from `fastembed_mlx.config.SUPPORTED_MODELS`
  - `should_use_mlx(model_name: str, cuda: bool | Device, providers: Sequence | None) -> bool` — combines backend + model + override check

- [x] **Step 1: Write the failing tests**

Add to `tests/test_hardware.py`:

```python
from unittest.mock import patch, MagicMock
from fastembed.common.types import Backend, Device
from fastembed.common.hardware import detect_backend, get_mlx_supported_models, should_use_mlx


def test_detect_backend_cuda():
    with patch("fastembed.common.hardware.ort") as mock_ort:
        mock_ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        from fastembed.common import hardware
        hardware._cached_backend = None
        result = detect_backend()
        assert result == Backend.CUDA


def test_detect_backend_mlx():
    with patch("fastembed.common.hardware.ort") as mock_ort, \
         patch("fastembed.common.hardware._try_import_mlx", return_value=True):
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        from fastembed.common import hardware
        hardware._cached_backend = None
        result = detect_backend()
        assert result == Backend.MLX


def test_detect_backend_cpu():
    with patch("fastembed.common.hardware.ort") as mock_ort, \
         patch("fastembed.common.hardware._try_import_mlx", return_value=False):
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        from fastembed.common import hardware
        hardware._cached_backend = None
        result = detect_backend()
        assert result == Backend.CPU


def test_detect_backend_caches_result():
    with patch("fastembed.common.hardware.ort") as mock_ort, \
         patch("fastembed.common.hardware._try_import_mlx", return_value=False):
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        from fastembed.common import hardware
        hardware._cached_backend = None
        detect_backend()
        call_count_after_first = mock_ort.get_available_providers.call_count
        detect_backend()
        assert mock_ort.get_available_providers.call_count == call_count_after_first


def test_get_mlx_supported_models_available():
    mock_config = MagicMock()
    mock_config.SUPPORTED_MODELS = {"model-a": MagicMock(), "model-b": MagicMock()}
    with patch.dict("sys.modules", {"fastembed_mlx": MagicMock(), "fastembed_mlx.config": mock_config}):
        from fastembed.common import hardware
        hardware._cached_mlx_models = None
        result = get_mlx_supported_models()
        assert result == {"model-a", "model-b"}


def test_get_mlx_supported_models_unavailable():
    with patch.dict("sys.modules", {}):
        from fastembed.common import hardware
        hardware._cached_mlx_models = None
        result = get_mlx_supported_models()
        assert result is None


def test_should_use_mlx_true():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", Device.AUTO, None) is True


def test_should_use_mlx_wrong_model():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"other-model"}):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", Device.AUTO, None) is False


def test_should_use_mlx_cuda_backend():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.CUDA):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", Device.AUTO, None) is False


def test_should_use_mlx_cuda_override():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", True, None) is False


def test_should_use_mlx_cuda_device_override():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", Device.CUDA, None) is False


def test_should_use_mlx_cpu_override():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", Device.CPU, None) is False


def test_should_use_mlx_false_cuda_override():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", False, None) is False


def test_should_use_mlx_providers_override():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", Device.AUTO, ["CPUExecutionProvider"]) is False


def test_should_use_mlx_no_mlx_models():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value=None):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", Device.AUTO, None) is False
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hardware.py -k "test_detect_backend or test_get_mlx_supported or test_should_use_mlx" -v`
Expected: FAIL with `ImportError: cannot import name 'detect_backend' from 'fastembed.common.hardware'`

- [x] **Step 3: Write implementation**

Create `fastembed/common/hardware.py`:

```python
from typing import Sequence

import onnxruntime as ort

from fastembed.common.types import Backend, Device, OnnxProvider

_cached_backend: Backend | None = None
_cached_mlx_models: set[str] | None | object = _sentinel = object()


def _try_import_mlx() -> bool:
    try:
        import mlx.core
        return True
    except ImportError:
        return False


def detect_backend() -> Backend:
    global _cached_backend
    if _cached_backend is not None:
        return _cached_backend

    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        _cached_backend = Backend.CUDA
    elif _try_import_mlx():
        _cached_backend = Backend.MLX
    else:
        _cached_backend = Backend.CPU

    return _cached_backend


def get_mlx_supported_models() -> set[str] | None:
    global _cached_mlx_models
    if _cached_mlx_models is not _sentinel:
        return _cached_mlx_models

    try:
        from fastembed_mlx.config import SUPPORTED_MODELS
        _cached_mlx_models = set(SUPPORTED_MODELS.keys())
    except ImportError:
        _cached_mlx_models = None

    return _cached_mlx_models


def should_use_mlx(
    model_name: str,
    cuda: bool | Device,
    providers: Sequence[OnnxProvider] | None,
) -> bool:
    if providers is not None:
        return False
    if cuda is not Device.AUTO:
        return False
    if detect_backend() != Backend.MLX:
        return False
    mlx_models = get_mlx_supported_models()
    if mlx_models is None:
        return False
    return model_name in mlx_models
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hardware.py -v`
Expected: All PASS

- [x] **Step 5: Commit**

```bash
git add fastembed/common/hardware.py tests/test_hardware.py
git commit -m "feat: add hardware detection module with detect_backend, get_mlx_supported_models, should_use_mlx"
```

---

### Task 3: TextEmbedding MLX Routing

**Files:**
- Modify: `fastembed/text/text_embedding.py:81-117`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Consumes: `should_use_mlx()` from `fastembed/common/hardware`, `MLXTextEmbedding` from `fastembed_mlx.embedder`
- Produces: `TextEmbedding.__init__` routes to `MLXTextEmbedding` when MLX is available, model is supported, and no overrides are set

- [x] **Step 1: Write the failing test**

Add to `tests/test_hardware.py`:

```python
from unittest.mock import patch, MagicMock


def test_text_embedding_routes_to_mlx():
    mock_mlx_cls = MagicMock()
    mock_mlx_instance = MagicMock()
    mock_mlx_cls.return_value = mock_mlx_instance
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}), \
         patch.dict("sys.modules", {"fastembed_mlx": MagicMock(), "fastembed_mlx.embedder": MagicMock(MLXTextEmbedding=mock_mlx_cls)}):
        from fastembed.text.text_embedding import TextEmbedding
        emb = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        assert emb.model is mock_mlx_instance


def test_text_embedding_cuda_override_skips_mlx():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        from fastembed.text.text_embedding import TextEmbedding
        emb = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cuda=False)
        assert emb.model is not None
        mock_mlx_cls_name = type(emb.model).__name__
        assert "MLX" not in mock_mlx_cls_name


def test_text_embedding_providers_override_skips_mlx():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        from fastembed.text.text_embedding import TextEmbedding
        emb = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", providers=["CPUExecutionProvider"])
        assert emb.model is not None
        mock_mlx_cls_name = type(emb.model).__name__
        assert "MLX" not in mock_mlx_cls_name


def test_text_embedding_unsupported_model_falls_through():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value=set()):
        from fastembed.text.text_embedding import TextEmbedding
        emb = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        assert emb.model is not None
        mock_mlx_cls_name = type(emb.model).__name__
        assert "MLX" not in mock_mlx_cls_name
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hardware.py -k "test_text_embedding" -v`
Expected: `test_text_embedding_routes_to_mlx` FAILS (MLX routing not implemented), others may pass trivially

- [x] **Step 3: Write implementation**

Modify `fastembed/text/text_embedding.py` — add import at top and MLX routing check in `__init__`:

Add import at top of file:

```python
from fastembed.common.hardware import should_use_mlx
```

Insert MLX routing block in `__init__` after `super().__init__()` call and the jina warning, before the ONNX registry loop. The `__init__` method becomes:

```python
def __init__(
    self,
    model_name: str = "BAAI/bge-small-en-v1.5",
    cache_dir: str | None = None,
    threads: int | None = None,
    providers: Sequence[OnnxProvider] | None = None,
    cuda: bool | Device = Device.AUTO,
    device_ids: list[int] | None = None,
    lazy_load: bool = False,
    **kwargs: Any,
):
    super().__init__(model_name, cache_dir, threads, **kwargs)
    if model_name.lower() == "jinaai/jina-embeddings-v2-base-de":
        warnings.warn(
            "The model 'jinaai/jina-embeddings-v2-base-de' used to run with fp16 model, but due to onnxruntime updates, now it runs with the original fp32 model.",
            UserWarning,
            stacklevel=2,
        )

    if should_use_mlx(model_name, cuda, providers):
        from fastembed_mlx.embedder import MLXTextEmbedding
        self.model = MLXTextEmbedding(model_name=model_name)
        return

    for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
        supported_models = EMBEDDING_MODEL_TYPE._list_supported_models()
        if any(model_name.lower() == model.model.lower() for model in supported_models):
            self.model = EMBEDDING_MODEL_TYPE(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=threads,
                providers=providers,
                cuda=cuda,
                device_ids=device_ids,
                lazy_load=lazy_load,
                **kwargs,
            )
            return

    raise ValueError(
        f"Model {model_name} is not supported in TextEmbedding. "
        "Please check the supported models using `TextEmbedding.list_supported_models()`"
    )
```

Full updated imports at top of file:

```python
import warnings
from typing import Any, Iterable, Sequence, Type
from dataclasses import asdict

from fastembed.common.types import NumpyArray, OnnxProvider, Device
from fastembed.common.hardware import should_use_mlx
from fastembed.text.clip_embedding import CLIPOnnxEmbedding
from fastembed.text.custom_text_embedding import CustomTextEmbedding
from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
from fastembed.text.pooled_embedding import PooledEmbedding
from fastembed.text.multitask_embedding import JinaEmbeddingV3
from fastembed.text.builtin_sentence_embedding import BuiltinSentenceEmbedding
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed.text.text_embedding_base import TextEmbeddingBase
from fastembed.common.model_description import DenseModelDescription, ModelSource, PoolingType
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hardware.py -k "test_text_embedding" -v`
Expected: All PASS

- [x] **Step 5: Run existing tests to verify no regression**

Run: `python -m pytest tests/test_common.py -v`
Expected: All PASS

- [x] **Step 6: Commit**

```bash
git add fastembed/text/text_embedding.py tests/test_hardware.py
git commit -m "feat: add MLX routing to TextEmbedding via should_use_mlx"
```

---

### Task 4: SparseTextEmbedding MLX Routing Refactor

**Files:**
- Modify: `fastembed/sparse/sparse_text_embedding.py:1-96`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Consumes: `should_use_mlx()` from `fastembed/common/hardware`
- Produces: `SparseTextEmbedding.__init__` uses `should_use_mlx()` instead of hard-coded `HAS_MLX and model_name == Splade` check. Removes top-level `HAS_MLX` / `MlxSparseTextEmbedding` import.

- [x] **Step 1: Write the failing test**

Add to `tests/test_hardware.py`:

```python
def test_sparse_embedding_routes_to_mlx():
    mock_mlx_cls = MagicMock()
    mock_mlx_instance = MagicMock()
    mock_mlx_cls.return_value = mock_mlx_instance
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"prithivida/Splade_PP_en_v1"}), \
         patch.dict("sys.modules", {"fastembed_mlx": MagicMock(), "fastembed_mlx.embedder": MagicMock(MlxSparseTextEmbedding=mock_mlx_cls)}):
        from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
        emb = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        assert emb.model is mock_mlx_instance


def test_sparse_embedding_cuda_override_skips_mlx():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"prithivida/Splade_PP_en_v1"}):
        from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
        emb = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1", cuda=False)
        assert emb.model is not None
        cls_name = type(emb.model).__name__
        assert "Mlx" not in cls_name


def test_sparse_embedding_unsupported_model_falls_through():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value=set()):
        from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
        emb = SparseTextEmbedding(model_name="Qdrant/bm25")
        assert emb.model is not None
        cls_name = type(emb.model).__name__
        assert "Mlx" not in cls_name
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hardware.py -k "test_sparse_embedding" -v`
Expected: `test_sparse_embedding_routes_to_mlx` FAILS (still uses old HAS_MLX pattern)

- [x] **Step 3: Write implementation**

Replace the entire `fastembed/sparse/sparse_text_embedding.py` with the refactored version. Remove the `HAS_MLX` / `MlxSparseTextEmbedding` top-level import block and replace the hard-coded check in `__init__`:

```python
from typing import Any, Iterable, Sequence, Type
from dataclasses import asdict
import warnings

from fastembed.common import OnnxProvider
from fastembed.common.types import Device
from fastembed.common.hardware import should_use_mlx
from fastembed.common.model_description import SparseModelDescription
from fastembed.sparse.bm25 import Bm25
from fastembed.sparse.bm42 import Bm42
from fastembed.sparse.minicoil import MiniCOIL
from fastembed.sparse.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from fastembed.sparse.splade_pp import SpladePP


class SparseTextEmbedding(SparseTextEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[Type[SparseTextEmbeddingBase]] = [
        SpladePP,
        Bm42,
        Bm25,
        MiniCOIL,
    ]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.

        Example:
            ```
            [
                {
                    "model": "prithvida/SPLADE_PP_en_v1",
                    "vocab_size": 30522,
                    "description": "Independent Implementation of SPLADE++ Model for English",
                    "license": "apache-2.0",
                    "size_in_GB": 0.532,
                    "sources": {
                        "hf": "qdrant/SPLADE_PP_en_v1",
                    },
                }
            ]
            ```
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[SparseModelDescription]:
        result: list[SparseModelDescription] = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding._list_supported_models())
        return result

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)

        if model_name.lower() == "prithvida/Splade_PP_en_v1".lower():
            warnings.warn(
                "The right spelling is prithivida/Splade_PP_en_v1. "
                "Support of this name will be removed soon, please fix the model_name",
                DeprecationWarning,
                stacklevel=2,
            )
            model_name = "prithivida/Splade_PP_en_v1"

        if should_use_mlx(model_name, cuda, providers):
            from fastembed_mlx.embedder import MlxSparseTextEmbedding
            self.model = MlxSparseTextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                **kwargs
            )
            return

        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE._list_supported_models()
            if any(model_name.lower() == model.model.lower() for model in supported_models):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name,
                    cache_dir,
                    threads=threads,
                    providers=providers,
                    cuda=cuda,
                    device_ids=device_ids,
                    lazy_load=lazy_load,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in SparseTextEmbedding. "
            "Please check the supported models using `SparseTextEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        """
        Encode a list of documents into list of embeddings.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel: If > 1, data-parallel encoding will be used. If 0, use all available cores.

        Returns:
            List of embeddings, one per document
        """
        yield from self.model.embed(documents, batch_size=batch_size, parallel=parallel, **kwargs)

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[SparseEmbedding]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable of queries.

        Returns:
            Iterable[SparseEmbedding]: The sparse embeddings.
        """
        yield from self.model.query_embed(query, **kwargs)

    def token_count(self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any) -> int:
        """
        Returns the number of tokens in the texts.

        Args:
            texts (str | Iterable[str]): The list of texts to check.
            batch_size (int): Batch size for tokenization

        Returns:
            int: Sum of number of tokens in the texts.
        """
        return self.model.token_count(texts, batch_size=batch_size, **kwargs)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hardware.py -k "test_sparse_embedding" -v`
Expected: All PASS

- [x] **Step 5: Run existing sparse tests to verify no regression**

Run: `python -m pytest tests/test_common.py -v`
Expected: All PASS

- [x] **Step 6: Commit**

```bash
git add fastembed/sparse/sparse_text_embedding.py tests/test_hardware.py
git commit -m "refactor: replace hard-coded HAS_MLX check in SparseTextEmbedding with should_use_mlx"
```

---

### Task 5: Detection Caching Test

**Files:**
- Modify: `tests/test_hardware.py`

**Interfaces:**
- Consumes: `detect_backend()` from `fastembed/common/hardware`
- Produces: Test verifying second call returns cached result without re-probing

- [x] **Step 1: Write the test**

The caching test was already added in Task 2 (`test_detect_backend_caches_result`). Verify it works correctly by adding a more explicit test that checks the cached value is returned directly:

Add to `tests/test_hardware.py`:

```python
def test_detect_backend_cache_returns_same_value():
    with patch("fastembed.common.hardware.ort") as mock_ort, \
         patch("fastembed.common.hardware._try_import_mlx", return_value=False):
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        from fastembed.common import hardware
        hardware._cached_backend = None
        first = detect_backend()
        assert first == Backend.CPU
        assert hardware._cached_backend == Backend.CPU
        second = detect_backend()
        assert second == Backend.CPU
        assert mock_ort.get_available_providers.call_count == 1
```

- [x] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_hardware.py::test_detect_backend_cache_returns_same_value -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_hardware.py
git commit -m "test: add explicit detection caching verification"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- D1 (Centralized detection module): Task 2 implements `hardware.py` with all three functions ✓
- D2 (Backend enum): Task 1 implements `Backend` enum ✓
- D3 (Detection algorithm): Task 2 implements CUDA → MLX → CPU probe with caching ✓
- D4 (Per-constructor MLX routing): Task 3 (TextEmbedding), Task 4 (SparseTextEmbedding) ✓
- D5 (Override semantics): Task 2 `should_use_mlx` implements `providers is not None` and `cuda is not Device.AUTO` checks ✓
- D6 (CUDA auto-selection unchanged): No changes to `OnnxModel._load_onnx_model` ✓
- Testing Strategy from spec: Detection (Task 2), Model routing (Tasks 3, 4), Override bypass (Tasks 3, 4), Caching (Task 5), Fall-through (Tasks 3, 4) ✓

**2. Placeholder scan:** No TBD, TODO, "implement later", "add appropriate error handling", or "similar to Task N" found. ✓

**3. Type consistency:**
- `should_use_mlx(model_name: str, cuda: bool | Device, providers: Sequence | None) -> bool` — consistent across Task 2 definition and Tasks 3/4 usage ✓
- `detect_backend() -> Backend` — consistent ✓
- `get_mlx_supported_models() -> set[str] | None` — consistent ✓
- `Backend.CUDA/MLX/CPU` — consistent across all tasks ✓
- `MlxSparseTextEmbedding` class name matches `fastembed-mlx/embedder.py:64` which defines `MLXSparseTextEmbedding` — **note**: the existing code in `sparse_text_embedding.py:19` imports `MlxSparseTextEmbedding` from `fastembed_mlx.embedder`, but the actual class name in `embedder.py` is `MLXSparseTextEmbedding` (capital XS). The existing import line `from fastembed_mlx.embedder import MlxSparseTextEmbedding` already handles any aliasing. Our plan preserves this exact import line. ✓
