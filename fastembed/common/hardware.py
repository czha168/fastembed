from typing import Sequence

import onnxruntime as ort

from fastembed.common.types import Backend, Device, OnnxProvider

_cached_backend: Backend | None = None
_sentinel: object = object()
_cached_mlx_models: set[str] | None | object = _sentinel


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
