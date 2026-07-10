from unittest.mock import patch, MagicMock

from fastembed.common.types import Backend, Device
from fastembed.common.hardware import (
    detect_backend,
    get_mlx_supported_models,
    should_use_mlx,
)


def test_backend_enum_values():
    assert Backend.CUDA.value == "cuda"
    assert Backend.MLX.value == "mlx"
    assert Backend.CPU.value == "cpu"
    assert set(Backend._value2member_map_.keys()) == {"cuda", "mlx", "cpu"}


def test_detect_backend_cuda():
    from fastembed.common import hardware
    hardware._cached_backend = None
    with patch.object(hardware.ort, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
        result = detect_backend()
    assert result == Backend.CUDA
    hardware._cached_backend = None


def test_detect_backend_mlx():
    from fastembed.common import hardware
    hardware._cached_backend = None
    with patch.object(hardware.ort, "get_available_providers", return_value=["CPUExecutionProvider"]), \
         patch.object(hardware, "_try_import_mlx", return_value=True):
        result = detect_backend()
    assert result == Backend.MLX
    hardware._cached_backend = None


def test_detect_backend_cpu():
    from fastembed.common import hardware
    hardware._cached_backend = None
    with patch.object(hardware.ort, "get_available_providers", return_value=["CPUExecutionProvider"]), \
         patch.object(hardware, "_try_import_mlx", return_value=False):
        result = detect_backend()
    assert result == Backend.CPU
    hardware._cached_backend = None


def test_detect_backend_caches_result():
    from fastembed.common import hardware
    hardware._cached_backend = None
    with patch.object(hardware.ort, "get_available_providers", return_value=["CPUExecutionProvider"]) as mock_providers, \
         patch.object(hardware, "_try_import_mlx", return_value=False):
        detect_backend()
        call_count_after_first = mock_providers.call_count
        detect_backend()
        assert mock_providers.call_count == call_count_after_first
    hardware._cached_backend = None


def test_detect_backend_cache_returns_same_value():
    from fastembed.common import hardware
    hardware._cached_backend = None
    with patch.object(hardware.ort, "get_available_providers", return_value=["CPUExecutionProvider"]), \
         patch.object(hardware, "_try_import_mlx", return_value=False):
        first = detect_backend()
        assert first == Backend.CPU
        assert hardware._cached_backend == Backend.CPU
        second = detect_backend()
        assert second == Backend.CPU
    hardware._cached_backend = None


def test_get_mlx_supported_models_available():
    from fastembed.common import hardware
    hardware._cached_mlx_models = hardware._sentinel
    mock_config = MagicMock()
    mock_config.SUPPORTED_MODELS = {"model-a": MagicMock(), "model-b": MagicMock()}
    with patch.dict("sys.modules", {"fastembed_mlx": MagicMock(), "fastembed_mlx.config": mock_config}):
        result = get_mlx_supported_models()
    assert result == {"model-a", "model-b"}
    hardware._cached_mlx_models = hardware._sentinel


def test_get_mlx_supported_models_unavailable():
    from fastembed.common import hardware
    hardware._cached_mlx_models = hardware._sentinel
    # Set modules to None to make Python treat them as unimportable, since
    # fastembed_mlx is now a real installed package and patch.dict({}) alone
    # would allow it to be re-imported.
    with patch.dict("sys.modules", {"fastembed_mlx": None, "fastembed_mlx.config": None}):
        result = get_mlx_supported_models()
    assert result is None
    hardware._cached_mlx_models = hardware._sentinel


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


def test_should_use_mlx_cuda_true_override():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", True, None) is False


def test_should_use_mlx_cuda_device_override():
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX), \
         patch("fastembed.common.hardware.get_mlx_supported_models", return_value={"BAAI/bge-small-en-v1.5"}):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", Device.CUDA, None) is False


def test_should_use_mlx_cpu_device_override():
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


def test_bge_small_bf16_registered_and_routable():
    """The full-precision MLX BGE-small model must be registered so it routes to MLX."""
    from fastembed.common import hardware
    hardware._cached_mlx_models = hardware._sentinel
    mlx_models = get_mlx_supported_models()
    assert "mlx-community/bge-small-en-v1.5-bf16" in mlx_models
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX):
        assert should_use_mlx("mlx-community/bge-small-en-v1.5-bf16", Device.AUTO, None) is True
    hardware._cached_mlx_models = hardware._sentinel


def test_all_registered_pooling_strategies_are_supported():
    """Every pooling value declared in SUPPORTED_MODELS must be handled by pool_and_normalize."""
    from fastembed_mlx.config import SUPPORTED_MODELS
    from fastembed_mlx.pooling import pool_and_normalize

    supported = {"mean", "cls", "first", "max", "splade"}
    declared = {cfg.pooling for cfg in SUPPORTED_MODELS.values()}
    # "splade" is handled by the SPLADE model class itself, not pool_and_normalize,
    # so it is exempt from this check.
    unhandled = (declared - supported) - {"splade"}
    assert not unhandled, f"Pooling strategies declared but never handled: {unhandled}"


def test_onnx_default_model_routes_to_mlx_via_alias():
    """The ONNX default 'BAAI/bge-small-en-v1.5' must route to MLX via alias."""
    from fastembed.common import hardware
    hardware._cached_mlx_models = hardware._sentinel
    assert "BAAI/bge-small-en-v1.5" in get_mlx_supported_models()
    with patch("fastembed.common.hardware.detect_backend", return_value=Backend.MLX):
        assert should_use_mlx("BAAI/bge-small-en-v1.5", Device.AUTO, None) is True
    hardware._cached_mlx_models = hardware._sentinel


def test_resolve_mlx_model_name():
    """Aliased names resolve to their native MLX equivalents; others pass through."""
    from fastembed_mlx.config import resolve_mlx_model_name
    assert resolve_mlx_model_name("BAAI/bge-small-en-v1.5") == "mlx-community/bge-small-en-v1.5-bf16"
    # Non-aliased names pass through unchanged
    assert resolve_mlx_model_name("mlx-community/all-MiniLM-L6-v2-4bit") == "mlx-community/all-MiniLM-L6-v2-4bit"
    assert resolve_mlx_model_name("unknown/model") == "unknown/model"


def test_resolve_mlx_model_name_reexported_from_hardware():
    """hardware.resolve_mlx_model_name mirrors fastembed_mlx.config.resolve_mlx_model_name."""
    from fastembed.common.hardware import resolve_mlx_model_name
    assert resolve_mlx_model_name("BAAI/bge-small-en-v1.5") == "mlx-community/bge-small-en-v1.5-bf16"


def test_mlx_text_embedding_default_matches_onnx_default_resolution():
    """Standalone MLXTextEmbedding default resolves to the same model as the ONNX default alias."""
    import inspect
    from fastembed_mlx.embedder import MLXTextEmbedding
    from fastembed_mlx.config import resolve_mlx_model_name

    mlx_default = inspect.signature(MLXTextEmbedding.__init__).parameters["model_name"].default
    onnx_default_resolved = resolve_mlx_model_name("BAAI/bge-small-en-v1.5")
    assert mlx_default == onnx_default_resolved == "mlx-community/bge-small-en-v1.5-bf16"
