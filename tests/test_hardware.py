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
    with patch.dict("sys.modules", {}):
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
