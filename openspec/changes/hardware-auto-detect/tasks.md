## 1. Backend Enum and Detection Utility

- [x] 1.1 Add `Backend` enum (`CUDA`, `MLX`, `CPU`) to `fastembed/common/types.py`
- [x] 1.2 Create `fastembed/common/hardware.py` with `detect_backend()` function implementing CUDA → MLX → CPU probe with result caching
- [x] 1.3 Add `get_mlx_supported_models()` helper in `hardware.py` that imports `fastembed_mlx.config.SUPPORTED_MODELS` when MLX is available and returns the model name set

## 2. TextEmbedding Integration

- [x] 2.1 Modify `TextEmbedding.__init__()` to call `detect_backend()` and route to `MLXTextEmbedding` when backend is MLX and model is in MLX supported models, before entering the ONNX registry loop
- [x] 2.2 Ensure explicit `cuda`/`providers` params bypass MLX routing (only `Device.AUTO` with no `providers` triggers auto-detection)

## 3. SparseTextEmbedding Integration

- [x] 3.1 Refactor `SparseTextEmbedding.__init__()` to replace the hard-coded `HAS_MLX and model_name == Splade` check with `detect_backend()` + MLX model registry lookup
- [x] 3.2 Ensure explicit `cuda`/`providers` params bypass MLX routing in sparse path

## 4. Tests

- [x] 4.1 Add unit tests for `detect_backend()` with mocked onnxruntime providers and mlx import (CUDA-available, MLX-available, CPU-only scenarios)
- [x] 4.2 Add unit tests for `TextEmbedding` MLX routing (auto-detect routes to MLX, explicit cuda=False bypasses, unsupported model falls through)
- [x] 4.3 Add unit tests for `SparseTextEmbedding` MLX routing (auto-detect routes to MLX, explicit override bypasses, unsupported model falls through)
- [x] 4.4 Add test for detection caching (second call returns cached result without re-probing)
