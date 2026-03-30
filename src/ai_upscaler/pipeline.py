from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import Backend, OutputFormat, OutputMode, UpscaleJob

ProgressCallback = Callable[[str], None]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FORMAT_SUFFIXES = {
    OutputFormat.JPG: ".jpg",
    OutputFormat.PNG: ".png",
    OutputFormat.TIFF: ".tiff",
}


class UpscaleError(RuntimeError):
    pass


@dataclass(slots=True)
class MediaInfo:
    width: int
    height: int


@dataclass(slots=True)
class ModelSpec:
    input_name: str
    channels: int | None
    input_height: int | None
    input_width: int | None
    output_height: int | None
    output_width: int | None

    @property
    def has_fixed_input_size(self) -> bool:
        return self.input_height is not None and self.input_width is not None

    @property
    def scale_factor(self) -> int | None:
        if (
            self.input_height
            and self.input_width
            and self.output_height
            and self.output_width
            and self.output_height % self.input_height == 0
            and self.output_width % self.input_width == 0
        ):
            scale_y = self.output_height // self.input_height
            scale_x = self.output_width // self.input_width
            if scale_x == scale_y:
                return scale_x
        return None


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def resolve_output_path(target: Path, output_format: OutputFormat) -> Path:
    if output_format == OutputFormat.KEEP:
        return target
    return target.with_suffix(FORMAT_SUFFIXES[output_format])


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise UpscaleError(f"Could not open image: {path}")
    return image


def get_media_info(path: Path) -> MediaInfo:
    image = load_image(path)
    height, width = image.shape[:2]
    return MediaInfo(width=width, height=height)


def save_image(job: UpscaleJob, image: np.ndarray) -> Path:
    output_path = resolve_output_path(job.output_path, job.output_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    params: list[int] = []

    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, job.jpeg_quality))]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, job.png_compression))]
    elif suffix in {".tif", ".tiff"}:
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]

    if not cv2.imwrite(str(output_path), image, params):
        raise UpscaleError(f"Could not save image: {output_path}")
    return output_path


def _bgr_to_model_tensor(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(image_rgb, (2, 0, 1))[None, ...]


def _tensor_to_bgr_image(output: np.ndarray) -> np.ndarray:
    array = np.squeeze(output, axis=0)
    array = np.transpose(array, (1, 2, 0))
    array = np.clip(array, 0.0, 1.0)
    rgb = (array * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _normalize_dim(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _axis_positions(length: int, tile: int, overlap: int) -> list[int]:
    stride = max(1, tile - (2 * overlap))
    positions = [0]
    current = 0
    while current + tile < length:
        current += stride
        if current + tile >= length:
            current = max(0, length - tile)
        if current == positions[-1]:
            break
        positions.append(current)
    return positions


def _default_tile_overlap(tile_h: int, tile_w: int) -> int:
    overlap = min(tile_h, tile_w) // 8
    overlap = max(8, min(overlap, 24))
    if tile_h <= overlap * 2 or tile_w <= overlap * 2:
        overlap = max(0, min(tile_h, tile_w) // 10)
    return overlap


class ModelRunner:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def _run_tensor(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def upscale(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.spec.channels not in {None, 3}:
            raise UpscaleError(
                "The selected model is not a 3-channel RGB upscaler. "
                "Choose an RGB image super-resolution ONNX model such as Real-ESRGAN x4."
            )

        if self.spec.has_fixed_input_size:
            return self._upscale_tiled(image_bgr)

        return _tensor_to_bgr_image(self._run_tensor(_bgr_to_model_tensor(image_bgr)))

    def _upscale_tiled(self, image_bgr: np.ndarray) -> np.ndarray:
        tile_h = self.spec.input_height
        tile_w = self.spec.input_width
        scale = self.spec.scale_factor
        if tile_h is None or tile_w is None or scale is None:
            raise UpscaleError("The selected fixed-size model has unsupported shape metadata.")

        overlap = _default_tile_overlap(tile_h, tile_w)
        height, width = image_bgr.shape[:2]
        out = np.empty((height * scale, width * scale, 3), dtype=np.uint8)

        y_positions = _axis_positions(height, tile_h, overlap)
        x_positions = _axis_positions(width, tile_w, overlap)

        for y in y_positions:
            for x in x_positions:
                tile = image_bgr[y:min(y + tile_h, height), x:min(x + tile_w, width)]
                actual_h, actual_w = tile.shape[:2]
                pad_bottom = tile_h - actual_h
                pad_right = tile_w - actual_w
                if pad_bottom or pad_right:
                    tile = cv2.copyMakeBorder(tile, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)

                output = _tensor_to_bgr_image(self._run_tensor(_bgr_to_model_tensor(tile)))
                crop_top = 0 if y == 0 else overlap * scale
                crop_left = 0 if x == 0 else overlap * scale
                crop_bottom = 0 if y + actual_h >= height else overlap * scale
                crop_right = 0 if x + actual_w >= width else overlap * scale
                valid_h = actual_h * scale
                valid_w = actual_w * scale
                output = output[crop_top:valid_h - crop_bottom, crop_left:valid_w - crop_right]
                out_y = (y * scale) + crop_top
                out_x = (x * scale) + crop_left
                out[out_y:out_y + output.shape[0], out_x:out_x + output.shape[1]] = output

        return out


class OnnxRuntimeRunner(ModelRunner):
    def __init__(self, model_path: Path, backend: Backend, device_id: int | None = None) -> None:
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True
        session_options.intra_op_num_threads = 0
        session_options.inter_op_num_threads = 0
        session_options.log_severity_level = 3

        cuda_provider = (
            "CUDAExecutionProvider",
            {
                "device_id": str(device_id if device_id is not None else 0),
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_use_max_workspace": "1",
                "do_copy_in_default_stream": "1",
            },
        )
        provider_map: dict[Backend, list] = {
            Backend.CPU: ["CPUExecutionProvider"],
            Backend.CUDA: [cuda_provider, "CPUExecutionProvider"],
            Backend.TENSORRT: [
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": str(device_id if device_id is not None else 0),
                        "trt_fp16_enable": "1",
                        "trt_engine_cache_enable": "1",
                        "trt_timing_cache_enable": "1",
                    },
                ),
                cuda_provider,
                "CPUExecutionProvider",
            ],
        }
        providers = provider_map.get(backend)
        if providers is None:
            raise UpscaleError(f"Unsupported ONNX Runtime backend: {backend.value}")

        try:
            session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
        except Exception as exc:
            if backend == Backend.TENSORRT:
                try:
                    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=[cuda_provider, "CPUExecutionProvider"])
                except Exception as cuda_exc:
                    raise UpscaleError(f"TensorRT initialization failed and CUDA fallback also failed: {cuda_exc}") from cuda_exc
            else:
                raise UpscaleError(f"Failed to initialize {backend.value} session: {exc}") from exc

        actual_providers = session.get_providers()
        expected_provider = {
            Backend.CUDA: "CUDAExecutionProvider",
            Backend.TENSORRT: "TensorrtExecutionProvider",
            Backend.CPU: "CPUExecutionProvider",
        }.get(backend)
        if expected_provider and expected_provider not in actual_providers:
            raise UpscaleError(
                f"{backend.value} was requested but is not active. Active providers: {', '.join(actual_providers)}. "
                "Check the NVIDIA runtime setup before starting a large batch."
            )

        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        spec = ModelSpec(
            input_name=input_meta.name,
            channels=_normalize_dim(input_meta.shape[1]) if len(input_meta.shape) >= 4 else None,
            input_height=_normalize_dim(input_meta.shape[2]) if len(input_meta.shape) >= 4 else None,
            input_width=_normalize_dim(input_meta.shape[3]) if len(input_meta.shape) >= 4 else None,
            output_height=_normalize_dim(output_meta.shape[2]) if len(output_meta.shape) >= 4 else None,
            output_width=_normalize_dim(output_meta.shape[3]) if len(output_meta.shape) >= 4 else None,
        )
        super().__init__(spec)
        self.session = session
        self.input_name = input_meta.name

    def _run_tensor(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]


class OpenCVDnnRunner(ModelRunner):
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        cv2.ocl.setUseOpenCL(True)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        spec = ModelSpec(
            input_name=input_meta.name,
            channels=_normalize_dim(input_meta.shape[1]) if len(input_meta.shape) >= 4 else None,
            input_height=_normalize_dim(input_meta.shape[2]) if len(input_meta.shape) >= 4 else None,
            input_width=_normalize_dim(input_meta.shape[3]) if len(input_meta.shape) >= 4 else None,
            output_height=_normalize_dim(output_meta.shape[2]) if len(output_meta.shape) >= 4 else None,
            output_width=_normalize_dim(output_meta.shape[3]) if len(output_meta.shape) >= 4 else None,
        )
        super().__init__(spec)
        net = cv2.dnn.readNetFromONNX(str(model_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        self.net = net

    def _run_tensor(self, tensor: np.ndarray) -> np.ndarray:
        self.net.setInput(tensor)
        return self.net.forward()


def create_runner(job: UpscaleJob) -> ModelRunner:
    if not job.model_path.exists():
        raise UpscaleError(f"Model file not found: {job.model_path}")
    if job.backend == Backend.OPENCL:
        return OpenCVDnnRunner(job.model_path)
    return OnnxRuntimeRunner(job.model_path, job.backend, device_id=job.device_id)


def _fit_to_resolution(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = min(target_width / width, target_height / height)
    return cv2.resize(image, (max(1, int(width * ratio)), max(1, int(height * ratio))), interpolation=cv2.INTER_CUBIC)


def _upscale_image_to_mode(image: np.ndarray, runner: ModelRunner, mode: OutputMode, model_scale: int) -> np.ndarray:
    if mode == OutputMode.SCALE_2X:
        if model_scale == 2:
            return runner.upscale(image)
        if model_scale == 4:
            upscaled = runner.upscale(image)
            h, w = image.shape[:2]
            return cv2.resize(upscaled, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    if mode == OutputMode.SCALE_4X:
        if model_scale == 4:
            return runner.upscale(image)
        if model_scale == 2:
            return runner.upscale(runner.upscale(image))

    targets = {
        OutputMode.FIT_1080P: (1920, 1080),
        OutputMode.FIT_2K: (2560, 1440),
        OutputMode.FIT_4K: (3840, 2160),
    }
    if mode in targets:
        target_width, target_height = targets[mode]
        current = image
        while current.shape[1] < target_width and current.shape[0] < target_height:
            current = runner.upscale(current)
        return _fit_to_resolution(current, target_width, target_height)

    raise UpscaleError(f"Unsupported mode/model combination: mode={mode.value}, model_scale={model_scale}")


def upscale_image(job: UpscaleJob, progress: ProgressCallback | None = None, runner: ModelRunner | None = None) -> Path:
    progress = progress or (lambda _message: None)
    progress(f"Loading image on {job.backend.value} / {job.device_label}...")
    runner = runner or create_runner(job)
    image = load_image(job.input_path)
    progress(f"Upscaling image with {job.backend.value} / {job.device_label}...")
    upscaled = _upscale_image_to_mode(image, runner, job.output_mode, job.model_scale)
    progress("Saving image...")
    return save_image(job, upscaled)


def process_image_batch(jobs: list[UpscaleJob], progress: ProgressCallback | None = None) -> list[Path]:
    if not jobs:
        return []
    progress = progress or (lambda _message: None)
    progress(f"Initializing {jobs[0].backend.value} / {jobs[0].device_label} for batch...")
    runner = create_runner(jobs[0])
    outputs: list[Path] = []
    total = len(jobs)
    for index, job in enumerate(jobs, start=1):
        progress(f"PROGRESS_IMAGE|{index}|{total}|{job.input_path.name}|{job.backend.value}|{job.device_label}")
        outputs.append(upscale_image(job, progress, runner=runner))
    progress("Batch image processing complete.")
    return outputs


def process_job(job: UpscaleJob, progress: ProgressCallback | None = None) -> Path:
    return upscale_image(job, progress)
