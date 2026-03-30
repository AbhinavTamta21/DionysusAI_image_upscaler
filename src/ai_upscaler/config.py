from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Backend(str, Enum):
    CPU = "CPU"
    CUDA = "CUDA"
    TENSORRT = "TensorRT"
    OPENCL = "OPENCL"


class OutputMode(str, Enum):
    SCALE_2X = "2x"
    SCALE_4X = "4x"
    FIT_1080P = "Fit to 1080p"
    FIT_2K = "Fit to 2K"
    FIT_4K = "Fit to 4K"


class OutputFormat(str, Enum):
    KEEP = "Keep input format"
    JPG = "JPG"
    PNG = "PNG"
    TIFF = "TIFF"


class ExportPreset(str, Enum):
    STANDARD = "Standard"
    PRODUCTION = "Production"


@dataclass(slots=True)
class UpscaleJob:
    input_path: Path
    output_path: Path
    model_path: Path
    backend: Backend
    output_mode: OutputMode
    model_scale: int
    device_id: int | None = None
    device_label: str = "Default"
    output_format: OutputFormat = OutputFormat.KEEP
    jpeg_quality: int = 95
    png_compression: int = 1
