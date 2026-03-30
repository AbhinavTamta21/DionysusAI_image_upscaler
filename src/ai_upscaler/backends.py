from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

import cv2

from .config import Backend


@dataclass(slots=True)
class BackendStatus:
    backend: Backend
    available: bool
    detail: str


@dataclass(slots=True)
class DeviceInfo:
    backend: Backend
    label: str
    device_id: int | None
    selectable: bool
    detail: str


def _candidate_runtime_dirs() -> list[Path]:
    candidates: list[Path] = []
    module_names = [
        "tensorrt_libs",
        "nvidia.cublas",
        "nvidia.cuda_runtime",
        "nvidia.cudnn",
        "torch",
    ]
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=["__path__", "__file__"])
            package_paths = list(getattr(module, "__path__", []))
            if package_paths:
                for package_path in package_paths:
                    base = Path(package_path).resolve()
                    candidates.append(base)
                    for child in (base / "bin", base / "lib"):
                        if child.exists():
                            candidates.append(child)
            else:
                module_file = getattr(module, "__file__", None)
                if module_file:
                    base = Path(module_file).resolve().parent
                    candidates.append(base)
                    for child in (base / "bin", base / "lib"):
                        if child.exists():
                            candidates.append(child)
        except Exception:
            continue

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    static_candidates = [
        site_packages / "tensorrt_libs",
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cuda_runtime" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "torch" / "lib",
    ]
    for path in static_candidates:
        if path.exists():
            candidates.append(path)
    return candidates


def prepare_acceleration_runtime() -> list[Path]:
    added: list[Path] = []
    seen: set[str] = set()
    for path in _candidate_runtime_dirs():
        resolved = str(path.resolve())
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        try:
            os.add_dll_directory(resolved)
        except (AttributeError, FileNotFoundError, OSError):
            pass
        os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
        added.append(path)
    return added


def _list_nvidia_devices(backend: Backend, detail: str) -> list[DeviceInfo]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        return [
            DeviceInfo(
                backend=backend,
                label=f"{backend.value} device 0",
                device_id=0,
                selectable=True,
                detail=f"{detail} GPU names could not be queried: {exc}",
            )
        ]

    devices: list[DeviceInfo] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) != 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        devices.append(
            DeviceInfo(
                backend=backend,
                label=f"GPU {index}: {parts[1]}",
                device_id=index,
                selectable=True,
                detail=detail,
            )
        )
    return devices


def _list_opencl_devices() -> list[DeviceInfo]:
    try:
        import pyopencl as cl  # type: ignore

        devices: list[DeviceInfo] = []
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                if device.type & cl.device_type.GPU:
                    devices.append(
                        DeviceInfo(
                            backend=Backend.OPENCL,
                            label=f"{platform.name.strip()} / {device.name.strip()}",
                            device_id=None,
                            selectable=False,
                            detail="Detected via pyopencl. OpenCV DNN may still use the default OpenCL device.",
                        )
                    )
        if devices:
            return devices
    except Exception:
        pass

    if cv2.ocl.haveOpenCL():
        return [
            DeviceInfo(
                backend=Backend.OPENCL,
                label="OpenCL default device",
                device_id=None,
                selectable=False,
                detail="OpenCL detected by OpenCV. Install pyopencl for detailed device listing.",
            )
        ]
    return []


def _has_tensorrt_runtime() -> tuple[bool, str]:
    added = prepare_acceleration_runtime()
    candidates = [
        "nvinfer.dll",
        "nvinfer_plugin.dll",
        "nvinfer_10.dll",
        "nvinfer_plugin_10.dll",
        "nvinfer_9.dll",
        "nvinfer_plugin_9.dll",
        "nvonnxparser.dll",
        "nvonnxparser_10.dll",
        "nvonnxparser_9.dll",
    ]
    loaded = []
    for name in candidates:
        try:
            ctypes.WinDLL(name)
            loaded.append(name)
        except OSError:
            continue

    has_core = any(name.startswith("nvinfer") for name in loaded)
    has_parser = any(name.startswith("nvonnxparser") for name in loaded)
    if has_core and has_parser:
        detail = f"TensorRT runtime libraries found: {', '.join(loaded)}"
        if added:
            detail += f" | DLL path added: {added[0]}"
        return True, detail

    detail = "TensorRT provider is reported by ONNX Runtime, but TensorRT runtime DLLs were not found on PATH."
    if added:
        detail += f" Tried DLL path: {added[0]}"
    return False, detail


def detect_backends() -> tuple[list[BackendStatus], dict[Backend, list[DeviceInfo]]]:
    prepare_acceleration_runtime()

    cpu_devices = [
        DeviceInfo(
            backend=Backend.CPU,
            label="CPU",
            device_id=None,
            selectable=True,
            detail="Default CPU execution.",
        )
    ]

    try:
        import onnxruntime as ort

        providers = set(ort.get_available_providers())
        provider_text = ", ".join(sorted(providers)) if providers else "No providers reported."
    except Exception as exc:
        providers = set()
        provider_text = f"onnxruntime unavailable: {exc}"

    has_cuda = "CUDAExecutionProvider" in providers
    provider_has_tensorrt = "TensorrtExecutionProvider" in providers
    tensorrt_runtime_ok, tensorrt_detail = _has_tensorrt_runtime() if provider_has_tensorrt else (False, "TensorRT provider not reported by ONNX Runtime.")
    has_tensorrt = provider_has_tensorrt and tensorrt_runtime_ok

    cuda_devices = _list_nvidia_devices(Backend.CUDA, "CUDA / ONNX Runtime.") if has_cuda else []
    tensorrt_devices = _list_nvidia_devices(Backend.TENSORRT, "TensorRT / ONNX Runtime.") if has_tensorrt else []
    opencl_devices = _list_opencl_devices()

    statuses = [
        BackendStatus(backend=Backend.CPU, available=True, detail="Always available."),
        BackendStatus(backend=Backend.CUDA, available=has_cuda, detail=provider_text),
        BackendStatus(backend=Backend.TENSORRT, available=has_tensorrt, detail=tensorrt_detail if provider_has_tensorrt else provider_text),
        BackendStatus(
            backend=Backend.OPENCL,
            available=bool(opencl_devices),
            detail=(
                f"OpenCL path detected. {len(opencl_devices)} device entry available."
                if opencl_devices
                else "OpenCL not detected by OpenCV."
            ),
        ),
    ]

    devices = {
        Backend.CPU: cpu_devices,
        Backend.CUDA: cuda_devices,
        Backend.TENSORRT: tensorrt_devices,
        Backend.OPENCL: opencl_devices,
    }
    return statuses, devices

