# Dionysus AI Image Upscaler

Python desktop app for AI image upscaling with a Dionysus-branded dark GUI and selectable inference backend.

## Features

- Dark desktop GUI built with `customtkinter`
- Single-image and bulk-image input
- Backend selection:
  - `CPU`
  - `CUDA` for NVIDIA GPUs through `onnxruntime-gpu`
  - `TensorRT` for NVIDIA GPUs when the runtime is installed
  - `OpenCL` through OpenCV DNN for AMD / Intel / OpenCL-capable GPUs
- Output presets:
  - `2x`
  - `4x`
  - `Fit to 1080p`
  - `Fit to 2K`
  - `Fit to 4K`
- Export presets for standard and production-oriented image output
- Local model dropdown that scans the `models/` folder
- Dionysus logo branding, splash screen, and custom window icon

## Important limitations

- This repository includes sample ONNX models in the local `models/` folder for testing and demonstration.
- For `CUDA`, install `onnxruntime-gpu` and matching NVIDIA CUDA/cuDNN dependencies.
- For `TensorRT`, install the TensorRT runtime libraries supported by your ONNX Runtime build.
- For `OpenCL`, install a GPU driver that exposes OpenCL and use an ONNX model OpenCV DNN can run.
- Best results come from 2x or 4x ESRGAN-style ONNX models.

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

For NVIDIA CUDA execution:

```powershell
pip install -e .[nvidia]
```

## Run

```powershell
ai-upscaler
```

## Models

The repository already includes sample ONNX upscaling models in the local `models/` folder, and you can add more models there to make them appear in the GUI dropdown. You can also browse to a model manually from the app.

The current pipeline assumes a typical image-to-image super-resolution model with:

- Input tensor: `NCHW`
- 3-channel RGB input
- Float input scaled to `0..1`
- Float output scaled to `0..1`

If your model uses different preprocessing or fixed input sizes, update the preprocessing logic in `src/ai_upscaler/pipeline.py`.

## GitHub-ready files

Keep these in the repository:

- `src/`
- `pyproject.toml`
- `README.md`
- `Dionysus_AI_logo.png`
- `Dionysus_AI_logo.ico`
- `models/`

Do not commit generated folders like `build/`, `dist/`, or large local model weights unless you intentionally want them versioned.
