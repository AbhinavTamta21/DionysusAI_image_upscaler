from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from ai_upscaler.backends import prepare_acceleration_runtime
else:
    from .backends import prepare_acceleration_runtime

prepare_acceleration_runtime()

try:
    import onnxruntime as ort

    try:
        ort.preload_dlls(directory="")
    except Exception:
        try:
            ort.preload_dlls()
        except Exception:
            pass
except Exception:
    pass

if __package__ in {None, ""}:
    from ai_upscaler.gui import UpscalerApp
else:
    from .gui import UpscalerApp


def main() -> None:
    app = UpscalerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
