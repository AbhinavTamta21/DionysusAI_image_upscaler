from __future__ import annotations

import threading
import time
import sys
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import customtkinter as ctk
from PIL import Image

from .backends import DeviceInfo, detect_backends
from .config import Backend, ExportPreset, OutputFormat, OutputMode, UpscaleJob
from .pipeline import UpscaleError, get_media_info, is_image_file, process_image_batch, process_job


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]

IMAGE_PROGRESS_PREFIX = "PROGRESS_IMAGE|"


class UpscalerApp(ctk.CTk):
    BG = "#050505"
    PANEL = "#0f0f0f"
    PANEL_ALT = "#151515"
    BORDER = "#f7c61d"
    ACCENT = "#eca11f"
    ACCENT_HOVER = "#ffcc32"
    TEXT = "#f7f4ef"
    MUTED = "#c9b88a"

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Dionysus AI Image Upscaler")
        self.geometry("1240x860")
        self.minsize(1060, 720)
        self.configure(fg_color=self.BG)

        self.bundle_root = _bundle_root()
        self.runtime_root = _runtime_root()
        self.models_dir = self.runtime_root / "models"
        self.logo_path = self.bundle_root / "Dionysus_AI_logo.png"
        self.logo_icon_path = self.bundle_root / "Dionysus_AI_logo.ico"
        self.logo_image: ctk.CTkImage | None = None
        self.footer_logo_image: ctk.CTkImage | None = None
        self.window_icon: tk.PhotoImage | None = None
        self.splash_logo: ctk.CTkImage | None = None
        self.model_choices: dict[str, Path] = {}

        self.backend_var = ctk.StringVar(value=Backend.CPU.value)
        self.device_var = ctk.StringVar(value="Detecting devices...")
        self.output_mode_var = ctk.StringVar(value=OutputMode.FIT_2K.value)
        self.model_scale_var = ctk.StringVar(value="2")
        self.export_preset_var = ctk.StringVar(value=ExportPreset.STANDARD.value)
        self.output_format_var = ctk.StringVar(value=OutputFormat.KEEP.value)
        self.jpeg_quality_var = ctk.StringVar(value="95")
        self.png_compression_var = ctk.StringVar(value="1")
        self.model_choice_var = ctk.StringVar(value="No local models found")
        self.model_var = ctk.StringVar()
        self.image_input_mode_var = ctk.StringVar(value="Single image")
        self.image_input_var = ctk.StringVar()
        self.image_output_var = ctk.StringVar()

        self.status_var = ctk.StringVar(value="Choose an image or batch, then start upscaling.")
        self.media_info_var = ctk.StringVar(value="No image selected.")
        self.device_info_var = ctk.StringVar(value="Scanning available acceleration backends...")
        self.export_info_var = ctk.StringVar(value="Standard export: JPG quality 95, PNG compression 1.")

        self.available_devices: dict[Backend, list[DeviceInfo]] = {}
        self.job_started_at: float | None = None
        self.selected_images: list[Path] = []
        self.splash_opened_at: float | None = None
        self.splash_window: ctk.CTkToplevel | None = None
        self.splash_progress: ctk.CTkProgressBar | None = None
        self.backend_ready = False
        self._pending_media_refresh: Path | None = None

        self._load_logo()
        self._apply_window_branding()
        self._build_layout()
        self._refresh_model_choices()
        self._apply_export_preset(ExportPreset.STANDARD.value)
        self.after(40, self._show_splash)
        self.after(80, self._load_backends_async)

    def _load_logo(self) -> None:
        if not self.logo_path.exists():
            self.logo_image = None
            self.footer_logo_image = None
            self.splash_logo = None
            return
        image = Image.open(self.logo_path)
        self.logo_image = ctk.CTkImage(light_image=image, dark_image=image, size=(136, 136))
        self.footer_logo_image = ctk.CTkImage(light_image=image, dark_image=image, size=(92, 92))
        self.splash_logo = ctk.CTkImage(light_image=image, dark_image=image, size=(210, 210))

    def _apply_window_branding(self) -> None:
        if self.logo_icon_path.exists():
            try:
                self.iconbitmap(default=str(self.logo_icon_path))
            except Exception:
                pass
        if not self.logo_path.exists():
            return
        try:
            self.window_icon = tk.PhotoImage(file=str(self.logo_path))
            self.iconphoto(True, self.window_icon)
        except Exception:
            self.window_icon = None

    def _show_splash(self) -> None:
        if self.splash_logo is None:
            return

        self.withdraw()
        splash = ctk.CTkToplevel(self)
        self.splash_window = splash
        self.splash_opened_at = time.perf_counter()
        splash.title("Dionysus AI")
        splash.geometry("500x500")
        splash.resizable(False, False)
        splash.configure(fg_color=self.BG)
        splash.overrideredirect(True)
        if self.window_icon is not None:
            try:
                splash.iconphoto(True, self.window_icon)
            except Exception:
                pass

        splash.update_idletasks()
        screen_w = splash.winfo_screenwidth()
        screen_h = splash.winfo_screenheight()
        x = max(0, (screen_w - 500) // 2)
        y = max(0, (screen_h - 500) // 2)
        splash.geometry(f"500x500+{x}+{y}")

        frame = ctk.CTkFrame(splash, fg_color=self.PANEL, corner_radius=28, border_width=1, border_color=self.BORDER)
        frame.pack(expand=True, fill="both", padx=18, pady=18)

        ctk.CTkLabel(frame, text="", image=self.splash_logo).pack(pady=(26, 16))
        ctk.CTkLabel(frame, text="Dionysus AI", text_color=self.TEXT, font=ctk.CTkFont(size=30, weight="bold")).pack()
        ctk.CTkLabel(frame, text="Open Source AI Image Upscaler", text_color=self.MUTED, font=ctk.CTkFont(size=15)).pack(pady=(8, 18))

        bar = ctk.CTkProgressBar(frame, width=260, progress_color=self.ACCENT, fg_color="#2a2a2a")
        bar.pack(pady=(6, 10))
        bar.configure(mode="indeterminate")
        bar.start()
        self.splash_progress = bar

        self.splash_status_label = ctk.CTkLabel(frame, text="Detecting models, backends, and UI...", text_color=self.MUTED)
        self.splash_status_label.pack()

        if self.backend_ready:
            self._close_splash_when_ready()

    def _close_splash_when_ready(self) -> None:
        if self.splash_window is None:
            self.deiconify()
            self.lift()
            return

        elapsed = 0.0 if self.splash_opened_at is None else time.perf_counter() - self.splash_opened_at
        remaining_ms = max(0, int((0.85 - elapsed) * 1000))

        def close_now() -> None:
            if self.splash_progress is not None:
                try:
                    self.splash_progress.stop()
                except Exception:
                    pass
            if self.splash_window is not None and self.splash_window.winfo_exists():
                self.splash_window.destroy()
            self.splash_window = None
            self.deiconify()
            self.lift()
            self.focus_force()

        self.after(remaining_ms, close_now)

    def _load_backends_async(self) -> None:
        threading.Thread(target=self._load_backends_worker, daemon=True).start()

    def _load_backends_worker(self) -> None:
        try:
            statuses, devices = detect_backends()
        except Exception as exc:
            self.after(0, lambda: self._apply_backend_failure(str(exc)))
            return
        self.after(0, lambda: self._apply_backend_results(statuses, devices))

    def _apply_backend_failure(self, message: str) -> None:
        self.available_devices = {Backend.CPU: [DeviceInfo(backend=Backend.CPU, label="CPU", device_id=None, selectable=True, detail="Default CPU execution.")]}
        self.backend_menu.configure(values=[Backend.CPU.value], state="normal")
        self.backend_var.set(Backend.CPU.value)
        self.device_menu.configure(values=["CPU"], state="normal")
        self.device_var.set("CPU")
        self.device_info_var.set(f"Backend scan failed, using CPU only.\n{message}")
        self.backend_ready = True
        self._close_splash_when_ready()

    def _apply_backend_results(self, statuses, devices) -> None:
        self.available_devices = devices
        backend_values = [status.backend.value for status in statuses if status.available] or [Backend.CPU.value]
        self.backend_menu.configure(values=backend_values, state="normal")
        preferred_value = Backend.CPU.value
        for preferred in (Backend.TENSORRT.value, Backend.CUDA.value, Backend.OPENCL.value, Backend.CPU.value):
            if preferred in backend_values:
                preferred_value = preferred
                break
        self.backend_var.set(preferred_value)
        self._refresh_device_menu(Backend(preferred_value))
        self.backend_ready = True
        if hasattr(self, "splash_status_label"):
            self.splash_status_label.configure(text="Backends ready. Launching workspace...")
        if self._pending_media_refresh is not None:
            path = self._pending_media_refresh
            self._pending_media_refresh = None
            if self.image_input_mode_var.get() == "Bulk images":
                self._update_batch_info()
            else:
                self._update_media_info(path)
        self._close_splash_when_ready()

    def _menu_kwargs(self) -> dict:
        return {
            "fg_color": self.PANEL_ALT,
            "button_color": self.ACCENT,
            "button_hover_color": self.ACCENT_HOVER,
            "dropdown_fg_color": self.PANEL_ALT,
            "dropdown_hover_color": self.ACCENT,
            "text_color": self.TEXT,
            "corner_radius": 12,
            "dynamic_resizing": False,
        }

    def _button_kwargs(self) -> dict:
        return {
            "fg_color": self.ACCENT,
            "hover_color": self.ACCENT_HOVER,
            "text_color": "#111111",
            "corner_radius": 12,
        }

    def _entry_kwargs(self) -> dict:
        return {
            "fg_color": self.PANEL_ALT,
            "border_color": self.BORDER,
            "text_color": self.TEXT,
            "corner_radius": 12,
        }

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        root = ctk.CTkFrame(self, fg_color=self.BG, corner_radius=0)
        root.grid(row=0, column=0, sticky="nsew")
        root.grid_columnconfigure(0, weight=2)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(root, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, padx=28, pady=(24, 16), sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        if self.logo_image is not None:
            logo_shell = ctk.CTkFrame(header, fg_color=self.PANEL, corner_radius=22, border_width=1, border_color=self.BORDER)
            logo_shell.grid(row=0, column=0, rowspan=2, padx=(0, 18), sticky="w")
            ctk.CTkLabel(logo_shell, text="", image=self.logo_image).pack(padx=10, pady=10)

        ctk.CTkLabel(header, text="Dionysus AI", text_color=self.TEXT, font=ctk.CTkFont(size=34, weight="bold")).grid(row=0, column=1, sticky="sw")
        ctk.CTkLabel(header, text="Open source  AI image upscaler .", text_color=self.MUTED, font=ctk.CTkFont(size=16)).grid(row=1, column=1, sticky="nw")

        left = ctk.CTkScrollableFrame(root, fg_color=self.PANEL, corner_radius=24, border_width=1, border_color=self.BORDER)
        left.grid(row=1, column=0, padx=(28, 14), pady=(0, 28), sticky="nsew")
        left.grid_columnconfigure(1, weight=1)

        right = ctk.CTkFrame(root, fg_color=self.PANEL_ALT, corner_radius=24, border_width=1, border_color=self.BORDER)
        right.grid(row=1, column=1, padx=(14, 28), pady=(0, 28), sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)
        right.grid(row=1, column=1, padx=(14, 28), pady=(0, 28), sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        self._build_controls(left)
        self._build_status_panel(right)

    def _build_controls(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(parent, text="Image Workflow", text_color=self.TEXT, font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, columnspan=4, padx=18, pady=(12, 8), sticky="w")
        ctk.CTkLabel(parent, text="Single-image and bulk-image processing only.", text_color=self.MUTED).grid(row=1, column=0, columnspan=4, padx=18, pady=(0, 16), sticky="w")

        ctk.CTkLabel(parent, text="Mode", text_color=self.TEXT).grid(row=2, column=0, padx=18, pady=10, sticky="w")
        ctk.CTkOptionMenu(parent, variable=self.image_input_mode_var, values=["Single image", "Bulk images"], command=lambda _value: self._on_image_mode_changed(), **self._menu_kwargs()).grid(row=2, column=1, padx=18, pady=10, sticky="ew")

        self._add_path_row(parent, 3, "Image Input", self.image_input_var, self._pick_image_input)
        self._add_path_row(parent, 4, "Image Output", self.image_output_var, self._pick_image_output)

        ctk.CTkLabel(parent, text="Model", text_color=self.TEXT).grid(row=5, column=0, padx=18, pady=10, sticky="w")
        self.model_menu = ctk.CTkOptionMenu(parent, variable=self.model_choice_var, values=["No local models found"], command=self._on_model_selected, **self._menu_kwargs())
        self.model_menu.grid(row=5, column=1, padx=18, pady=10, sticky="ew")
        ctk.CTkButton(parent, text="Refresh", width=92, command=self._refresh_model_choices, **self._button_kwargs()).grid(row=5, column=2, padx=(0, 8), pady=10)
        ctk.CTkButton(parent, text="Browse", width=92, command=self._pick_model, **self._button_kwargs()).grid(row=5, column=3, padx=(0, 18), pady=10)
        ctk.CTkLabel(parent, textvariable=self.model_var, text_color=self.MUTED, wraplength=580, justify="left").grid(row=6, column=1, columnspan=3, padx=18, pady=(0, 12), sticky="ew")

        ctk.CTkLabel(parent, text="Backend", text_color=self.TEXT).grid(row=7, column=0, padx=18, pady=10, sticky="w")
        self.backend_menu = ctk.CTkOptionMenu(parent, variable=self.backend_var, values=["Detecting..."], state="disabled", command=self._on_backend_changed, **self._menu_kwargs())
        self.backend_menu.grid(row=7, column=1, padx=18, pady=10, sticky="ew")

        ctk.CTkLabel(parent, text="Device", text_color=self.TEXT).grid(row=8, column=0, padx=18, pady=10, sticky="w")
        self.device_menu = ctk.CTkOptionMenu(parent, variable=self.device_var, values=["Detecting devices..."], state="disabled", command=lambda _value: self._update_device_info(), **self._menu_kwargs())
        self.device_menu.grid(row=8, column=1, padx=18, pady=10, sticky="ew")
        ctk.CTkLabel(parent, textvariable=self.device_info_var, text_color=self.MUTED, wraplength=580, justify="left").grid(row=9, column=1, columnspan=3, padx=18, pady=(0, 12), sticky="ew")

        ctk.CTkLabel(parent, text="Output Mode", text_color=self.TEXT).grid(row=10, column=0, padx=18, pady=10, sticky="w")
        ctk.CTkOptionMenu(parent, variable=self.output_mode_var, values=[mode.value for mode in OutputMode], **self._menu_kwargs()).grid(row=10, column=1, padx=18, pady=10, sticky="ew")

        ctk.CTkLabel(parent, text="Model Scale", text_color=self.TEXT).grid(row=11, column=0, padx=18, pady=10, sticky="w")
        ctk.CTkOptionMenu(parent, variable=self.model_scale_var, values=["2", "4"], **self._menu_kwargs()).grid(row=11, column=1, padx=18, pady=10, sticky="ew")

        ctk.CTkLabel(parent, text="Export Preset", text_color=self.TEXT).grid(row=12, column=0, padx=18, pady=10, sticky="w")
        ctk.CTkOptionMenu(parent, variable=self.export_preset_var, values=[preset.value for preset in ExportPreset], command=self._apply_export_preset, **self._menu_kwargs()).grid(row=12, column=1, padx=18, pady=10, sticky="ew")

        ctk.CTkLabel(parent, text="Output Format", text_color=self.TEXT).grid(row=13, column=0, padx=18, pady=10, sticky="w")
        ctk.CTkOptionMenu(parent, variable=self.output_format_var, values=[fmt.value for fmt in OutputFormat], command=lambda _value: self._update_export_info(), **self._menu_kwargs()).grid(row=13, column=1, padx=18, pady=10, sticky="ew")

        ctk.CTkLabel(parent, text="JPEG Quality", text_color=self.TEXT).grid(row=14, column=0, padx=18, pady=10, sticky="w")
        ctk.CTkOptionMenu(parent, variable=self.jpeg_quality_var, values=["85", "90", "95", "98", "100"], command=lambda _value: self._update_export_info(), **self._menu_kwargs()).grid(row=14, column=1, padx=18, pady=10, sticky="ew")

        ctk.CTkLabel(parent, text="PNG Compression", text_color=self.TEXT).grid(row=15, column=0, padx=18, pady=10, sticky="w")
        ctk.CTkOptionMenu(parent, variable=self.png_compression_var, values=["0", "1", "3", "6", "9"], command=lambda _value: self._update_export_info(), **self._menu_kwargs()).grid(row=15, column=1, padx=18, pady=10, sticky="ew")

        ctk.CTkLabel(parent, textvariable=self.export_info_var, text_color=self.MUTED, wraplength=580, justify="left").grid(row=16, column=1, columnspan=3, padx=18, pady=(0, 18), sticky="ew")

        self.image_run_button = ctk.CTkButton(parent, text="Start Image Upscaling", height=48, command=self._start_image_job, **self._button_kwargs())
        self.image_run_button.grid(row=17, column=0, columnspan=4, padx=18, pady=(10, 18), sticky="ew")

    def _build_status_panel(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(parent, text="Session", text_color=self.TEXT, font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        ctk.CTkLabel(parent, textvariable=self.media_info_var, text_color=self.TEXT, justify="left", anchor="w", wraplength=320).grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        ctk.CTkLabel(parent, textvariable=self.status_var, text_color=self.ACCENT_HOVER, justify="left", anchor="w", wraplength=320).grid(row=2, column=0, padx=20, pady=(0, 18), sticky="ew")

        tips = ctk.CTkTextbox(parent, height=320, fg_color=self.PANEL, border_width=1, border_color=self.BORDER, text_color=self.TEXT)
        tips.grid(row=3, column=0, padx=20, pady=(0, 16), sticky="nsew")
        tips.insert(
            "1.0",
            "Recommended workflow:\n\n"
            "1. Keep ONNX models inside the local models folder for quick selection.\n"
            "2. Use CUDA or TensorRT only when the session really activates that provider.\n"
            "3. Use PNG or TIFF for your best-quality master exports.\n"
            "4. Bulk mode reuses the same model session, so it is faster than launching one image at a time.\n"
            "5. Supports CUDA , TensorRT , CPU , OpenCL.\n",
        )
        tips.configure(state="disabled")

        if self.footer_logo_image is not None:
            footer = ctk.CTkFrame(parent, fg_color="transparent")
            footer.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="e")
            ctk.CTkLabel(footer, text="", image=self.footer_logo_image).pack(anchor="e")

    def _add_path_row(self, parent, row: int, label: str, variable: ctk.StringVar, command) -> None:
        ctk.CTkLabel(parent, text=label, text_color=self.TEXT).grid(row=row, column=0, padx=18, pady=10, sticky="w")
        ctk.CTkEntry(parent, textvariable=variable, **self._entry_kwargs()).grid(row=row, column=1, padx=(18, 8), pady=10, sticky="ew")
        ctk.CTkButton(parent, text="Browse", width=92, command=command, **self._button_kwargs()).grid(row=row, column=2, columnspan=2, padx=(0, 18), pady=10)

    def _refresh_model_choices(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_choices = {path.name: path for path in sorted(self.models_dir.glob("*.onnx"))}
        values = list(self.model_choices.keys()) or ["No local models found"]
        self.model_menu.configure(values=values)
        choice = values[0]
        self.model_choice_var.set(choice)
        if choice in self.model_choices:
            self.model_var.set(str(self.model_choices[choice]))
        else:
            self.model_var.set(f"Put .onnx models in: {self.models_dir}")

    def _on_model_selected(self, selected: str) -> None:
        path = self.model_choices.get(selected)
        if path:
            self.model_var.set(str(path))

    def _pick_model(self) -> None:
        path = filedialog.askopenfilename(title="Select ONNX model", initialdir=str(self.models_dir), filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")])
        if not path:
            return
        selected = Path(path)
        if selected.parent.resolve() == self.models_dir.resolve():
            self._refresh_model_choices()
            self.model_choice_var.set(selected.name)
        else:
            self.model_choices[f"External: {selected.name}"] = selected
            self.model_menu.configure(values=list(self.model_choices.keys()))
            self.model_choice_var.set(f"External: {selected.name}")
        self.model_var.set(str(selected))

    def _on_backend_changed(self, selected: str) -> None:
        self._refresh_device_menu(Backend(selected))
        if self.selected_images:
            if self.image_input_mode_var.get() == "Bulk images":
                self._update_batch_info()
            else:
                self._update_media_info(self.selected_images[0])

    def _refresh_device_menu(self, backend: Backend) -> None:
        devices = self.available_devices.get(backend) or [DeviceInfo(backend=backend, label="Default", device_id=None, selectable=False, detail="No specific device listing available.")]
        self.available_devices[backend] = devices
        values = [device.label for device in devices]
        self.device_menu.configure(values=values, state="normal")
        self.device_var.set(values[0])
        self._update_device_info()

    def _selected_device(self) -> DeviceInfo:
        backend = Backend(self.backend_var.get())
        for device in self.available_devices.get(backend, []):
            if device.label == self.device_var.get():
                return device
        return DeviceInfo(backend=backend, label=self.device_var.get() or "Default", device_id=None, selectable=False, detail="No device details available.")

    def _update_device_info(self) -> None:
        device = self._selected_device()
        extra = "Selectable" if device.selectable else "Display only"
        self.device_info_var.set(f"{device.detail}\n{extra}")

    def _apply_export_preset(self, selected: str) -> None:
        if selected == ExportPreset.PRODUCTION.value:
            self.output_format_var.set(OutputFormat.PNG.value)
            self.jpeg_quality_var.set("100")
            self.png_compression_var.set("0")
        else:
            self.output_format_var.set(OutputFormat.KEEP.value)
            self.jpeg_quality_var.set("95")
            self.png_compression_var.set("1")
        self._update_export_info()

    def _update_export_info(self) -> None:
        self.export_info_var.set(
            f"Format: {self.output_format_var.get()} | JPEG quality: {self.jpeg_quality_var.get()} | PNG compression: {self.png_compression_var.get()}\n"
            f"Models folder: {self.models_dir}"
        )

    def _on_image_mode_changed(self) -> None:
        self.selected_images = []
        self.image_input_var.set("")
        self.image_output_var.set("")
        self.media_info_var.set("No image selected.")

    def _image_suffix(self, source: Path) -> str:
        fmt = OutputFormat(self.output_format_var.get())
        if fmt == OutputFormat.KEEP:
            return source.suffix or ".png"
        return {OutputFormat.JPG: ".jpg", OutputFormat.PNG: ".png", OutputFormat.TIFF: ".tiff"}[fmt]

    def _pick_image_input(self) -> None:
        if self.image_input_mode_var.get() == "Bulk images":
            paths = filedialog.askopenfilenames(title="Select images", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
            if not paths:
                return
            self.selected_images = [Path(path) for path in paths]
            self.image_input_var.set(f"{len(self.selected_images)} image(s) selected")
            self._update_batch_info()
            if not self.image_output_var.get():
                self.image_output_var.set(str(Path.cwd() / "upscaled_images"))
            return

        path = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
        if not path:
            return
        selected = Path(path)
        self.selected_images = [selected]
        self.image_input_var.set(path)
        if self.backend_ready:
            self._update_media_info(selected)
        else:
            self._pending_media_refresh = selected
            self.media_info_var.set(f"File: {selected.name}\nResolution: scanning...\nBackend: detecting...")
        if not self.image_output_var.get():
            self.image_output_var.set(str(selected.with_name(f"{selected.stem}_upscaled{self._image_suffix(selected)}")))

    def _pick_image_output(self) -> None:
        if self.image_input_mode_var.get() == "Bulk images":
            path = filedialog.askdirectory(title="Select output folder")
            if path:
                self.image_output_var.set(path)
            return

        source = self.selected_images[0] if self.selected_images else Path.cwd() / "output.png"
        suffix = self._image_suffix(source)
        path = filedialog.asksaveasfilename(title="Select output image path", defaultextension=suffix, initialfile=f"{source.stem}_upscaled{suffix}")
        if path:
            self.image_output_var.set(path)

    def _update_media_info(self, path: Path) -> None:
        try:
            info = get_media_info(path)
        except Exception as exc:
            self.media_info_var.set(f"Could not read image info.\n{exc}")
            return
        self.media_info_var.set(
            f"File: {path.name}\n"
            f"Resolution: {info.width} x {info.height}\n"
            f"Backend: {self.backend_var.get()} / {self.device_var.get()}"
        )

    def _update_batch_info(self) -> None:
        count = len(self.selected_images)
        if not count:
            self.media_info_var.set("No image selected.")
            return
        self.media_info_var.set(
            f"Batch images: {count}\n"
            f"First file: {self.selected_images[0].name}\n"
            f"Backend: {self.backend_var.get()} / {self.device_var.get()}"
        )

    def _job_kwargs(self) -> dict:
        selected_device = self._selected_device()
        return {
            "backend": Backend(self.backend_var.get()),
            "output_mode": OutputMode(self.output_mode_var.get()),
            "model_scale": int(self.model_scale_var.get()),
            "device_id": selected_device.device_id,
            "device_label": selected_device.label,
            "output_format": OutputFormat(self.output_format_var.get()),
            "jpeg_quality": int(self.jpeg_quality_var.get()),
            "png_compression": int(self.png_compression_var.get()),
        }

    def _build_image_jobs(self):
        model_path = Path(self.model_var.get())
        if not self.model_var.get() or not model_path.exists():
            raise UpscaleError("Choose an existing ONNX model.")
        if self.image_input_mode_var.get() == "Bulk images":
            output_dir = Path(self.image_output_var.get())
            if not self.selected_images:
                raise UpscaleError("Choose one or more images for bulk mode.")
            if not self.image_output_var.get():
                raise UpscaleError("Choose an output folder for images.")
            jobs = []
            for input_path in self.selected_images:
                if not is_image_file(input_path):
                    raise UpscaleError("Bulk mode supports image files only.")
                base_output = output_dir / f"{input_path.stem}_upscaled{input_path.suffix}"
                jobs.append(UpscaleJob(input_path=input_path, output_path=base_output, model_path=model_path, **self._job_kwargs()))
            return jobs

        input_path = self.selected_images[0] if self.selected_images else Path(self.image_input_var.get())
        output_path = Path(self.image_output_var.get())
        if not input_path.exists():
            raise UpscaleError("Choose a valid input image.")
        if not self.image_output_var.get():
            raise UpscaleError("Choose an output path for the image.")
        return UpscaleJob(input_path=input_path, output_path=output_path, model_path=model_path, **self._job_kwargs())

    def _set_status(self, message: str) -> None:
        self.after(0, lambda: self.status_var.set(message))

    def _set_running(self, running: bool) -> None:
        def apply() -> None:
            self.image_run_button.configure(state="disabled" if running else "normal", text="Processing..." if running else "Start Image Upscaling")
        self.after(0, apply)

    def _format_seconds(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _progress_callback(self, message: str) -> None:
        if not message.startswith(IMAGE_PROGRESS_PREFIX):
            self._set_status(message)
            return

        _, index_text, total_text, filename, backend, device = message.split("|", maxsplit=5)
        index = int(index_text)
        total = int(total_text)
        now = time.perf_counter()
        if self.job_started_at is None:
            self.job_started_at = now
        elapsed = now - self.job_started_at
        avg = elapsed / index if index else 0.0
        remaining = max(total - index, 0) * avg if total else 0.0
        rate = index / elapsed if elapsed > 0 else 0.0
        self._set_status(
            f"Image {index}/{total}: {filename} on {backend} / {device}\n"
            f"Elapsed: {self._format_seconds(elapsed)} | Avg: {avg:.2f}s/item | Throughput: {rate:.2f}/s | ETA: {self._format_seconds(remaining)}"
        )

    def _start_image_job(self) -> None:
        try:
            payload = self._build_image_jobs()
        except UpscaleError as exc:
            messagebox.showerror("Validation error", str(exc))
            return
        self.job_started_at = time.perf_counter()
        self._set_running(True)
        self._set_status("Preparing image job...")
        threading.Thread(target=self._run_job, args=(payload,), daemon=True).start()

    def _run_job(self, payload) -> None:
        try:
            if isinstance(payload, list):
                outputs = process_image_batch(payload, self._progress_callback)
                self._set_status(f"Completed image batch: {len(outputs)} image(s)")
                self.after(0, lambda: messagebox.showinfo("Upscaling complete", f"Saved {len(outputs)} image(s) to:\n{payload[0].output_path.parent}"))
            else:
                output = process_job(payload, self._progress_callback)
                self._set_status(f"Completed on {payload.backend.value} / {payload.device_label}: {output}")
                self.after(0, lambda: messagebox.showinfo("Upscaling complete", f"Saved to:\n{output}"))
        except Exception as exc:
            self._set_status(f"Failed: {exc}")
            self.after(0, lambda: messagebox.showerror("Upscaling failed", str(exc)))
        finally:
            self._set_running(False)












