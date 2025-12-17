import os
import random
import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageTk

# Check for OpenCV (best for reflect padding)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

# Check for PyTorch GPU support
try:
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


@dataclass
class AugMax:
    rot_deg: float = 5.0
    translate_pct: float = 2.0
    zoom_pct: float = 3.0
    brightness_pct: float = 8.0
    contrast_pct: float = 8.0
    saturation_pct: float = 10.0
    hue_deg: float = 6.0
    blur_radius: float = 0.6
    noise_sigma: float = 6.0
    jpeg_quality_drop: int = 25


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def list_images_dedup(root: Path):
    """
    Case-insensitive extension match + dedupe by normalized (lowercased) relative path.
    Prevents 'a.jpg' and 'a.JPG' being treated as two different inputs.
    """
    if not root.exists():
        return []

    seen = set()
    out = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue

        try:
            rel = p.relative_to(root)
        except Exception:
            rel = p.name

        key = str(rel).replace("\\", "/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    return sorted(out)


def subdirs_with_images(root: Path):
    """Return sorted relative subdirectories (including '.') that contain at least one supported image."""
    imgs = list_images_dedup(root)
    dirs = set()
    for p in imgs:
        try:
            rel = p.relative_to(root)
            parent = str(rel.parent).replace("\\", "/")
        except Exception:
            parent = "."
        if parent == "":
            parent = "."
        dirs.add(parent)
    if not dirs:
        return []
    return sorted(dirs)


def filter_images_by_subdirs(imgs, src_root: Path, allowed):
    """Filter images to only those whose parent directory (relative) is in allowed set. If allowed is None, keep all."""
    if allowed is None:
        return imgs
    out = []
    for p in imgs:
        try:
            rel = p.relative_to(src_root)
            parent = str(rel.parent).replace("\\", "/")
        except Exception:
            parent = "."
        if parent == "":
            parent = "."
        if parent in allowed:
            out.append(p)
    return out


def pil_to_fit(img: Image.Image, max_w: int, max_h: int):
    w, h = img.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    scale = min(scale, 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def hue_shift_rgb(img: Image.Image, hue_degrees: float) -> Image.Image:
    if abs(hue_degrees) < 1e-6:
        return img
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    rgb = arr.astype(np.float32) / 255.0

    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    diff = mx - mn

    h = np.zeros_like(mx)
    s = np.zeros_like(mx)
    v = mx

    mask = diff > 1e-6
    s[mask] = diff[mask] / (mx[mask] + 1e-12)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    idx = (mx == r) & mask
    h[idx] = (60 * ((g[idx] - b[idx]) / (diff[idx] + 1e-12)) + 360) % 360
    idx = (mx == g) & mask
    h[idx] = (60 * ((b[idx] - r[idx]) / (diff[idx] + 1e-12)) + 120) % 360
    idx = (mx == b) & mask
    h[idx] = (60 * ((r[idx] - g[idx]) / (diff[idx] + 1e-12)) + 240) % 360

    h = (h + hue_degrees) % 360

    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c

    z = np.zeros_like(h)
    rp = np.zeros_like(h); gp = np.zeros_like(h); bp = np.zeros_like(h)

    h0 = (h >= 0) & (h < 60)
    rp[h0], gp[h0], bp[h0] = c[h0], x[h0], z[h0]
    h1 = (h >= 60) & (h < 120)
    rp[h1], gp[h1], bp[h1] = x[h1], c[h1], z[h1]
    h2 = (h >= 120) & (h < 180)
    rp[h2], gp[h2], bp[h2] = z[h2], c[h2], x[h2]
    h3 = (h >= 180) & (h < 240)
    rp[h3], gp[h3], bp[h3] = z[h3], x[h3], c[h3]
    h4 = (h >= 240) & (h < 300)
    rp[h4], gp[h4], bp[h4] = x[h4], z[h4], c[h4]
    h5 = (h >= 300) & (h < 360)
    rp[h5], gp[h5], bp[h5] = c[h5], z[h5], x[h5]

    out = np.stack([(rp + m), (gp + m), (bp + m)], axis=2)
    out = (out * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def apply_affine_keep_size(img: Image.Image, angle_deg=0.0, translate_px=(0, 0), scale=1.0):
    """
    Apply affine transformation with REFLECT border mode for natural-looking edges.
    Uses OpenCV if available (best quality), otherwise PIL with edge color averaging.
    """
    w, h = img.size

    if CV2_AVAILABLE and cv2 is not None:
        # OpenCV method: BORDER_REFLECT_101 for best quality
        arr = np.array(img.convert("RGB"), dtype=np.uint8)

        # Build rotation matrix around center
        cx, cy = w / 2.0, h / 2.0
        tx, ty = translate_px

        # Rotation + scale matrix
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

        # Add translation
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # Apply with BORDER_REFLECT_101 (mirror at edges)
        result = cv2.warpAffine(
            arr, M_rot, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101
        )

        return Image.fromarray(result, mode="RGB")

    else:
        # PIL fallback: use averaged edge colors
        cx, cy = w / 2.0, h / 2.0
        theta = np.deg2rad(angle_deg)

        cos_t = np.cos(theta) * scale
        sin_t = np.sin(theta) * scale
        tx, ty = translate_px

        a = cos_t
        b = -sin_t
        d = sin_t
        e = cos_t

        c = cx + tx - a * cx - b * cy
        f = cy + ty - d * cx - e * cy

        det = a * e - b * d
        if abs(det) < 1e-12:
            return img
        ia = e / det
        ib = -b / det
        id_ = -d / det
        ie = a / det
        ic = -(ia * c + ib * f)
        if_ = -(id_ * c + ie * f)

        # Better fill color: average of image edges
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb)

        # Sample all 4 edges and compute average color
        top = arr[0, :, :]
        bottom = arr[-1, :, :]
        left = arr[:, 0, :]
        right = arr[:, -1, :]

        edge_colors = np.vstack([top, bottom, left, right])
        fill = tuple(edge_colors.mean(axis=0).round().astype(int).tolist())

        return img_rgb.transform(
            (w, h),
            Image.Transform.AFFINE,
            (ia, ib, ic, id_, ie, if_),
            resample=Image.Resampling.BICUBIC,
            fillcolor=fill
        )


def add_gaussian_noise(img: Image.Image, sigma: float):
    if sigma <= 0:
        return img
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def jpeg_roundtrip(img: Image.Image, quality: int):
    from io import BytesIO
    quality = int(clamp(quality, 5, 95))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_params(enabled, maxv: AugMax, strength: float):
    s = clamp(strength, 0.0, 1.0)
    p = {}

    if enabled["flip_h"]:
        p["flip_h"] = random.choice([True, False])
    if enabled["flip_v"]:
        p["flip_v"] = random.choice([True, False])

    if enabled["rotate"]:
        m = maxv.rot_deg * s
        p["angle"] = random.uniform(-m, m)

    if enabled["translate"]:
        m = maxv.translate_pct * s / 100.0
        p["tx_pct"] = random.uniform(-m, m)
        p["ty_pct"] = random.uniform(-m, m)

    if enabled["zoom"]:
        m = maxv.zoom_pct * s / 100.0
        p["zoom"] = 1.0 + random.uniform(-m, m)

    if enabled["brightness"]:
        m = maxv.brightness_pct * s / 100.0
        p["brightness"] = 1.0 + random.uniform(-m, m)

    if enabled["contrast"]:
        m = maxv.contrast_pct * s / 100.0
        p["contrast"] = 1.0 + random.uniform(-m, m)

    if enabled["saturation"]:
        m = maxv.saturation_pct * s / 100.0
        p["saturation"] = 1.0 + random.uniform(-m, m)

    if enabled["hue"]:
        m = maxv.hue_deg * s
        p["hue_deg"] = random.uniform(-m, m)

    if enabled["blur"]:
        m = maxv.blur_radius * s
        p["blur"] = random.uniform(0.0, m)

    if enabled["noise"]:
        m = maxv.noise_sigma * s
        p["noise_sigma"] = random.uniform(0.0, m)

    if enabled["jpeg"]:
        drop = int(round(maxv.jpeg_quality_drop * s))
        p["jpeg_quality"] = 95 - drop

    return p


def max_params(enabled, maxv: AugMax, strength: float):
    s = clamp(strength, 0.0, 1.0)
    p = {}

    if enabled["flip_h"]:
        p["flip_h"] = True
    if enabled["flip_v"]:
        p["flip_v"] = True

    if enabled["rotate"]:
        p["angle"] = maxv.rot_deg * s

    if enabled["translate"]:
        m = maxv.translate_pct * s / 100.0
        p["tx_pct"] = m
        p["ty_pct"] = m

    if enabled["zoom"]:
        m = maxv.zoom_pct * s / 100.0
        p["zoom"] = 1.0 + abs(m)

    if enabled["brightness"]:
        m = maxv.brightness_pct * s / 100.0
        p["brightness"] = 1.0 + abs(m)

    if enabled["contrast"]:
        m = maxv.contrast_pct * s / 100.0
        p["contrast"] = 1.0 + abs(m)

    if enabled["saturation"]:
        m = maxv.saturation_pct * s / 100.0
        p["saturation"] = 1.0 + abs(m)

    if enabled["hue"]:
        p["hue_deg"] = maxv.hue_deg * s

    if enabled["blur"]:
        p["blur"] = maxv.blur_radius * s

    if enabled["noise"]:
        p["noise_sigma"] = maxv.noise_sigma * s

    if enabled["jpeg"]:
        drop = int(round(maxv.jpeg_quality_drop * s))
        p["jpeg_quality"] = 95 - drop

    return p


def apply_aug(img: Image.Image, params: dict):
    out = img.convert("RGB")

    if params.get("flip_h", False):
        out = ImageOps.mirror(out)
    if params.get("flip_v", False):
        out = ImageOps.flip(out)

    angle = float(params.get("angle", 0.0))
    zoom = float(params.get("zoom", 1.0))
    tx_pct = float(params.get("tx_pct", 0.0))
    ty_pct = float(params.get("ty_pct", 0.0))
    w, h = out.size
    tx = int(round(tx_pct * w))
    ty = int(round(ty_pct * h))
    if abs(angle) > 1e-6 or abs(zoom - 1.0) > 1e-6 or tx != 0 or ty != 0:
        out = apply_affine_keep_size(out, angle_deg=angle, translate_px=(tx, ty), scale=zoom)

    if "brightness" in params:
        out = ImageEnhance.Brightness(out).enhance(float(params["brightness"]))
    if "contrast" in params:
        out = ImageEnhance.Contrast(out).enhance(float(params["contrast"]))
    if "saturation" in params:
        out = ImageEnhance.Color(out).enhance(float(params["saturation"]))

    if "hue_deg" in params and abs(float(params["hue_deg"])) > 1e-6:
        out = hue_shift_rgb(out, float(params["hue_deg"]))

    if "blur" in params:
        r = float(params["blur"])
        if r > 1e-6:
            out = out.filter(ImageFilter.GaussianBlur(radius=r))

    if "noise_sigma" in params:
        out = add_gaussian_noise(out, float(params["noise_sigma"]))

    if "jpeg_quality" in params:
        out = jpeg_roundtrip(out, int(params["jpeg_quality"]))

    return out


def resize_by_percent(img: Image.Image, pct: int) -> Image.Image:
    """Resize while keeping aspect ratio by a percentage (1-100)."""
    pct = int(clamp(pct, 1, 100))
    if pct == 100:
        return img
    w, h = img.size
    new_w = max(1, int(round(w * pct / 100.0)))
    new_h = max(1, int(round(h * pct / 100.0)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def process_single_image(args):
    """Worker function for multiprocessing"""
    img_path, src_root, dst_root, per_img, enabled, maxv, strength, resize_pct = args

    try:
        rel = img_path.relative_to(src_root)
    except Exception:
        rel = Path(img_path.name)

    out_folder = dst_root / rel.parent
    try:
        out_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {"error": f"Ordner-Fehler: {out_folder} ({e})", "done": 0}

    try:
        base = Image.open(img_path).convert("RGB")
    except Exception as e:
        return {"error": f"Skip (load error): {img_path.name} ({e})", "done": 0}

    stem = rel.stem
    results = []

    for k in range(per_img):
        params = random_params(enabled, maxv, strength)
        aug = apply_aug(base, params)
        aug = resize_by_percent(aug, resize_pct)

        sig_parts = []
        for kk in sorted(params.keys()):
            vv = params[kk]
            if isinstance(vv, bool):
                if vv:
                    sig_parts.append(kk)
            else:
                if isinstance(vv, float):
                    sig_parts.append(f"{kk}={vv:.3f}")
                else:
                    sig_parts.append(f"{kk}={vv}")
        sig = "_".join(sig_parts)[:160].replace(" ", "")

        out_name = f"{stem}__aug_{k+1:03d}__{sig}.jpg"
        out_path = out_folder / out_name

        try:
            aug.save(out_path, format="JPEG", quality=95, optimize=True)
            results.append(str(out_path))
        except Exception as e:
            return {"error": f"Save error: {out_path.name} ({e})", "done": len(results)}

    return {"done": len(results), "last_path": results[-1] if results else None}


class AugApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dataset Augmentation App (simple)")
        self.geometry("1200x720")
        self.minsize(1100, 650)

        self.msg_q = queue.Queue()
        self.worker = None
        self.stop_flag = False

        self.src_dir = tk.StringVar(value="")
        self.dst_dir = tk.StringVar(value="")
        self.aug_per_image = tk.IntVar(value=3)
        self.strength = tk.DoubleVar(value=1.0)     # 0..1
        self.resize_pct = tk.IntVar(value=100)
        self.use_gpu = tk.BooleanVar(value=CUDA_AVAILABLE)
        self.num_workers = tk.IntVar(value=min(4, os.cpu_count() or 1))
        self.subdir_options = []

        self.en = {
            "rotate": tk.BooleanVar(value=True),
            "translate": tk.BooleanVar(value=True),
            "zoom": tk.BooleanVar(value=True),
            "flip_h": tk.BooleanVar(value=True),
            "flip_v": tk.BooleanVar(value=False),
            "brightness": tk.BooleanVar(value=True),
            "contrast": tk.BooleanVar(value=True),
            "saturation": tk.BooleanVar(value=True),
            "hue": tk.BooleanVar(value=True),
            "blur": tk.BooleanVar(value=True),
            "noise": tk.BooleanVar(value=True),
            "jpeg": tk.BooleanVar(value=False),
        }

        self.maxv = AugMax()
        self.max_vars = {
            "rot_deg": tk.DoubleVar(value=self.maxv.rot_deg),
            "translate_pct": tk.DoubleVar(value=self.maxv.translate_pct),
            "zoom_pct": tk.DoubleVar(value=self.maxv.zoom_pct),
            "brightness_pct": tk.DoubleVar(value=self.maxv.brightness_pct),
            "contrast_pct": tk.DoubleVar(value=self.maxv.contrast_pct),
            "saturation_pct": tk.DoubleVar(value=self.maxv.saturation_pct),
            "hue_deg": tk.DoubleVar(value=self.maxv.hue_deg),
            "blur_radius": tk.DoubleVar(value=self.maxv.blur_radius),
            "noise_sigma": tk.DoubleVar(value=self.maxv.noise_sigma),
            "jpeg_quality_drop": tk.IntVar(value=self.maxv.jpeg_quality_drop),
        }

        self.status = tk.StringVar(value="Bereit.")
        self.progress = tk.DoubleVar(value=0.0)
        self.image_count_text = tk.StringVar(value="Quellbilder: 0 → Resultierende Bilder: 0")

        # Border mode info
        border_mode = "REFLECT (OpenCV)" if CV2_AVAILABLE else "Edge-Averaging (PIL)"
        self.border_info = tk.StringVar(value=f"Füllmethode: {border_mode}")

        self._build_ui()

        self.tk_orig = None
        self.tk_aug = None

        self.after(80, self._poll_queue)

    def populate_subdirs(self):
        """Populate subdir list based on current source directory."""
        src = Path(self.src_dir.get().strip())
        if not self.src_dir.get().strip():
            return
        self.subdir_options = []
        self.subdir_list.delete(0, tk.END)
        if not src.exists():
            return
        dirs = subdirs_with_images(src)
        for d in dirs:
            label = "." if d in ("", ".") else d
            self.subdir_options.append(d)
            self.subdir_list.insert(tk.END, label)
        if dirs:
            self.subdir_list.select_set(0, tk.END)
        self.update_image_count()
        self.refresh_preview()

    def _select_all_subdirs(self, select_all: bool):
        if select_all:
            self.subdir_list.select_set(0, tk.END)
        else:
            self.subdir_list.selection_clear(0, tk.END)
        self.update_image_count()
        self.refresh_preview()

    def _selected_subdirs(self):
        if not self.subdir_options:
            return None
        sel = self.subdir_list.curselection()
        if not sel:
            return set()
        return {self.subdir_options[i] for i in sel}

    def _on_subdir_change(self):
        self.update_image_count()
        self.refresh_preview()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        def browse_src():
            p = filedialog.askdirectory(title="Quellverzeichnis wählen")
            if p:
                self.src_dir.set(p)
                self.populate_subdirs()
                self.update_image_count()
                self.refresh_preview()

        def browse_dst():
            p = filedialog.askdirectory(title="Zielverzeichnis wählen")
            if p:
                self.dst_dir.set(p)

        row1 = ttk.Frame(top)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Quelle:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.src_dir, width=70).pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="Browse", command=browse_src).pack(side=tk.LEFT)

        row2 = ttk.Frame(top)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Ziel:   ").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.dst_dir, width=70).pack(side=tk.LEFT, padx=6)
        ttk.Button(row2, text="Browse", command=browse_dst).pack(side=tk.LEFT)

        row_sub = ttk.Frame(top)
        row_sub.pack(fill=tk.X, pady=6)
        ttk.Label(row_sub, text="Unterordner (mit Bildern) ausw„hlen:").pack(side=tk.TOP, anchor="w")
        sub_frame = ttk.Frame(row_sub)
        sub_frame.pack(fill=tk.X, pady=2)

        self.subdir_list = tk.Listbox(sub_frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        self.subdir_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb = ttk.Scrollbar(sub_frame, orient=tk.VERTICAL, command=self.subdir_list.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.subdir_list.config(yscrollcommand=sb.set)
        self.subdir_list.bind("<<ListboxSelect>>", lambda _=None: self._on_subdir_change())

        btns_sub = ttk.Frame(row_sub)
        btns_sub.pack(side=tk.RIGHT, anchor="n", padx=6)
        ttk.Button(btns_sub, text="Alle", command=lambda: self._select_all_subdirs(True)).pack(fill=tk.X, pady=1)
        ttk.Button(btns_sub, text="Keine", command=lambda: self._select_all_subdirs(False)).pack(fill=tk.X, pady=1)
        ttk.Button(btns_sub, text="Aktualisieren", command=self.populate_subdirs).pack(fill=tk.X, pady=1)

        row3 = ttk.Frame(top)
        row3.pack(fill=tk.X, pady=6)

        ttk.Label(row3, text="Augmentierungen pro Originalbild:").pack(side=tk.LEFT)
        spin = ttk.Spinbox(row3, from_=1, to=10000, textvariable=self.aug_per_image, width=8, command=self.update_image_count)
        spin.pack(side=tk.LEFT, padx=8)
        spin.bind("<KeyRelease>", lambda _: self.update_image_count())

        ttk.Label(row3, textvariable=self.image_count_text, foreground="blue").pack(side=tk.LEFT, padx=8)

        # GPU/CPU options
        if CUDA_AVAILABLE:
            ttk.Checkbutton(row3, text="GPU nutzen", variable=self.use_gpu).pack(side=tk.LEFT, padx=(20, 4))
        elif TORCH_AVAILABLE:
            ttk.Label(row3, text="(GPU nicht verfügbar)", foreground="gray").pack(side=tk.LEFT, padx=(20, 4))

        ttk.Label(row3, text="CPU-Worker:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Spinbox(row3, from_=1, to=16, textvariable=self.num_workers, width=4).pack(side=tk.LEFT)

        ttk.Label(row3, text="Strength:").pack(side=tk.LEFT, padx=(20, 6))
        s = ttk.Scale(row3, from_=0.0, to=1.0, variable=self.strength,
                      command=lambda _=None: self.refresh_preview())
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(row3, text="Output-Aufloesung (%):").pack(side=tk.LEFT, padx=(12, 4))
        resize_choices = [str(p) for p in (100, 90, 80, 70, 60, 50, 40, 30, 20, 10)]
        resize_cb = ttk.Combobox(row3, values=resize_choices, width=5, textvariable=self.resize_pct, state="readonly")
        resize_cb.pack(side=tk.LEFT)
        resize_cb.set(str(self.resize_pct.get()))
        resize_cb.bind("<<ComboboxSelected>>", lambda _=None: self.refresh_preview())

        btns = ttk.Frame(row3)
        btns.pack(side=tk.RIGHT)
        ttk.Button(btns, text="Vorschau neu", command=self.refresh_preview).pack(side=tk.LEFT, padx=6)
        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

        mid = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(mid, padding=10)
        right = ttk.Frame(mid, padding=10)
        mid.add(left, weight=1)
        mid.add(right, weight=2)

        ttk.Label(left, text="Augmentationsmethoden (auswählen):", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        mgrid = ttk.Frame(left)
        mgrid.pack(fill=tk.X, pady=(6, 10))

        def mk_cb(text, key, r, c):
            cb = ttk.Checkbutton(mgrid, text=text, variable=self.en[key], command=self.refresh_preview)
            cb.grid(row=r, column=c, sticky="w", padx=6, pady=2)

        mk_cb("Rotation", "rotate", 0, 0)
        mk_cb("Translate", "translate", 0, 1)
        mk_cb("Zoom", "zoom", 0, 2)
        mk_cb("Flip H", "flip_h", 1, 0)
        mk_cb("Flip V", "flip_v", 1, 1)
        mk_cb("Brightness", "brightness", 2, 0)
        mk_cb("Contrast", "contrast", 2, 1)
        mk_cb("Saturation", "saturation", 2, 2)
        mk_cb("Hue shift", "hue", 3, 0)
        mk_cb("Blur", "blur", 3, 1)
        mk_cb("Noise", "noise", 3, 2)
        mk_cb("JPEG degrade", "jpeg", 4, 0)

        ttk.Label(left, text="Maximalwerte (werden mit Strength skaliert):", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        form = ttk.Frame(left)
        form.pack(fill=tk.X, pady=(6, 0))

        def add_row(label, var, row, unit=""):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
            e = ttk.Entry(form, textvariable=var, width=10)
            e.grid(row=row, column=1, sticky="w", padx=6)
            if unit:
                ttk.Label(form, text=unit).grid(row=row, column=2, sticky="w")
            e.bind("<KeyRelease>", lambda _e=None: self.refresh_preview())
            e.bind("<FocusOut>", lambda _e=None: self.refresh_preview())

        add_row("Rotation ±", self.max_vars["rot_deg"], 0, "deg")
        add_row("Translate ±", self.max_vars["translate_pct"], 1, "%")
        add_row("Zoom ±", self.max_vars["zoom_pct"], 2, "%")
        add_row("Brightness ±", self.max_vars["brightness_pct"], 3, "%")
        add_row("Contrast ±", self.max_vars["contrast_pct"], 4, "%")
        add_row("Saturation ±", self.max_vars["saturation_pct"], 5, "%")
        add_row("Hue shift ±", self.max_vars["hue_deg"], 6, "deg")
        add_row("Blur max", self.max_vars["blur_radius"], 7, "radius")
        add_row("Noise max", self.max_vars["noise_sigma"], 8, "sigma")
        add_row("JPEG drop", self.max_vars["jpeg_quality_drop"], 9, "quality points")

        ttk.Label(right, text="Vorschau (random Quelle / Augmentiert = Maximalwerte):",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        pv = ttk.Frame(right)
        pv.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.canvas_orig = tk.Label(pv, text="(kein Bild)", relief=tk.GROOVE, anchor="center")
        self.canvas_aug = tk.Label(pv, text="(kein Bild)", relief=tk.GROOVE, anchor="center")
        self.canvas_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.canvas_aug.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.pb = ttk.Progressbar(bottom, variable=self.progress, maximum=100.0)
        self.pb.pack(side=tk.TOP, fill=tk.X, pady=6)
        ttk.Label(bottom, textvariable=self.status).pack(side=tk.LEFT)
        ttk.Label(bottom, textvariable=self.border_info, foreground="darkgreen" if CV2_AVAILABLE else "orange").pack(side=tk.RIGHT, padx=10)

    def _read_max(self) -> AugMax:
        def f(name, default, min_val=0.0):
            try:
                val = float(self.max_vars[name].get())
                return max(min_val, val)
            except Exception:
                return default

        def i(name, default, min_val=0):
            try:
                val = int(self.max_vars[name].get())
                return max(min_val, val)
            except Exception:
                return default

        return AugMax(
            rot_deg=f("rot_deg", 5.0, 0.0),
            translate_pct=f("translate_pct", 2.0, 0.0),
            zoom_pct=f("zoom_pct", 3.0, 0.0),
            brightness_pct=f("brightness_pct", 8.0, 0.0),
            contrast_pct=f("contrast_pct", 8.0, 0.0),
            saturation_pct=f("saturation_pct", 10.0, 0.0),
            hue_deg=f("hue_deg", 6.0, 0.0),
            blur_radius=f("blur_radius", 0.6, 0.0),
            noise_sigma=f("noise_sigma", 6.0, 0.0),
            jpeg_quality_drop=i("jpeg_quality_drop", 25, 0),
        )

    def _enabled_dict(self):
        return {k: bool(v.get()) for k, v in self.en.items()}

    def update_image_count(self):
        src = Path(self.src_dir.get().strip())
        imgs = list_images_dedup(src)
        allowed = self._selected_subdirs()
        imgs = filter_images_by_subdirs(imgs, src, allowed)
        num_imgs = len(imgs)
        try:
            per_img = int(self.aug_per_image.get())
        except Exception:
            per_img = 1
        total = num_imgs * per_img
        self.image_count_text.set(f"Quellbilder: {num_imgs} → Resultierende Bilder: {total}")

    def refresh_preview(self):
        src = Path(self.src_dir.get().strip())
        imgs = list_images_dedup(src)
        allowed = self._selected_subdirs()
        imgs = filter_images_by_subdirs(imgs, src, allowed)
        if not imgs:
            self.canvas_orig.config(image="", text="(kein Bild im Quellverzeichnis)")
            self.canvas_aug.config(image="", text="(kein Bild)")
            self.tk_orig = None
            self.tk_aug = None
            if allowed == set():
                self.status.set("Keine Unterordner ausgew„hlt.")
            return

        p = random.choice(imgs)
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            self.status.set(f"Preview: Fehler beim Laden: {p.name} ({e})")
            return

        enabled = self._enabled_dict()
        maxv = self._read_max()
        strength = float(self.strength.get())
        try:
            resize_pct = int(self.resize_pct.get())
        except Exception:
            resize_pct = 100

        params = max_params(enabled, maxv, strength)
        aug = apply_aug(img, params)
        aug = resize_by_percent(aug, resize_pct)

        self.update_idletasks()
        ow = max(200, self.canvas_orig.winfo_width() - 10)
        oh = max(200, self.canvas_orig.winfo_height() - 10)
        aw = max(200, self.canvas_aug.winfo_width() - 10)
        ah = max(200, self.canvas_aug.winfo_height() - 10)

        img_fit = pil_to_fit(img, ow, oh)
        aug_fit = pil_to_fit(aug, aw, ah)

        self.tk_orig = ImageTk.PhotoImage(img_fit)
        self.tk_aug = ImageTk.PhotoImage(aug_fit)

        self.canvas_orig.config(image=self.tk_orig, text="")
        self.canvas_aug.config(image=self.tk_aug, text="")
        self.status.set(f"Vorschau: {p.name}")

    def start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Läuft", "Augmentierung läuft bereits.")
            return

        src = Path(self.src_dir.get().strip())
        dst = Path(self.dst_dir.get().strip())
        if not src.exists():
            messagebox.showerror("Fehler", "Quellverzeichnis ist ungültig.")
            return
        if not dst.exists():
            messagebox.showerror("Fehler", "Zielverzeichnis ist ungültig.")
            return

        imgs = list_images_dedup(src)
        allowed_subdirs = self._selected_subdirs()
        if allowed_subdirs == set() and self.subdir_options:
            messagebox.showerror("Fehler", "Bitte mindestens einen Unterordner ausw„hlen.")
            return
        imgs = filter_images_by_subdirs(imgs, src, allowed_subdirs)
        if not imgs:
            messagebox.showerror("Fehler", "Im Quellverzeichnis wurden keine Bilder (in den gew„hlten Ordnern) gefunden.")
            return

        per_img = int(self.aug_per_image.get())
        if per_img <= 0:
            messagebox.showerror("Fehler", "Augmentierungen pro Bild müssen > 0 sein.")
            return

        enabled = self._enabled_dict()
        if not any(enabled.values()):
            messagebox.showerror("Fehler", "Bitte mindestens eine Augmentationsmethode auswählen.")
            return

        maxv = self._read_max()
        strength = float(self.strength.get())
        use_gpu = self.use_gpu.get() if CUDA_AVAILABLE else False
        num_workers = int(self.num_workers.get())
        try:
            resize_pct = int(self.resize_pct.get())
        except Exception:
            resize_pct = 100
        resize_pct = int(clamp(resize_pct, 10, 100))

        total = len(imgs) * per_img
        self.progress.set(0.0)
        mode_str = "GPU" if use_gpu else f"{num_workers} CPU-Worker"
        self.status.set(f"Starte ({mode_str}) ... ({len(imgs)} Bilder, je {per_img} Augs => {total} Outputs)")

        self.stop_flag = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        args = (src, dst, imgs, per_img, enabled, maxv, strength, use_gpu, num_workers, resize_pct, allowed_subdirs)
        self.worker = threading.Thread(target=self._worker, args=args, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_flag = True
        self.status.set("Stoppe nach aktuellem Bild...")
        self.stop_btn.config(state=tk.DISABLED)

    def _worker(self, src_root: Path, dst_root: Path, imgs, per_img, enabled, maxv: AugMax, strength: float, use_gpu: bool, num_workers: int, resize_pct: int, _allowed_subdirs=None):
        t0 = time.time()
        total = len(imgs) * per_img
        done = 0

        if use_gpu and CUDA_AVAILABLE:
            # GPU processing (currently same as CPU, but could be optimized with batching)
            self.msg_q.put(("status", "GPU-Modus: Funktioniert aktuell wie CPU (Batching nicht implementiert)"))
            use_gpu = False  # Fallback to CPU for now

        # Multiprocessing
        if num_workers > 1:
            tasks = [(img_path, src_root, dst_root, per_img, enabled, maxv, strength, resize_pct) for img_path in imgs]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_single_image, task): task for task in tasks}

                for future in as_completed(futures):
                    if self.stop_flag:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    try:
                        result = future.result()
                        if "error" in result:
                            self.msg_q.put(("status", result["error"]))

                        done += result.get("done", 0)

                        if done > 0 and (done % max(1, total // 100) == 0 or done == total):
                            pct = done / total * 100.0
                            self.msg_q.put(("progress", pct))
                            last = result.get("last_path", "")
                            self.msg_q.put(("status", f"{done}/{total} gespeichert: {last}"))
                    except Exception as e:
                        self.msg_q.put(("status", f"Worker error: {e}"))

        else:
            # Single-threaded processing
            for img_path in imgs:
                if self.stop_flag:
                    break

                result = process_single_image((img_path, src_root, dst_root, per_img, enabled, maxv, strength, resize_pct))

                if "error" in result:
                    self.msg_q.put(("status", result["error"]))

                done += result.get("done", 0)

                if done > 0 and (done % max(1, total // 100) == 0 or done == total):
                    pct = done / total * 100.0
                    self.msg_q.put(("progress", pct))
                    last = result.get("last_path", "")
                    self.msg_q.put(("status", f"{done}/{total} gespeichert: {last}"))

        dt = time.time() - t0
        status_msg = "Abgebrochen" if self.stop_flag else "Fertig"
        speed = done / dt if dt > 0 else 0
        self.msg_q.put(("progress", 100.0))
        self.msg_q.put(("done", f"{status_msg}. Outputs: {done}/{total}. Dauer: {dt:.1f}s ({speed:.1f} Bilder/s)\nOutput Root: {dst_root}"))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind == "progress":
                    self.progress.set(float(payload))
                elif kind == "status":
                    self.status.set(str(payload))
                elif kind == "done":
                    self.status.set("Fertig.")
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    messagebox.showinfo("Fertig", str(payload))
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)


if __name__ == "__main__":
    app = AugApp()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    app.mainloop()
