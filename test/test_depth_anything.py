#!/usr/bin/env python3
"""Inference smoke test for Depth Anything V2.

Runs zero-shot monocular depth estimation (MDE) on a single image with the
Hugging Face Transformers ``depth-estimation`` pipeline, then reports the raw
relative-depth statistics and writes colorized visualizations.

The model is used exactly as-is (zero-shot, no fine-tuning), matching the
project goal of benchmarking foundation models rather than training new ones.

Usage
-----
    python test/test_depth_anything.py --image photo.jpg
    python test/test_depth_anything.py --image photo.jpg \
        --model depth-anything/Depth-Anything-V2-Base-hf --out test/outputs

If ``--image`` is omitted, a synthetic image is generated so the script also
doubles as a dependency / environment smoke test (no meaningful depth values
are expected on noise).

Install
-------
    pip install torch transformers pillow numpy

For a GPU build, install torch from the CUDA index first, e.g.::

    pip install torch --index-url https://download.pytorch.org/whl/cu124
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from transformers import pipeline
except ImportError as exc:  # environment guard
    sys.exit(
        f"Missing dependency '{getattr(exc, 'name', exc)}'.\n"
        "Install with: pip install torch transformers pillow numpy"
    )


# Transformers-compatible Depth Anything V2 checkpoints. The "-hf" suffix is
# required by the Transformers pipeline; the original (non "-hf") repos ship
# custom code and are not directly loadable.
MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
DEFAULT_MODEL = MODELS["small"]

_SCRIPT_DIR = Path(__file__).resolve().parent

# Approximate "magma" control points for a dependency-free color map.
_MAGMA = np.array(
    [
        [0, 0, 4],
        [28, 16, 68],
        [79, 18, 123],
        [129, 37, 129],
        [181, 54, 122],
        [229, 80, 100],
        [251, 135, 97],
        [254, 194, 135],
        [252, 253, 191],
    ],
    dtype=np.float32,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to the input image. A synthetic image is used if omitted.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id, or one of: small, base, large "
        f"(default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_SCRIPT_DIR / "outputs",
        help="Directory for output visualizations (default: test/outputs).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Side length of the synthetic image when --image is omitted (default: 512).",
    )
    return parser.parse_args(argv)


def make_synthetic_image(size):
    """Build a deterministic RGB image with a few shapes on texture."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([size * 0.15, size * 0.20, size * 0.50, size * 0.70], fill=(230, 60, 60))
    draw.ellipse([size * 0.55, size * 0.40, size * 0.85, size * 0.80], fill=(60, 120, 230))
    draw.rectangle([size * 0.30, size * 0.75, size * 0.80, size * 0.90], fill=(90, 200, 90))
    return img


def load_image(path, size):
    """Return ``(PIL.Image RGB, source_path_or_None)``."""
    if path is None:
        return make_synthetic_image(size), None
    path = Path(path)
    if not path.is_file():
        sys.exit(f"Input image not found: {path}")
    return Image.open(path).convert("RGB"), path


def apply_colormap(norm01):
    """Map a [0, 1] float array to an RGB uint8 image via linear interpolation."""
    x = np.clip(norm01, 0.0, 1.0) * (len(_MAGMA) - 1)
    lo = np.floor(x).astype(np.int32)
    hi = np.minimum(lo + 1, len(_MAGMA) - 1)
    frac = (x - lo)[..., None]
    rgb = _MAGMA[lo] * (1.0 - frac) + _MAGMA[hi] * frac
    return rgb.astype(np.uint8)


def to_uint8(norm01):
    return (np.clip(norm01, 0.0, 1.0) * 255.0).astype(np.uint8)


def main(argv=None):
    args = parse_args(argv)
    model_id = MODELS.get(args.model, args.model)
    device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if device == 0 else "cpu"

    image, src_path = load_image(args.image, args.size)
    args.out.mkdir(parents=True, exist_ok=True)
    if src_path is None:
        print("[warn] --image not given; using a synthetic image (smoke test only).")
        src_path = args.out / "synthetic_input.png"
        image.save(src_path)
        print(f"[info] synthetic input written to {src_path}")

    print(f"[info] model  : {model_id}")
    print(f"[info] device : {device_name}")
    print(f"[info] image  : {src_path}  ({image.width}x{image.height})")

    t0 = time.perf_counter()
    pipe = pipeline(task="depth-estimation", model=model_id, device=device)
    print(f"[info] loaded in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    result = pipe(image)
    elapsed = time.perf_counter() - t0

    predicted = result["predicted_depth"]
    depth = predicted.squeeze().detach().cpu().numpy().astype(np.float32)

    print(f"[info] inference: {elapsed:.3f}s")
    print(f"[info] depth map: shape={depth.shape} dtype={depth.dtype}")
    print(
        "[info] values   : "
        f"min={depth.min():.4f} max={depth.max():.4f} "
        f"mean={depth.mean():.4f} median={float(np.median(depth)):.4f}"
    )

    lo, hi = float(depth.min()), float(depth.max())
    norm = (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)

    stem = Path(src_path).stem
    gray_path = args.out / f"{stem}_depth_gray.png"
    color_path = args.out / f"{stem}_depth_color.png"
    Image.fromarray(to_uint8(norm), "L").save(gray_path)
    Image.fromarray(apply_colormap(norm), "RGB").save(color_path)
    print(f"[info] wrote    : {gray_path}")
    print(f"[info] wrote    : {color_path}")

    pipe_depth = result.get("depth")
    if isinstance(pipe_depth, Image.Image):
        pipe_path = args.out / f"{stem}_depth_pipeline.png"
        pipe_depth.save(pipe_path)
        print(f"[info] wrote    : {pipe_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
