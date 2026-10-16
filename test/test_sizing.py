#!/usr/bin/env python3
"""Estimate an object's distance and real-world size from a single RGB image.

Combines the two models already used in this folder:

    Depth Anything V2  ->  relative depth map   D
    YOLOv8             ->  bounding box         (u1, v1, u2, v2)

and applies the geometric conversion from ``Research_Proposals.md`` (Part 1):

    Z = alpha * D + beta          depth -> meters
    W = |u2 - u1| * Z / f_x       metric width
    H = |v2 - v1| * Z / f_y       metric height

The depth model only produces *relative* depth, so ``alpha``/``beta`` must be
pinned to reality with tape-measured distances:

    --distance 0.5          one measured distance to the object (assumes beta=0)
    --anchor U,V,DIST       repeat two or more times for a full affine fit,
                            which is what lets the model predict Z on its own

Usage
-----
    python test/test_sizing.py --image bottle.jpg --target bottle --distance 0.5
    python test/test_sizing.py --image bottle.jpg --target bottle \
        --anchor 1900,1500,0.5 --anchor 3400,1500,1.2 --fx 3200

If ``--image`` is omitted a synthetic image is generated, so the script also
doubles as an environment smoke test (expect meaningless numbers).

Install
-------
    pip install torch transformers pillow numpy ultralytics
    # Headless hosts: use opencv-python-headless instead of opencv-python.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from transformers import pipeline
    from ultralytics import YOLO
except ImportError as exc:  # environment guard
    sys.exit(
        f"Missing dependency '{getattr(exc, 'name', exc)}'.\n"
        "Install with: pip install torch transformers pillow numpy ultralytics"
    )


# Transformers-compatible Depth Anything V2 checkpoints ("-hf" suffix required).
MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
DEFAULT_DEPTH_MODEL = MODELS["small"]
DEFAULT_YOLO = "yolov8n.pt"
_SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=Path, default=None, help="Input image (synthetic if omitted).")
    parser.add_argument("--target", default=None, help="COCO class to size, e.g. bottle (default: largest box).")
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL, help="small/base/large or a HF id.")
    parser.add_argument("--yolo", default=DEFAULT_YOLO, help=f"Ultralytics weights (default: {DEFAULT_YOLO}).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25).")
    parser.add_argument("--distance", type=float, default=None,
                        help="Measured camera-to-object distance in meters (single-anchor calibration).")
    parser.add_argument("--anchor", action="append", default=[], metavar="U,V,DIST",
                        help="Ground-truth anchor as 'u,v,distance_m'. Repeat for an affine fit.")
    parser.add_argument("--depth-sample", choices=["box_median", "box_center", "bottom_center"],
                        default="box_median", help="Where to read the object's depth from its box.")
    parser.add_argument("--fx", type=float, default=None, help="Focal length in pixels (x).")
    parser.add_argument("--fy", type=float, default=None, help="Focal length in pixels (y).")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x (default: image center).")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y (default: image center).")
    parser.add_argument("--hfov", type=float, default=65.0,
                        help="Assumed horizontal FOV (deg) when --fx/--fy omitted (default: 65).")
    parser.add_argument("--out", type=Path, default=_SCRIPT_DIR / "outputs", help="Output directory.")
    parser.add_argument("--size", type=int, default=512, help="Synthetic image side length.")
    return parser.parse_args(argv)


def make_synthetic_image(size):
    """Deterministic RGB image with a few shapes (env smoke test only)."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([size * 0.15, size * 0.20, size * 0.50, size * 0.70], fill=(230, 60, 60))
    draw.ellipse([size * 0.55, size * 0.40, size * 0.85, size * 0.80], fill=(60, 120, 230))
    draw.rectangle([size * 0.30, size * 0.75, size * 0.80, size * 0.90], fill=(90, 200, 90))
    return img


def load_image(path, size):
    if path is None:
        return make_synthetic_image(size), None
    path = Path(path)
    if not path.is_file():
        sys.exit(f"Input image not found: {path}")
    return Image.open(path).convert("RGB"), path


def sample_depth(depth, u, v, radius=2):
    """Median relative depth in a small window (robust to stray pixels)."""
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    return float(np.median(depth[y0:y1, x0:x1]))


def object_depth(depth, box, mode):
    x1, y1, x2, y2 = box
    if mode == "box_median":
        bw, bh = x2 - x1, y2 - y1
        xa, xb = int(x1 + 0.2 * bw), int(x2 - 0.2 * bw)
        ya, yb = int(y1 + 0.2 * bh), int(y2 - 0.2 * bh)
        return float(np.median(depth[ya:yb, xa:xb]))
    if mode == "box_center":
        return sample_depth(depth, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return sample_depth(depth, (x1 + x2) / 2.0, y2 - 0.05 * (y2 - y1))


def resolve_intrinsics(args, width, height):
    if args.fx and args.fy:
        return args.fx, args.fy, (args.cx if args.cx is not None else width / 2.0), (
            args.cy if args.cy is not None else height / 2.0
        )
    import math

    fx = 0.5 * width / math.tan(math.radians(args.hfov) / 2.0)
    print(f"[warn] --fx/--fy not given; assuming a {args.hfov:g} deg horizontal FOV (f={fx:.0f} px).")
    return fx, fx, width / 2.0, height / 2.0


def run_depth(image, model_id):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(task="depth-estimation", model=MODELS.get(model_id, model_id), device=device)
    result = pipe(image)
    predicted = result["predicted_depth"]
    if hasattr(predicted, "detach"):
        predicted = predicted.detach().cpu().numpy()
    return np.asarray(predicted, dtype=np.float32).squeeze()


def run_yolo(image, weights, conf, target):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)
    detections = []
    for box in model.predict(source=image, conf=conf, device=device, verbose=False)[0].boxes:
        class_id = int(box.cls.item())
        label = model.names[class_id]
        detections.append((label, float(box.conf.item()), tuple(float(v) for v in box.xyxy[0].tolist())))
    if not detections:
        sys.exit("No objects detected; try a lower --conf or a different --target.")
    if target:
        matches = [d for d in detections if d[0].lower() == target.lower()]
        if not matches:
            found = ", ".join(sorted({d[0] for d in detections}))
            sys.exit(f"No '{target}' detected. Detected: {found}")
        return max(matches, key=lambda d: d[1])
    return max(detections, key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]))


def fit_calibration(args, depth, object_depth_value):
    """Return (alpha, beta, description)."""
    if args.anchor:
        points = []
        for spec in args.anchor:
            parts = spec.split(",")
            if len(parts) != 3:
                sys.exit(f"--anchor expects 'u,v,distance_m', got '{spec}'")
            u, v, dist = (float(p) for p in parts)
            points.append((sample_depth(depth, u, v), dist))
        depths = [p[0] for p in points]
        distances = [p[1] for p in points]
        if len(points) >= 2:
            alpha, beta = np.polyfit(depths, distances, 1)
            return float(alpha), float(beta), f"affine fit on {len(points)} anchors"
        alpha = float(np.dot(depths, distances) / np.dot(depths, depths))
        return alpha, 0.0, "scale fit on 1 anchor (beta=0)"
    if args.distance is not None:
        return args.distance / object_depth_value, 0.0, "--distance (beta=0)"
    sys.exit("No calibration: pass --distance <m> or two or more --anchor u,v,distance_m.")


def annotate(image, box, lines):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = box
    width = max(2, canvas.size[1] // 400)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=width)
    try:
        font = ImageFont.load_default(size=max(16, canvas.size[1] // 45))
    except TypeError:
        font = ImageFont.load_default()
    text = "\n".join(lines)
    draw.multiline_text((x1 + 4, max(0, y1 - 4 * len(lines) * 8)), text, fill=(255, 255, 255), font=font,
                        stroke_width=2, stroke_fill=(0, 0, 0))
    return canvas


def main(argv=None):
    args = parse_args(argv)
    image, src_path = load_image(args.image, args.size)
    args.out.mkdir(parents=True, exist_ok=True)
    if src_path is None:
        print("[warn] --image not given; using a synthetic image (smoke test only).")
        src_path = args.out / "synthetic_input.png"
        image.save(src_path)

    fx, fy, cx, cy = resolve_intrinsics(args, image.width, image.height)
    print(f"[info] image     : {src_path}  ({image.width}x{image.height})")
    print(f"[info] camera    : fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    t0 = time.perf_counter()
    depth = run_depth(image, args.depth_model)
    print(f"[info] depth     : {depth.shape} in {time.perf_counter() - t0:.2f}s "
          f"range=[{depth.min():.3f}, {depth.max():.3f}]")

    label, conf, box = run_yolo(image, args.yolo, args.conf, args.target)
    print(f"[info] detection : {label} {conf:.3f} box={tuple(round(v, 1) for v in box)}")

    z_depth = object_depth(depth, box, args.depth_sample)
    alpha, beta, how = fit_calibration(args, depth, z_depth)
    z = alpha * z_depth + beta
    if z <= 0:
        sys.exit(f"Non-positive distance Z={z:.3f} m; check the anchors/distances.")
    print(f"[info] calibration: alpha={alpha:.4f} beta={beta:.4f} ({how})")

    width_m = abs(box[2] - box[0]) * z / fx
    height_m = abs(box[3] - box[1]) * z / fy

    print(f"[info] object    : {label}")
    print(f"[info] distance  : Z = {z:.3f} m ({z * 100:.1f} cm)")
    print(f"[info] width     : W = {width_m:.4f} m ({width_m * 100:.1f} cm)")
    print(f"[info] height    : H = {height_m:.4f} m ({height_m * 100:.1f} cm)")
    print(f"[info] area      : W*H = {width_m * height_m:.4f} m^2")

    lines = [f"{label} {conf:.2f}", f"Z={z * 100:.1f}cm",
             f"W={width_m * 100:.1f} H={height_m * 100:.1f}cm"]
    out_png = args.out / f"{Path(src_path).stem}_size.png"
    annotate(image, box, lines).save(out_png)
    print(f"[info] wrote     : {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
