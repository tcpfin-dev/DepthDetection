#!/usr/bin/env python3
"""Inference smoke test for YOLOv8 object detection.

Loads an Ultralytics YOLOv8 model (COCO-pretrained by default) and runs object
detection on a single image. Prints every detection (class, confidence, box)
and saves an annotated result image.

Usage
-----
    python test/test_yolo.py --image photo.jpg
    python test/test_yolo.py --image photo.jpg --model yolov8s.pt --conf 0.4

If ``--image`` is omitted, a synthetic image is generated so the script also
doubles as a dependency / environment smoke test (expect zero detections).

Install
-------
    pip install ultralytics        # pulls torch + torchvision automatically

Note
----
The first run downloads the weights (e.g. ``yolov8n.pt``) from the Ultralytics
assets release and caches them in the current working directory.
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
    from ultralytics import YOLO
except ImportError as exc:  # environment guard
    sys.exit(
        f"Missing dependency '{getattr(exc, 'name', exc)}'.\n"
        "Install with: pip install ultralytics"
    )


DEFAULT_MODEL = "yolov8n.pt"
_SCRIPT_DIR = Path(__file__).resolve().parent


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
        help="Ultralytics weights or .yaml, e.g. yolov8n.pt / yolov8s.pt "
        f"(default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640).")
    parser.add_argument(
        "--out",
        type=Path,
        default=_SCRIPT_DIR / "outputs",
        help="Directory for the annotated result (default: test/outputs).",
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


def main(argv=None):
    args = parse_args(argv)
    device = 0 if torch.cuda.is_available() else "cpu"

    image, src_path = load_image(args.image, args.size)
    args.out.mkdir(parents=True, exist_ok=True)
    if src_path is None:
        print("[warn] --image not given; using a synthetic image (smoke test only).")
        src_path = args.out / "synthetic_input.png"
        image.save(src_path)
        print(f"[info] synthetic input written to {src_path}")

    print(f"[info] model  : {args.model}")
    print(f"[info] device : {device}")
    print(f"[info] image  : {src_path}  ({image.width}x{image.height})")

    t0 = time.perf_counter()
    model = YOLO(args.model)
    print(f"[info] loaded in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    results = model.predict(
        source=image,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
    )
    print(f"[info] inference: {time.perf_counter() - t0:.3f}s")

    base = Path(src_path).stem
    total = 0
    for index, result in enumerate(results):
        names = result.names
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        total += count
        print(f"[info] result {index}: {count} detection(s)")
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                print(
                    f"       {label:<12} conf={conf:.3f} "
                    f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
                )
        out_png = args.out / f"{base}_yolo{'' if len(results) == 1 else f'_{index}'}.png"
        result.save(filename=str(out_png))
        print(f"[info] wrote    : {out_png}")

    print(f"[info] total detections: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
