#!/usr/bin/env python3
"""Combined zero-shot metric-sizing pipeline (single command).

This is the joined version of the two smoke scripts plus the geometric sizing
logic. One command runs the whole pipeline from ``Research_Proposals.md``:

    image
      -> depth model         (zero-shot relative depth map D; default Depth
                              Anything V2, MiDaS via --depth-model midas)
      -> YOLOv8              (bounding boxes, one per object)
      -> metric calibration  (Z = alpha * D + beta)
      -> metric distance / width / height / area for every detected object

Every detection (or just ``--target`` class) is sized, and the results are
printed as a table plus written to ``_results.json`` / ``_results.csv`` along
with an annotated overlay and depth visualizations.

The depth model only produces *relative* depth, so one of these metric inputs
is required to pin it to reality:

    --distance 0.5            measured camera-to-object distance (m) of the
                              most confident target; single-anchor scale (beta=0)
    --anchor U,V,DIST         repeat two or more times for a full affine
                              least-squares fit; the model then predicts Z for
                              every object on its own
    --object-height 0.20      known real height (m) of the --target class; the
                              distance is recovered from the box height and the
                              focal length (no depth anchor needed)

``--calib`` chooses how the anchors are turned into meters. ``affine`` (default)
fits ``Z = alpha*D + beta`` as in the proposal. ``inverse`` fits the disparity
``1/Z = alpha*D + beta``, which matches the roughly inverse-depth output of
monocular depth models and interpolates far more accurately between anchors:

    python test_pipeline.py --image image1.png --target bottle --calib inverse \
        --anchor 1463,935,0.2 --anchor 1229,308,1.0

Camera intrinsics come from ``--fx/--fy`` if given, otherwise from EXIF
(``FocalLengthIn35mmFilm``, matching the proposal's "student setup shortcut"),
otherwise an assumed ``--hfov`` (65 deg) is used.

Usage
-----
    python test_pipeline.py --image photo.jpg --target bottle --distance 0.5
    python test_pipeline.py --image photo.jpg --target bottle \
        --anchor 1900,1500,0.5 --anchor 3400,1500,1.2 --fx 3200
    python test_pipeline.py --image photo.jpg --target bottle --object-height 0.20
    python test_pipeline.py --image photo.jpg --target bottle --depth-model midas \
        --object-height 0.20

If ``--image`` is omitted a synthetic image is generated so the script also
doubles as an environment/dependency smoke test (a calibration flag is still
required; expect meaningless numbers).

Install
-------
    pip install torch transformers pillow numpy ultralytics
    # Headless hosts: use opencv-python-headless instead of opencv-python.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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


# Transformers-compatible depth checkpoints. Depth Anything V2 "-hf" repos are
# the default; MiDaS (DPT) is the classic control baseline to compare against.
MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    "midas": "Intel/dpt-hybrid-midas",
    "midas-small": "Intel/dpt-swinv2-tiny-256",
}
DEFAULT_DEPTH_MODEL = MODELS["small"]
DEFAULT_YOLO = "yolov8n.pt"
_SCRIPT_DIR = Path(__file__).resolve().parent

# EXIF tag ids used for the focal-length shortcut.
_EXIF_FOCAL_35MM = 41989  # FocalLengthIn35mmFilm
_FULLFRAME_WIDTH_MM = 36.0

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
    parser.add_argument("--image", type=Path, default=None, help="Input image (synthetic if omitted).")
    parser.add_argument("--target", default=None, help="COCO class to size, e.g. bottle (default: all detections).")
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL,
                        help="small/base/large (Depth Anything V2, default) | midas | midas-small | a HF id.")
    parser.add_argument("--yolo", default=DEFAULT_YOLO, help=f"Ultralytics weights (default: {DEFAULT_YOLO}).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25).")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (default: 640).")

    calib = parser.add_argument_group("metric calibration (pick one)")
    calib.add_argument("--distance", type=float, default=None,
                       help="Measured camera-to-object distance (m) of the most confident target (beta=0).")
    calib.add_argument("--anchor", action="append", default=[], metavar="U,V,DIST",
                       help="Ground-truth anchor 'u,v,distance_m'. Repeat >=2 times for an affine fit.")
    calib.add_argument("--object-height", type=float, default=None, metavar="M",
                       help="Known real height (m) of the --target class; distance from box height + focal length.")
    calib.add_argument("--calib", choices=["affine", "inverse"], default="affine",
                       help="Depth->distance model: 'affine' Z=alpha*D+beta (default) or "
                            "'inverse' 1/Z=alpha*D+beta (disparity-linear; better between anchors).")

    parser.add_argument("--depth-sample", choices=["box_median", "box_center", "bottom_center"],
                        default="box_median", help="Where to read an object's depth from its box (default: box_median).")
    parser.add_argument("--fx", type=float, default=None, help="Focal length in pixels (x).")
    parser.add_argument("--fy", type=float, default=None, help="Focal length in pixels (y).")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x (default: image center).")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y (default: image center).")
    parser.add_argument("--hfov", type=float, default=65.0,
                        help="Assumed horizontal FOV (deg) when --fx and EXIF are unavailable (default: 65).")
    parser.add_argument("--no-exif", dest="exif", action="store_false",
                        help="Do not read the focal length from EXIF metadata.")
    parser.set_defaults(exif=True)
    parser.add_argument("--out", type=Path, default=_SCRIPT_DIR / "outputs", help="Output directory.")
    parser.add_argument("--size", type=int, default=512, help="Synthetic image side when --image is omitted.")
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


def sample_depth(depth, u, v, radius=2):
    """Median relative depth in a small window (robust to stray pixels)."""
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    return float(np.median(depth[y0:y1, x0:x1]))


def object_depth(depth, box, mode):
    """Relative depth for a box: median over its interior, center, or ground line."""
    x1, y1, x2, y2 = box
    if mode == "box_median":
        bw, bh = x2 - x1, y2 - y1
        xa, xb = int(x1 + 0.2 * bw), int(x2 - 0.2 * bw)
        ya, yb = int(y1 + 0.2 * bh), int(y2 - 0.2 * bh)
        return float(np.median(depth[ya:yb, xa:xb]))
    if mode == "box_center":
        return sample_depth(depth, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return sample_depth(depth, (x1 + x2) / 2.0, y2 - 0.05 * (y2 - y1))


def exif_focal_pixels(src_path, width):
    """Focal length in pixels from EXIF ``FocalLengthIn35mmFilm``, else None."""
    try:
        with Image.open(src_path) as im:
            f35 = im.getexif().get(_EXIF_FOCAL_35MM)
    except Exception:
        return None
    if not f35:
        return None
    return float(f35) / _FULLFRAME_WIDTH_MM * float(width)


def resolve_intrinsics(args, src_path, width, height):
    """Return ``(fx, fy, cx, cy, source)`` trying CLI -> EXIF -> assumed FOV."""
    if args.fx or args.fy:
        fx = args.fx if args.fx else args.fy
        fy = args.fy if args.fy else args.fx
        cx = args.cx if args.cx is not None else width / 2.0
        cy = args.cy if args.cy is not None else height / 2.0
        return fx, fy, cx, cy, "cli"
    if args.exif and src_path is not None:
        f = exif_focal_pixels(src_path, width)
        if f:
            cx = args.cx if args.cx is not None else width / 2.0
            cy = args.cy if args.cy is not None else height / 2.0
            return f, f, cx, cy, "exif"
    fx = 0.5 * width / math.tan(math.radians(args.hfov) / 2.0)
    print(f"[warn] no --fx and no usable EXIF; assuming a {args.hfov:g} deg horizontal FOV (f={fx:.0f} px).")
    return fx, fx, width / 2.0, height / 2.0, "hfov"


def run_depth(image, model_id):
    """Zero-shot relative depth map (larger = nearer for the '-hf' checkpoints)."""
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(task="depth-estimation", model=MODELS.get(model_id, model_id), device=device)
    predicted = pipe(image)["predicted_depth"]
    if hasattr(predicted, "detach"):
        predicted = predicted.detach().cpu().numpy()
    return np.asarray(predicted, dtype=np.float32).squeeze()


def run_detections(image, weights, conf, imgsz):
    """All YOLO detections as dicts sorted by descending confidence."""
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)
    result = model.predict(source=image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    detections = []
    for box in result.boxes:
        detections.append(
            {
                "label": model.names[int(box.cls.item())],
                "conf": float(box.conf.item()),
                "box": tuple(float(v) for v in box.xyxy[0].tolist()),
            }
        )
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def fit_calibration(args, depth):
    """Return ``(alpha, beta, model, description)`` for the depth->distance model.

    ``model == 'affine'``  -> ``Z = alpha * D + beta`` (the proposal's linear form).
    ``model == 'inverse'`` -> ``Z = 1 / (alpha * D + beta)``, i.e. a linear fit of
    the disparity ``1/Z``, which matches the roughly inverse-depth output of
    monocular depth models and interpolates far better between anchors.
    """
    model = args.calib
    if args.anchor:
        points = []
        for spec in args.anchor:
            parts = spec.split(",")
            if len(parts) != 3:
                sys.exit(f"--anchor expects 'u,v,distance_m', got '{spec}'")
            u, v, dist = (float(p) for p in parts)
            if dist <= 0:
                sys.exit(f"--anchor distance must be positive meters, got '{spec}'")
            points.append((sample_depth(depth, u, v), dist))
        depths = [p[0] for p in points]
        distances = [p[1] for p in points]
        targets = [1.0 / d for d in distances] if model == "inverse" else distances
        if len(points) >= 2:
            alpha, beta = np.polyfit(depths, targets, 1)
            return float(alpha), float(beta), model, f"{model} fit on {len(points)} anchors"
        if depths[0] == 0:
            sys.exit("Anchor depth is 0; pick a different --anchor pixel.")
        return targets[0] / depths[0], 0.0, model, f"{model} scale fit on 1 anchor (beta=0)"
    if args.distance is not None:
        return None, None, model, "--distance (applied to the primary target)"
    return None, None, model, None


def annotate(image, entries):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    thickness = max(2, canvas.size[1] // 400)
    try:
        font = ImageFont.load_default(size=max(16, canvas.size[1] // 55))
    except TypeError:
        font = ImageFont.load_default()
    for entry in entries:
        x1, y1, x2, y2 = entry["x1"], entry["y1"], entry["x2"], entry["y2"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=thickness)
        lines = [
            f"{entry['label']} {entry['conf']:.2f}",
            f"Z={entry['z_m'] * 100:.1f}cm",
            f"W={entry['w_m'] * 100:.1f} H={entry['h_m'] * 100:.1f}cm",
        ]
        draw.multiline_text(
            (x1 + 4, max(0.0, y1 - 4 * len(lines) * 8)),
            "\n".join(lines),
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return canvas


def write_results(out_dir, stem, meta, results):
    json_path = out_dir / f"{stem}_results.json"
    csv_path = out_dir / f"{stem}_results.csv"
    json_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    fields = ["label", "conf", "x1", "y1", "x2", "y2", "depth", "z_m", "w_m", "h_m", "area_m2"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    return json_path, csv_path


def main(argv=None):
    args = parse_args(argv)
    if args.object_height is None and args.distance is None and not args.anchor:
        sys.exit("No calibration: pass --anchor u,v,distance_m (>=2), --distance <m>, or --object-height <m>.")

    image, src_path = load_image(args.image, args.size)
    args.out.mkdir(parents=True, exist_ok=True)
    if src_path is None:
        print("[warn] --image not given; using a synthetic image (smoke test only).")
        src_path = args.out / "synthetic_input.png"
        image.save(src_path)

    stem = Path(src_path).stem
    fx, fy, cx, cy, intrinsic_src = resolve_intrinsics(args, src_path, image.width, image.height)
    print(f"[info] image     : {src_path}  ({image.width}x{image.height})")
    print(f"[info] camera    : fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} (from {intrinsic_src})")

    t0 = time.perf_counter()
    depth = run_depth(image, args.depth_model)
    print(f"[info] depth     : {depth.shape} in {time.perf_counter() - t0:.2f}s "
          f"range=[{depth.min():.3f}, {depth.max():.3f}]")

    detections = run_detections(image, args.yolo, args.conf, args.imgsz)
    if not detections:
        sys.exit("No objects detected; try a lower --conf or a different image.")
    targets = detections
    if args.target:
        targets = [d for d in detections if d["label"].lower() == args.target.lower()]
        if not targets:
            found = ", ".join(sorted({d["label"] for d in detections}))
            sys.exit(f"No '{args.target}' detected. Detected: {found}")
    print(f"[info] detection : {len(detections)} total, {len(targets)} to size"
          + (f" (target='{args.target}')" if args.target else ""))

    # Metric calibration: depth affine, or per-object known-size geometry.
    alpha = beta = None
    model = args.calib
    how = "object-height (box height + focal length)"
    if args.object_height is None:
        alpha, beta, model, how = fit_calibration(args, depth)
        if alpha is None:  # --distance: anchor on the most confident target's own depth
            primary = targets[0]
            primary_depth = object_depth(depth, primary["box"], args.depth_sample)
            if primary_depth == 0:
                sys.exit("Primary target depth is 0; try a different --depth-sample.")
            target = 1.0 / args.distance if model == "inverse" else args.distance
            alpha, beta = target / primary_depth, 0.0
            how = f"--distance on '{primary['label']}' (beta=0)"
        print(f"[info] calibration: {model} alpha={alpha:.4f} beta={beta:.4f} ({how})")
    else:
        print(f"[info] calibration: object-height={args.object_height} m ({how})")

    results = []
    for det in targets:
        box = det["box"]
        rel = object_depth(depth, box, args.depth_sample)
        if args.object_height is not None:
            pixel_h = box[3] - box[1]
            if pixel_h <= 0:
                continue
            z = args.object_height * fy / pixel_h
        else:
            denom = alpha * rel + beta
            z = (1.0 / denom if denom > 0 else -1.0) if model == "inverse" else denom
        if z <= 0:
            print(f"[warn] skipping '{det['label']}': non-positive Z={z:.3f} m (check calibration).")
            continue
        w = abs(box[2] - box[0]) * z / fx
        h = abs(box[3] - box[1]) * z / fy
        results.append(
            {
                "label": det["label"],
                "conf": det["conf"],
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "depth": rel, "z_m": z, "w_m": w, "h_m": h, "area_m2": w * h,
            }
        )

    if not results:
        sys.exit("No objects could be sized (all had non-positive Z); check the calibration.")

    print(f"\n{'label':<12}{'conf':>6}{'depth':>9}{'Z(m)':>9}{'W(cm)':>9}{'H(cm)':>9}{'area(cm^2)':>12}")
    for r in results:
        print(f"{r['label']:<12}{r['conf']:>6.2f}{r['depth']:>9.3f}{r['z_m']:>9.3f}"
              f"{r['w_m'] * 100:>9.1f}{r['h_m'] * 100:>9.1f}{r['area_m2'] * 1e4:>12.1f}")

    lo, hi = float(depth.min()), float(depth.max())
    norm = (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)
    Image.fromarray(to_uint8(norm), "L").save(args.out / f"{stem}_depth_gray.png")
    Image.fromarray(apply_colormap(norm), "RGB").save(args.out / f"{stem}_depth_color.png")
    annotate(image, results).save(args.out / f"{stem}_pipeline.png")

    meta = {
        "image": str(src_path),
        "width": image.width, "height": image.height,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy, "intrinsics": intrinsic_src,
        "depth_model": MODELS.get(args.depth_model, args.depth_model),
        "yolo": args.yolo,
        "calibration": {"mode": how, "model": model, "alpha": alpha, "beta": beta,
                        "object_height_m": args.object_height, "distance_m": args.distance,
                        "anchors": args.anchor},
    }
    json_path, csv_path = write_results(args.out, stem, meta, results)
    print(f"\n[info] wrote     : {args.out / f'{stem}_pipeline.png'}")
    print(f"[info] wrote     : {args.out / f'{stem}_depth_color.png'}")
    print(f"[info] wrote     : {json_path}")
    print(f"[info] wrote     : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
