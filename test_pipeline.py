#!/usr/bin/env python3
"""Combined zero-shot metric-sizing pipeline (single command + batch mode).

This is the joined version of the two smoke scripts plus the geometric sizing
logic. One command runs the whole pipeline from ``Research_Proposals.md``:

    image
      -> depth model         (zero-shot relative depth map D; Depth Anything V2 by
                              default, MiDaS via --depth-model midas, Marigold
                              via --depth-model marigold)
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
    --anchor-file PATH        read the anchors straight from an anchor.txt
                              (lines like "cup1 anchor (636, 663, 40cm)")
    --object-height 0.20      known real height (m) of the --target class; the
                              distance is recovered from the box height and the
                              focal length (no depth anchor needed)

``--calib`` chooses how the anchors are turned into meters. ``affine`` (default)
fits ``Z = alpha*D + beta`` as in the proposal. ``inverse`` fits the disparity
``1/Z = alpha*D + beta``, which matches the roughly inverse-depth output of
monocular depth models and interpolates far more accurately between anchors:

    python test_pipeline.py --image image1.png --target bottle --calib inverse \\
        --anchor 1463,935,0.2 --anchor 1229,308,1.0

Camera intrinsics come from ``--fx/--fy`` if given, otherwise from EXIF
(``FocalLengthIn35mmFilm``, matching the proposal's "student setup shortcut"),
otherwise an assumed ``--hfov`` (65 deg) is used.

Batch / dataset mode
--------------------
Pass ``--dataset <root>`` to size a whole dataset in one shot. The script then

  * walks ``<root>`` recursively and picks up every image,
  * finds the cup anchors itself - ``--anchor`` if given, else ``--anchor-file``,
    else the ``anchor.txt`` in the image's own folder. Every object folder is
    expected to carry its own ``dataset/<lighting>/<object>/anchor.txt``; a
    folder without one is skipped as a whole and reported once, so a half-shot
    dataset never silently borrows another scene's calibration,
  * if ``--depth-model all`` is given, runs **every model family** - Depth
    Anything V2 small + base + large, MiDaS, and Marigold - and reports how each
    one does. Any comma-separated list works (``small,marigold``); note Marigold
    is far slower than the regression heads, so the wall clock of an ``all``
    batch is essentially Marigold's price. Drop it with an explicit list when
    you want a fast pass.

    # one command for the whole dataset, every depth model family:
    python test_pipeline.py --dataset dataset --depth-model all --out outputs

Output layout in batch mode (mirrors the dataset tree, one subdir per model):

    <out>/<model>/<relative/path>/<stem>_{pipeline,depth_color,depth_gray}.png
    <out>/<model>/<relative/path>/<stem>_results.json | _results.csv
    <out>/eval_per_image.csv     one row per (model, image), middle object only
    <out>/eval_summary.csv       MAE / RMSE of Z per model
    <out>/eval_by_lighting.csv   the same, grouped by lighting token
    <out>/eval_by_distance.csv   the same, grouped by true distance
    <out>/eval_by_object.csv     the same, grouped by object id (b01, c01, bwl01, ...)

The evaluation picks the **middle object** (the detection whose box centre sits
between the two anchor pixels), matching ``ImageCollection.md``. Ground truth is
read from a ``ground_truth.csv`` (Section 8 schema) if one is found next to the
dataset; otherwise the true distance is parsed from the filename
(``..._50cm`` or ``..._d050_...``) and the lighting from the folder / filename.
Each ``anchor.txt`` is stamped into that image's ``_results.json`` meta and the
``anchor_file`` column of ``eval_per_image.csv``.

Ground truth is **optional**. Without it the pipeline still sizes every image and
just prints the model outputs (Z / W / H) instead of a comparison - it never
errors. Pass ``--no-eval`` to skip the evaluation deliberately and only print the
predictions. Note the cup anchors are NOT ground truth: they are the fixed
calibration constant that turns relative depth into meters, always needed for a
metric result (``--distance`` or ``--object-height`` work too, when there are no
cups in the shot).

Object W/H ground truth is read from the ``objects.txt`` sitting next to the
dataset (``dataset/objects.txt``, or up to three folders above it). Its
``target objects:`` section holds the measured-once ``a01 W:7.5cm,H:7cm`` style
sizes for the five objects and fills ``true_w_m`` / ``true_h_m`` (hence the
``w_err_cm`` / ``h_err_cm`` columns and the per-image ``[W+1.1/H+0.8cm]`` tag).
The ``anchors`` section below it is calibration, not ground truth, and is
ignored - that covers the handle-less cup pair (8x8 cm) used as the anchors in
the b01 folders only (every other folder uses the with-handle 12x8 cm cups).

Usage
-----
    python test_pipeline.py --image photo.jpg --target bottle --distance 0.5
    python test_pipeline.py --image photo.jpg --target bottle \\
        --anchor 1900,1500,0.5 --anchor 3400,1500,1.2 --fx 3200
    python test_pipeline.py --image photo.jpg --target bottle --object-height 0.20
    python test_pipeline.py --image photo.jpg --target bottle --depth-model midas \\
        --object-height 0.20
    python test_pipeline.py --image photo.jpg --target bottle --depth-model marigold \\
        --anchor 1900,1500,0.5 --anchor 3400,1500,1.2
    python test_pipeline.py --image photo.jpg --target bottle --yolo-size large

    # whole-dataset batch, every depth model, auto anchors + auto eval (affine,
    # because Marigold is direct-depth; use --calib inverse for the inverse-depth
    # heads only, e.g. --depth-model small,base,large,midas):
    python test_pipeline.py --dataset dataset --depth-model all --calib affine
    python test_pipeline.py --dataset dataset/white --depth-model large --conf 0.2

``--yolo-size nano|small|medium|large|xlarge`` selects the YOLOv8 backbone
(default ``nano``); a larger one detects small, low-contrast objects such as the
dataset's apple far more reliably at the cost of speed. ``--yolo <weights>``
still takes an explicit path/name and overrides the preset.

``--depth-model marigold`` runs Marigold (``prs-eth/marigold-depth-v1-0``) through
``diffusers``, an optional dependency: ``pip install diffusers scipy`` (``scipy``
is only needed for ensembles). ``marigold-lcm`` selects the few-step LCM
checkpoint. ``--marigold-steps`` (default 10; the paper uses 50),
``--marigold-ensemble`` (default 1; the paper uses 10) and
``--marigold-resolution`` (default 768 px) trade runtime for accuracy. Marigold
emits *direct* depth (larger = farther), the opposite convention to the Depth
Anything V2 ``-hf`` checkpoints (larger = nearer); the affine fit learns the sign
either way, so keep ``--calib affine`` (the default) for Marigold and reserve
``--calib inverse`` for the inverse-depth checkpoints.

If ``--image`` is omitted a synthetic image is generated so the script also
doubles as an environment/dependency smoke test (a calibration flag is still
required; expect meaningless numbers).

Install
-------
    pip install torch transformers pillow numpy ultralytics
    # Marigold (optional): pip install diffusers scipy
    # Headless hosts: use opencv-python-headless instead of opencv-python.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
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
    # Marigold (CVPR'24) is a diffusers repo, not a transformers one, and predicts
    # direct depth (larger = farther) rather than inverse depth.
    "marigold": "prs-eth/marigold-depth-v1-0",
    "marigold-lcm": "prs-eth/marigold-depth-lcm-v1-0",
}
DEFAULT_DEPTH_MODEL = MODELS["small"]
DEFAULT_YOLO = "yolov8n.pt"
# ``--yolo-size`` presets; a larger model finds small/low-contrast objects better
# but is slower. An explicit ``--yolo`` path always wins over the preset.
YOLO_SIZES = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
    "xlarge": "yolov8x.pt",
}
_SCRIPT_DIR = Path(__file__).resolve().parent

# ``--depth-model all`` runs every model family (in this order) over the whole
# dataset: the three Depth Anything V2 sizes, the MiDaS control baseline, and
# Marigold. Marigold's diffusion loop is orders of magnitude slower than the
# regression heads, so an ``all`` batch is dominated by it - drop it with an
# explicit list (``--depth-model small,base,large,midas``) when you want a fast
# pass. The two variants (``midas-small``, ``marigold-lcm``) are NOT part of
# ``all``: they are faster stand-ins for a family already covered above, so ask
# for them explicitly (``--depth-model small,midas-small,marigold-lcm``).
BATCH_DEPTH_MODELS = ("small", "base", "large", "midas", "marigold")

# Marigold defaults. v1-0 was trained for 10-50 DDIM steps and evaluated with a
# 10-sample ensemble; 10 steps without ensembling keeps a dataset run bearable
# (raise both for final numbers), and 768 px is the checkpoint's native size.
MARIGOLD_STEPS = 10
MARIGOLD_ENSEMBLE = 1
MARIGOLD_RESOLUTION = 768

# Extensions and lighting tokens recognised while scanning a dataset folder.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
LIGHTING_TOKENS = ("white", "warm", "dim", "nv")

# Parsed anchor.txt contents, keyed by resolved path (batch mode hits the same
# per-folder file once per image per model, so parse it once).
_ANCHOR_CACHE = {}

# Parsed objects.txt (real W/H per object id), keyed by resolved path.
_OBJECT_SIZES_CACHE = {}

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


class PipelineSkip(Exception):
    """Raised when one image cannot be sized; batch mode logs it and moves on."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=Path, default=None, help="Input image (synthetic if omitted).")
    parser.add_argument("--dataset", type=Path, default=None, metavar="DIR",
                        help="Batch mode: scan DIR recursively, run every image, write an evaluation.")
    parser.add_argument("--target", default=None, help="COCO class to size, e.g. bottle (default: all detections).")
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL,
                        help="small/base/large (Depth Anything V2, default) | midas | midas-small | "
                             "marigold | marigold-lcm | 'all' (batch: small+base+large+midas+marigold) | a "
                             "comma list like small,marigold | a HF id.")
    parser.add_argument("--yolo", default=None,
                        help=f"Ultralytics weights, e.g. yolov8m.pt (default: {DEFAULT_YOLO}, or the "
                             "--yolo-size preset).")
    parser.add_argument("--yolo-size", choices=list(YOLO_SIZES), default=None,
                        help="YOLOv8 size preset: nano | small | medium | large | xlarge "
                             "(default: nano; overridden by --yolo).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25).")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (default: 640).")

    marigold = parser.add_argument_group("marigold (needs: pip install diffusers scipy)")
    marigold.add_argument("--marigold-steps", type=int, default=MARIGOLD_STEPS,
                          help=f"Marigold denoising steps (default: {MARIGOLD_STEPS}; paper uses 50).")
    marigold.add_argument("--marigold-ensemble", type=int, default=MARIGOLD_ENSEMBLE,
                          help=f"Marigold ensemble size (default: {MARIGOLD_ENSEMBLE}; paper uses 10).")
    marigold.add_argument("--marigold-resolution", type=int, default=MARIGOLD_RESOLUTION,
                          help=f"Marigold processing resolution in px (default: {MARIGOLD_RESOLUTION}).")

    calib = parser.add_argument_group("metric calibration (pick one)")
    calib.add_argument("--distance", type=float, default=None,
                       help="Measured camera-to-object distance (m) of the most confident target (beta=0).")
    calib.add_argument("--anchor", action="append", default=[], metavar="U,V,DIST",
                       help="Ground-truth anchor 'u,v,distance_m'. Repeat >=2 times for an "
                            "affine fit; overrides any anchor.txt.")
    calib.add_argument("--anchor-file", type=Path, default=None, metavar="PATH",
                       help="Force ONE anchor.txt for every image (overrides the per-folder "
                            "discovery; 'cup1 anchor (636, 663, 40cm)' lines).")
    calib.add_argument("--object-height", type=float, default=None, metavar="M",
                       help="Known real height (m) of the --target class; distance from box height + focal length.")
    calib.add_argument("--calib", choices=["affine", "inverse"], default="affine",
                       help="Depth->distance model: 'affine' Z=alpha*D+beta (default) or "
                            "'inverse' 1/Z=alpha*D+beta (disparity-linear; better between anchors, "
                            "but only valid for inverse-depth models - not Marigold).")

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
    parser.add_argument("--no-eval", dest="no_eval", action="store_true",
                        help="Batch mode: just print the model predictions and skip the ground-truth "
                             "evaluation entirely (no ground truth needed, no eval CSVs written).")
    parser.add_argument("--size", type=int, default=512, help="Synthetic image side when --image is omitted.")
    args = parser.parse_args(argv)
    if args.yolo is None:
        args.yolo = YOLO_SIZES[args.yolo_size] if args.yolo_size else DEFAULT_YOLO
    return args


# --------------------------------------------------------------------------- #
# dataset / anchor / ground-truth helpers                                      #
# --------------------------------------------------------------------------- #
def parse_anchor_file(path):
    """Parse an anchor file into ``['u,v,dist_m', ...]``.

    Accepts lines like ``cup1 anchor (636, 663, 40cm)`` (distance in ``cm`` or
    ``m``); blank lines and ``#`` comments are ignored.
    """
    path = Path(path)
    if not path.is_file():
        sys.exit(f"Anchor file not found: {path}")
    anchors = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.search(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*(cm|m)?\s*\)", line, re.I)
        if not m:
            continue
        u, v, dist = m.group(1), m.group(2), float(m.group(3))
        if (m.group(4) or "cm").lower() == "cm":
            dist /= 100.0
        anchors.append(f"{u},{v},{dist:g}")
    if not anchors:
        sys.exit(f"No anchors parsed from {path}; expected 'name anchor (u, v, 40cm)' lines.")
    return anchors


def anchor_file_for(image_path):
    """The ``anchor.txt`` that must sit in an image's OWN folder, or ``None``.

    Anchors are stored per object folder (``dataset/<lighting>/<object>/
    anchor.txt``). There is deliberately no inheritance from a parent folder: a
    folder without the file is a folder whose calibration is unknown, so callers
    skip it and warn instead of borrowing another scene's pixels.
    """
    path = Path(image_path).resolve().parent / "anchor.txt"
    path = path if path.is_file() else None
    return path


def load_anchor_file(path, cache=None):
    """``parse_anchor_file`` with memoisation - a batch reuses folders."""
    cache = _ANCHOR_CACHE if cache is None else cache
    key = str(Path(path).resolve())
    if key not in cache:
        cache[key] = parse_anchor_file(path)
    return cache[key]


def resolve_anchors(args, image_path, cache=None):
    """Anchors for one image: ``--anchor`` > ``--anchor-file`` > its folder's anchor.txt.

    Returns ``(anchors, anchor_file)``; ``anchors`` is an empty list when the
    image's folder has no anchor.txt, which makes the caller skip that folder
    (unless ``--distance`` / ``--object-height`` calibrates it another way).
    """
    if args.anchor:
        return list(args.anchor), None
    if args.anchor_file:
        return load_anchor_file(args.anchor_file, cache), Path(args.anchor_file)
    path = anchor_file_for(image_path)
    if path is not None:
        return load_anchor_file(path, cache), path
    return [], None


def find_images(root):
    """Every image under ``root`` (recursive), sorted for deterministic runs."""
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"--dataset is not a directory: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def derive_truth(image_path, dataset_root=None):
    """Best-effort ``(distance_m, lighting)`` from the path/filename.

    Distance: ``..._50cm`` or ``..._d050_...``. Lighting: a ``white|warm|dim|nv``
    folder or filename token. Both may be ``None`` (then that metric is skipped).
    """
    name = Path(image_path).name.lower()
    distance = None
    m = re.search(r"(\d{2,3})\s*cm", name)
    if m:
        distance = int(m.group(1)) / 100.0
    else:
        m = re.search(r"_d(\d{3})_", name)
        if m:
            distance = int(m.group(1)) / 100.0

    lighting = None
    parts = [p.lower() for p in Path(image_path).parts]
    for tok in LIGHTING_TOKENS:
        if tok in parts:
            lighting = tok
            break
    if lighting is None:
        for tok in LIGHTING_TOKENS:
            if re.search(rf"(^|_){tok}($|_)", name):
                lighting = tok
                break
    return distance, lighting


def load_ground_truth(dataset_root):
    """Return ``{filename: row}`` from a ``ground_truth.csv`` near the dataset."""
    root = Path(dataset_root)
    for cand in (root / "ground_truth.csv", root.parent / "ground_truth.csv",
                 _SCRIPT_DIR / "ground_truth.csv"):
        if not cand.is_file():
            continue
        try:
            with cand.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        except Exception as exc:  # malformed CSV should not kill the run
            print(f"[warn] could not read {cand}: {exc}")
            continue
        table = {}
        for row in rows:
            key = (row.get("filename") or "").strip()
            if key:
                table[key] = row
                table[Path(key).stem] = row
                light = (row.get("lighting") or "").strip().lower()
                if light:
                    table[f"{light}/{key}"] = row
                    table[f"{light}/{Path(key).stem}"] = row
        print(f"[info] ground truth: {len(rows)} rows from {cand}")
        return table
    return {}


# object_id aliases: ``objects.txt`` keys are the short ids (b01, c01, bwl01),
# while the ImageCollection.md filename pattern spells the class out
# (``bottle01_d050_white`` -> b01, ``bowl01_...`` -> bwl01).
_OBJECT_ID_WORDS = {"bottle": "b", "apple": "a", "cup": "c", "bowl": "bwl"}

# Reverse of ``_OBJECT_ID_WORDS``: the short id prefix -> the COCO class it is.
# ``a01`` -> apple, ``b01``/``b02`` -> bottle, ``bwl01`` -> bowl, ``c01`` -> cup.
_OBJECT_ID_CLASS = {v: k for k, v in _OBJECT_ID_WORDS.items()}


def expected_class_for(object_id):
    """COCO class a dataset object id is supposed to be, or None if unknown.

    ``b01`` -> ``bottle``, ``bwl01`` -> ``bowl``, ``bottle01`` -> ``bottle``.
    Used so the evaluation only ever scores the object the folder is about: a
    cup that happens to be in frame must never be scored as the subject.
    """
    if not object_id:
        return None
    match = re.match(r"([a-z]+)\d*$", str(object_id).strip().lower())
    if not match:
        return None
    word = match.group(1)
    return _OBJECT_ID_CLASS.get(_OBJECT_ID_WORDS.get(word, word))


def object_id_for(image_path, gt=None):
    """Dataset object id for an image: ground-truth column, else filename prefix."""
    if gt:
        for key in ("object_id", "bottle_id"):
            val = (gt.get(key) or "").strip()
            if val:
                return val
    match = re.match(r"([a-z]+\d+)_", Path(image_path).name.lower())
    return match.group(1) if match else ""



def load_object_sizes(dataset_root=None, quiet=False):
    """Return ``{object_id: {'w_m': W, 'h_m': H}}`` from a ``objects.txt``.

    ``dataset/objects.txt`` is the measured-once size table for the set::

        target objects:
        a01 W:7.5cm,H:7cm
        b01 W:6.2cm,H:24cm
        ...
        anchors
        cups W:12cm, H:8cm
        cups without handle W:8cm, H:8cm

    Only the ``target objects:`` section is used - those W/H values are the
    ground truth the eval scores ``w_m`` / ``h_m`` against, for every shot of
    that object (measure once, reuse).

    The ``anchors`` section is calibration, NOT ground truth, so it is skipped:
    the reference cups are the fixed 0.40 / 1.00 m constants in each folder's
    ``anchor.txt``. Which pair is on the marks differs by scene - the handle-less
    cups (8x8 cm) are used in the **b01** folders ONLY, every other object folder
    is shot with the with-handle cups (12x8 cm) - so neither ever becomes a
    scored object here.
    """
    if dataset_root is None:
        return {}
    root = Path(dataset_root).resolve()
    # The table usually sits at ``dataset/objects.txt`` while a batch may be
    # pointed at a single subfolder (``--dataset dataset/white/c1``), so look at
    # the folder itself and a few ancestors before giving up.
    candidates = [root / "objects.txt"]
    candidates += [parent / "objects.txt" for parent in list(root.parents)[:3]]
    candidates.append(_SCRIPT_DIR / "objects.txt")
    for cand in candidates:
        if not cand.is_file():
            continue
        key = str(cand.resolve())
        if key in _OBJECT_SIZES_CACHE:
            return _OBJECT_SIZES_CACHE[key]
        sizes, section = {}, None
        for line in cand.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.endswith(":"):  # section header, e.g. "target objects:"
                section = stripped[:-1].strip().lower()
                continue
            if section != "target objects":
                continue
            m = re.match(r"([A-Za-z]+\d+)\s+W:\s*([\d.]+)\s*cm\s*,?\s*H:\s*([\d.]+)\s*cm",
                         stripped, re.I)
            if not m:
                continue
            sizes[m.group(1).lower()] = {"w_m": float(m.group(2)) / 100.0,
                                         "h_m": float(m.group(3)) / 100.0}
        _OBJECT_SIZES_CACHE[key] = sizes
        if sizes and not quiet:
            print(f"[info] object size: {len(sizes)} object(s) W/H from {cand}")
        return sizes
    return {}


def object_size_for(object_id, object_sizes):
    """True ``(W_m, H_m)`` for an object id, or ``(None, None)``.

    Accepts both the short ids used by ``objects.txt`` (``b01``) and the
    spelled-out filename prefixes (``bottle01`` -> ``b01``, ``bowl01`` -> ``bwl01``).
    """
    if not object_id or not object_sizes:
        return None, None
    oid = str(object_id).strip().lower()
    entry = object_sizes.get(oid)
    if entry is None:
        m = re.match(r"([a-z]+)(\d+)$", oid)
        if m:
            entry = object_sizes.get(_OBJECT_ID_WORDS.get(m.group(1), m.group(1)) + m.group(2))
    if not entry:
        return None, None
    return entry.get("w_m"), entry.get("h_m")


# --------------------------------------------------------------------------- #
# imaging / depth / detection primitives                                       #
# --------------------------------------------------------------------------- #
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


def resolve_intrinsics(args, src_path, width, height, warn=True):
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
    if warn:
        print(f"[warn] no --fx and no usable EXIF; assuming a {args.hfov:g} deg horizontal FOV (f={fx:.0f} px).")
    return fx, fx, width / 2.0, height / 2.0, "hfov"


def is_marigold(model_name):
    """True for Marigold checkpoints (diffusers repo, direct depth)."""
    return "marigold" in str(model_name).lower()


def make_depth_pipe(model_id=None):
    """Build a depth model once (reused across a whole batch)."""
    model_name = MODELS.get(model_id, model_id)
    if not is_marigold(model_name):
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(task="depth-estimation", model=model_name, device=device)

    try:
        from diffusers import MarigoldDepthPipeline
    except ImportError:
        sys.exit("Marigold runs through 'diffusers', which is an optional dependency.\n"
                 "Install with: pip install diffusers scipy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
    if device == "cuda":
        kwargs["variant"] = "fp16"  # half the download; skipped below if absent
    try:
        pipe = MarigoldDepthPipeline.from_pretrained(model_name, **kwargs)
    except Exception:
        if "variant" not in kwargs:
            raise
        kwargs.pop("variant")
        kwargs["torch_dtype"] = torch.float32
        pipe = MarigoldDepthPipeline.from_pretrained(model_name, **kwargs)
    return pipe.to(device)


def run_depth(image, model_id, pipe=None, steps=None, ensemble=None, resolution=None):
    """Zero-shot relative depth map for one image.

    Orientation depends on the family: the Depth Anything V2 ``-hf`` checkpoints
    (and MiDaS) emit *inverse* depth, larger = nearer; Marigold emits *direct*
    depth, larger = farther. The affine fit learns the sign either way, but only
    the inverse-depth family suits ``--calib inverse``.
    """
    if pipe is None:
        pipe = make_depth_pipe(model_id)
    if is_marigold(MODELS.get(model_id, model_id)):
        out = pipe(
            image,
            num_inference_steps=steps or MARIGOLD_STEPS,
            ensemble_size=ensemble or MARIGOLD_ENSEMBLE,
            processing_resolution=resolution or MARIGOLD_RESOLUTION,
            match_input_resolution=True,
            output_type="np",
        )
        predicted = out.prediction
    else:
        predicted = pipe(image)["predicted_depth"]
    if hasattr(predicted, "detach"):
        predicted = predicted.detach().cpu().numpy()
    return np.asarray(predicted, dtype=np.float32).squeeze()


def make_yolo(weights):
    return YOLO(weights)


def run_detections(image, weights, conf, imgsz, model=None):
    """All YOLO detections as dicts sorted by descending confidence."""
    if model is None:
        model = make_yolo(weights)
    device = 0 if torch.cuda.is_available() else "cpu"
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


def fit_calibration(args, depth, anchors=None):
    """Return ``(alpha, beta, model, description)`` for the depth->distance model.

    ``model == 'affine'``  -> ``Z = alpha * D + beta`` (the proposal's linear form).
    ``model == 'inverse'`` -> ``Z = 1 / (alpha * D + beta)``, i.e. a linear fit of
    the disparity ``1/Z``, which matches the roughly inverse-depth output of
    monocular depth models and interpolates far better between anchors.
    """
    model = args.calib
    anchors = args.anchor if anchors is None else anchors
    if anchors:
        points = []
        for spec in anchors:
            parts = spec.split(",")
            if len(parts) != 3:
                sys.exit(f"anchor expects 'u,v,distance_m', got '{spec}'")
            u, v, dist = (float(p) for p in parts)
            if dist <= 0:
                sys.exit(f"anchor distance must be positive meters, got '{spec}'")
            points.append((sample_depth(depth, u, v), dist))
        depths = [p[0] for p in points]
        distances = [p[1] for p in points]
        targets = [1.0 / d for d in distances] if model == "inverse" else distances
        if len(points) >= 2:
            alpha, beta = np.polyfit(depths, targets, 1)
            return float(alpha), float(beta), model, f"{model} fit on {len(points)} anchors"
        if depths[0] == 0:
            sys.exit("Anchor depth is 0; pick a different anchor pixel.")
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


def save_outputs(out_dir, stem, image, depth, results, meta):
    """Write the annotated overlay, depth visualizations and results files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = float(depth.min()), float(depth.max())
    norm = (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)
    Image.fromarray(to_uint8(norm), "L").save(out_dir / f"{stem}_depth_gray.png")
    Image.fromarray(apply_colormap(norm), "RGB").save(out_dir / f"{stem}_depth_color.png")
    annotate(image, results).save(out_dir / f"{stem}_pipeline.png")
    return write_results(out_dir, stem, meta, results)


# --------------------------------------------------------------------------- #
# core per-image sizing                                                        #
# --------------------------------------------------------------------------- #
def size_image(args, image, src_path, model_key, pipe=None, yolo_model=None, anchors=None,
               anchor_file=None, quiet=False):
    """Run depth -> YOLO -> calibration -> metric sizing on one image.

    Returns a dict with ``meta`` / ``results`` / ``depth``. Raises PipelineSkip
    when the image cannot be sized (no detection, no target, all Z <= 0) so a
    batch can log it and carry on.
    """
    anchors = args.anchor if anchors is None else anchors
    fx, fy, cx, cy, intrinsic_src = resolve_intrinsics(args, src_path, image.width, image.height, warn=not quiet)
    if not quiet:
        print(f"[info] image     : {src_path}  ({image.width}x{image.height})")
        print(f"[info] camera    : fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} (from {intrinsic_src})")

    t0 = time.perf_counter()
    depth = run_depth(image, model_key, pipe, args.marigold_steps,
                      args.marigold_ensemble, args.marigold_resolution)
    if not quiet:
        print(f"[info] depth     : {depth.shape} in {time.perf_counter() - t0:.2f}s "
              f"range=[{depth.min():.3f}, {depth.max():.3f}]")

    detections = run_detections(image, args.yolo, args.conf, args.imgsz, yolo_model)
    if not detections:
        raise PipelineSkip("no objects detected (try a lower --conf)")
    targets = detections
    if args.target:
        targets = [d for d in detections if d["label"].lower() == args.target.lower()]
        if not targets:
            found = ", ".join(sorted({d["label"] for d in detections}))
            raise PipelineSkip(f"no '{args.target}' detected (found: {found})")
    if not quiet:
        print(f"[info] detection : {len(detections)} total, {len(targets)} to size"
              + (f" (target='{args.target}')" if args.target else ""))

    # Metric calibration: depth affine, or per-object known-size geometry.
    alpha = beta = None
    model = args.calib
    how = "object-height (box height + focal length)"
    if args.object_height is None:
        alpha, beta, model, how = fit_calibration(args, depth, anchors)
        if alpha is None:  # --distance: anchor on the most confident target's own depth
            primary = targets[0]
            primary_depth = object_depth(depth, primary["box"], args.depth_sample)
            if primary_depth == 0:
                raise PipelineSkip("primary target depth is 0 (try a different --depth-sample)")
            target = 1.0 / args.distance if model == "inverse" else args.distance
            alpha, beta = target / primary_depth, 0.0
            how = f"--distance on '{primary['label']}' (beta=0)"
        if not quiet:
            print(f"[info] calibration: {model} alpha={alpha:.4f} beta={beta:.4f} ({how})")
    elif not quiet:
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
        raise PipelineSkip("no objects could be sized (all had non-positive Z)")

    meta = {
        "image": str(src_path),
        "width": image.width, "height": image.height,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy, "intrinsics": intrinsic_src,
        "depth_model": MODELS.get(model_key, model_key),
        "yolo": args.yolo,
        "calibration": {"mode": how, "model": model, "alpha": alpha, "beta": beta,
                        "object_height_m": args.object_height, "distance_m": args.distance,
                        "anchor_file": str(anchor_file) if anchor_file else None,
                        "anchors": list(anchors)},
    }
    return {"meta": meta, "results": results, "depth": depth}


def print_results_table(results):
    print(f"\n{'label':<12}{'conf':>6}{'depth':>9}{'Z(m)':>9}{'W(cm)':>9}{'H(cm)':>9}{'area(cm^2)':>12}")
    for r in results:
        print(f"{r['label']:<12}{r['conf']:>6.2f}{r['depth']:>9.3f}{r['z_m']:>9.3f}"
              f"{r['w_m'] * 100:>9.1f}{r['h_m'] * 100:>9.1f}{r['area_m2'] * 1e4:>12.1f}")


def _anchor_points(anchors):
    """``[(u, v), ...]`` from ``"u,v,distance_m"`` anchor specs."""
    points = []
    for spec in anchors or []:
        parts = str(spec).split(",")
        try:
            points.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    return points


def _is_anchor_detection(result, points, slack=0.03):
    """True when a detection IS one of the calibration anchors.

    The user clicks a pixel on the reference cup, so that cup's bounding box
    contains the click. Testing containment is exact and does not depend on box
    size or on how far apart the objects are - unlike an x-range test, it cannot
    accidentally swallow the subject object.
    """
    x1, y1, x2, y2 = result["x1"], result["y1"], result["x2"], result["y2"]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    for u, v in points:
        if (x1 - slack * bw) <= u <= (x2 + slack * bw) and (y1 - slack * bh) <= v <= (y2 + slack * bh):
            return True
    return False


def pick_middle(results, anchors, expected=None):
    """The detection the dataset is actually measuring, or **None**.

    Candidates are the detections that are neither calibration anchor, further
    restricted to ``expected`` (a COCO class) when one is known. Among the rest,
    prefer the object whose box centre sits between the two anchor x's (the
    ``ImageCollection.md`` layout: the subject sits between cup1 and cup2) and
    take the most confident of those.

    Returning ``None`` is the important part: when the subject class was never
    detected, the caller must VOID the row. Scoring whatever else happens to be
    in frame is what silently attributed a cup's distance to the bottle/apple
    and inflated every aggregate in the evaluation.
    """
    if not results:
        return None
    points = _anchor_points(anchors)
    candidates = [r for r in results if not _is_anchor_detection(r, points)]
    if expected:
        wanted = expected.strip().lower()
        candidates = [r for r in candidates if str(r.get("label", "")).lower() == wanted]
    if not candidates:
        return None
    us = [u for u, _ in points]
    if len(us) >= 2:
        lo, hi = min(us), max(us)
        inside = [r for r in candidates if lo <= (r["x1"] + r["x2"]) / 2.0 <= hi]
        if inside:
            return max(inside, key=lambda r: r["conf"])
    return max(candidates, key=lambda r: r["conf"])


# --------------------------------------------------------------------------- #
# evaluation                                                                   #
# --------------------------------------------------------------------------- #
def eval_row(model, image_path, group, true_distance, lighting, picked, ground_truth,
             anchor_file=None, object_sizes=None):
    """One evaluation row for the picked (middle) object of one image."""
    row = {
        "model": model,
        "filename": Path(image_path).name,
        "group": group,
        "anchor_file": str(anchor_file) if anchor_file else "",
        "lighting": lighting,
        "distance_true_m": true_distance,
        "label": picked["label"] if picked else "",
        "conf": picked["conf"] if picked else None,
        "z_m": picked["z_m"] if picked else None,
        "w_m": picked["w_m"] if picked else None,
        "h_m": picked["h_m"] if picked else None,
        "true_w_m": None,
        "true_h_m": None,
    }
    name = Path(image_path).name
    stem = Path(image_path).stem
    light = (lighting or "").lower()
    gt = (ground_truth.get(f"{light}/{name}") or ground_truth.get(f"{light}/{stem}")
          or ground_truth.get(name) or ground_truth.get(stem))
    if gt:
        try:
            if gt.get("distance_m"):
                row["distance_true_m"] = float(gt["distance_m"])
        except ValueError:
            pass
        if gt.get("lighting"):
            row["lighting"] = gt["lighting"]
        for key in ("true_w_m", "true_h_m"):
            try:
                if gt.get(key):
                    row[key] = float(gt[key])
            except ValueError:
                pass

    # Object id: ground-truth column first (``object_id`` or the older
    # ``bottle_id``), else the filename prefix (``bottle01_d050_white`` ->
    # ``bottle01``). Grouped per object because the set mixes bottle/cup/bowl.
    row["object_id"] = object_id_for(image_path, gt)

    # W/H ground truth from objects.txt (measured once per object): used when
    # the ground_truth.csv row has no true_w_m / true_h_m of its own.
    if object_sizes and (row["true_w_m"] is None or row["true_h_m"] is None):
        true_w, true_h = object_size_for(row["object_id"], object_sizes)
        if row["true_w_m"] is None:
            row["true_w_m"] = true_w
        if row["true_h_m"] is None:
            row["true_h_m"] = true_h

    for name, pred_key, true_key, scale in (
        ("z", "z_m", "distance_true_m", 100.0),
        ("w", "w_m", "true_w_m", 100.0),
        ("h", "h_m", "true_h_m", 100.0),
    ):
        pred = row[pred_key]
        true = row[true_key]
        err = rel = None
        if pred is not None and true:
            err = (pred - true) * scale
            rel = (pred - true) / true * 100.0
        row[f"{name}_err_cm"] = err
        row[f"{name}_abs_err_cm"] = abs(err) if err is not None else None
        row[f"{name}_rel_err_pct"] = rel
    return row


def _agg(rows):
    """``(n, mae_z_cm, rmse_z_cm, mae_rel_pct)`` over rows that have a Z error."""
    zs = [r["z_abs_err_cm"] for r in rows if r["z_abs_err_cm"] is not None]
    if not zs:
        return None
    mae = sum(zs) / len(zs)
    rmse = math.sqrt(sum(x * x for x in zs) / len(zs))
    rels = [abs(r["z_rel_err_pct"]) for r in rows if r["z_rel_err_pct"] is not None]
    return len(zs), mae, rmse, (sum(rels) / len(rels) if rels else None)


def has_truth(rows):
    """True when any row has a real distance to compare against."""
    return any(r["z_err_cm"] is not None for r in rows)


def print_predictions_table(rows, model_keys):
    """Plain model-output listing for runs with no ground truth to compare to."""
    print("\n==================== model output (no ground truth) ====================")
    print(f"{'model':<8}{'image':<20}{'label':<10}{'Z(cm)':>8}{'W(cm)':>8}{'H(cm)':>8}")
    for model in model_keys:
        for r in [x for x in rows if x["model"] == model]:
            z = "" if r["z_m"] is None else f"{r['z_m'] * 100:.1f}"
            w = "" if r["w_m"] is None else f"{r['w_m'] * 100:.1f}"
            h = "" if r["h_m"] is None else f"{r['h_m'] * 100:.1f}"
            print(f"{model:<8}{r['filename']:<20}{(r['label'] or '-'):<10}{z:>8}{w:>8}{h:>8}")
    print("========================================================================")


def write_eval(out_dir, rows, model_keys, voided=None):
    """Write eval CSVs (per-image + grouped summaries) and print a summary table.

    ``rows`` holds scored rows only. ``voided`` lists the (model, image) pairs
    whose subject was never detected; they are excluded from every aggregate and
    written to ``eval_voided.csv`` so nothing disappears silently.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    voided = voided or []
    fields = [
        "model", "filename", "object_id", "group", "anchor_file", "lighting",
        "distance_true_m",
        "label", "conf", "z_m", "z_err_cm", "z_abs_err_cm", "z_rel_err_pct",
        "w_m", "true_w_m", "w_err_cm", "h_m", "true_h_m", "h_err_cm",
    ]
    per_path = out_dir / "eval_per_image.csv"
    with per_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})

    def group_by(key):
        out = {}
        for r in rows:
            out.setdefault(r[key], []).append(r)
        return out

    def write_grouped(path, key, header):
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["model", key, "n", "mae_z_cm", "rmse_z_cm", "mae_rel_pct"])
            for model in model_keys:
                buckets = group_by(key)
                for bucket in sorted({r[key] for r in rows if r["model"] == model},
                                     key=lambda x: (x is None, x)):
                    sub = [r for r in buckets.get(bucket, []) if r["model"] == model]
                    agg = _agg(sub)
                    if agg:
                        n, mae, rmse, rel = agg
                        writer.writerow([model, bucket, n, f"{mae:.2f}", f"{rmse:.2f}",
                                         "" if rel is None else f"{rel:.2f}"])
        print(f"[info] wrote     : {path}")

    summary_path = out_dir / "eval_summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "n_sized", "n_voided", "mae_z_cm", "rmse_z_cm", "mae_rel_pct"])
        for model in model_keys:
            n_void = sum(1 for v in voided if v["model"] == model)
            agg = _agg([r for r in rows if r["model"] == model])
            if agg:
                n, mae, rmse, rel = agg
                writer.writerow([model, n, n_void, f"{mae:.2f}", f"{rmse:.2f}",
                                 "" if rel is None else f"{rel:.2f}"])
            else:
                # Nothing scorable left for this model: keep the row so the model
                # does not silently vanish from the summary.
                writer.writerow([model, 0, n_void, "", "", ""])

    write_grouped(out_dir / "eval_by_lighting.csv", "lighting", "lighting")
    write_grouped(out_dir / "eval_by_distance.csv", "distance_true_m", "distance_m")
    write_grouped(out_dir / "eval_by_object.csv", "object_id", "object_id")

    if voided:
        void_path = out_dir / "eval_voided.csv"
        vfields = ["model", "filename", "object_id", "group", "lighting",
                   "distance_true_m", "expected", "found", "reason"]
        with void_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=vfields, extrasaction="ignore")
            writer.writeheader()
            for v in voided:
                writer.writerow({k: v.get(k) for k in vfields})
        print(f"[info] wrote     : {void_path}")

    print("\n==================== evaluation (target object) ====================")
    print(f"{'model':<8}{'n':>4}{'void':>5}{'MAE Z(cm)':>12}{'RMSE Z(cm)':>12}{'MAE rel(%)':>12}")
    for model in model_keys:
        agg = _agg([r for r in rows if r["model"] == model])
        n_void = sum(1 for v in voided if v["model"] == model)
        if agg:
            n, mae, rmse, rel = agg
            print(f"{model:<8}{n:>4}{n_void:>5}{mae:>12.2f}{rmse:>12.2f}"
                  f"{('' if rel is None else f'{rel:.1f}'):>12}")
        else:
            print(f"{model:<8}{0:>4}{n_void:>5}{'-':>12}{'-':>12}{'-':>12}")

    distances = sorted({r["distance_true_m"] for r in rows if r["distance_true_m"]})
    if distances and len(model_keys) >= 1:
        print("\nper true distance (Z abs error, cm):")
        print(f"{'model':<8}" + "".join(f"{d * 100:>8.0f}cm" for d in distances))
        for model in model_keys:
            line = f"{model:<8}"
            for d in distances:
                sub = [r for r in rows if r["model"] == model and r["distance_true_m"] == d]
                agg = _agg(sub)
                line += f"{agg[1]:>10.1f}" if agg else f"{'-':>10}"
            print(line)

    lightings = [t for t in LIGHTING_TOKENS if any(r["lighting"] == t for r in rows)]
    if lightings:
        print("\nper lighting (Z abs error, cm):")
        print(f"{'model':<8}" + "".join(f"{t:>9}" for t in lightings))
        for model in model_keys:
            line = f"{model:<8}"
            for t in lightings:
                sub = [r for r in rows if r["model"] == model and r["lighting"] == t]
                agg = _agg(sub)
                line += f"{agg[1]:>9.1f}" if agg else f"{'-':>9}"
            print(line)
    print("====================================================================")
    print(f"[info] wrote     : {per_path}")
    print(f"[info] wrote     : {summary_path}")


# --------------------------------------------------------------------------- #
# entry points                                                                 #
# --------------------------------------------------------------------------- #
def run_single(args, model_keys):
    """Original single-image flow, now with per-folder anchor discovery."""
    image, src_path = load_image(args.image, args.size)
    args.out.mkdir(parents=True, exist_ok=True)
    if src_path is None:
        print("[warn] --image not given; using a synthetic image (smoke test only).")
        anchors, anchor_file = list(args.anchor), None
        src_path = args.out / "synthetic_input.png"
        image.save(src_path)
    else:
        anchors, anchor_file = resolve_anchors(args, src_path)

    if anchor_file:
        print(f"[info] anchors  : {len(anchors)} from {anchor_file} -> {anchors}")
    if args.object_height is None and args.distance is None and not anchors:
        sys.exit("No calibration: pass --anchor u,v,distance_m (>=2), --anchor-file, "
                 "--distance <m>, or --object-height <m> "
                 f"({Path(src_path).resolve().parent} has no anchor.txt).")

    stem = Path(src_path).stem
    multi = len(model_keys) > 1
    for model_key in model_keys:
        try:
            rec = size_image(args, image, src_path, model_key, anchors=anchors,
                             anchor_file=anchor_file)
        except PipelineSkip as exc:
            sys.exit(f"[{model_key}] {exc}")
        print_results_table(rec["results"])
        out_dir = (args.out / model_key) if multi else args.out
        json_path, csv_path = save_outputs(out_dir, stem, image, rec["depth"], rec["results"], rec["meta"])
        print(f"\n[info] wrote     : {out_dir / f'{stem}_pipeline.png'}")
        print(f"[info] wrote     : {out_dir / f'{stem}_depth_color.png'}")
        print(f"[info] wrote     : {json_path}")
        print(f"[info] wrote     : {csv_path}")
    return 0


def run_batch(args, model_keys):
    """Scan a dataset folder, run every image through every model, write an eval.

    Every object folder must carry its own ``anchor.txt``. The decision is made
    per FOLDER, not per image: a folder without one is skipped as a whole and
    reported once, with a hint to add the file (``--anchor`` / ``--anchor-file``
    force anchors for everything, ``--distance`` / ``--object-height`` calibrate
    without them).
    """
    dataset_root = Path(args.dataset)
    images = find_images(dataset_root)
    if not images:
        sys.exit(f"No images found under {dataset_root}")

    # Group images by their object folder (the directory that holds anchor.txt).
    folders = {}
    for img in images:
        folders.setdefault(img.parent, []).append(img)

    cache = {}
    folder_anchors, folder_anchor_file = {}, {}
    for folder, imgs in folders.items():
        anchors, source = resolve_anchors(args, imgs[0], cache)
        folder_anchors[folder], folder_anchor_file[folder] = anchors, source

    have = sorted(f for f in folders if folder_anchors[f])
    missing = sorted(f for f in folders if not folder_anchors[f])
    if args.anchor:
        print(f"[info] anchors  : {len(args.anchor)} from --anchor, forced on all "
              f"{len(images)} image(s)")
    elif args.anchor_file:
        print(f"[info] anchors  : {args.anchor_file} forced on all {len(images)} image(s)")
    else:
        print(f"[info] anchors  : {len(have)}/{len(folders)} object folder(s) have an anchor.txt")
        for folder in have:
            print(f"                  {folder}  ({len(folder_anchors[folder])} anchors, "
                  f"{len(folders[folder])} image(s))")

    if missing:
        if args.object_height is not None or args.distance is not None:
            how = "--object-height" if args.object_height is not None else "--distance"
            print(f"[info] {len(missing)} folder(s) have no anchor.txt; {how} calibrates "
                  "them anyway")
        else:
            print(f"\n[warn] SKIPPING {len(missing)} object folder(s) - no anchor.txt in them:")
            for folder in missing:
                print(f"         {folder}")
            example = f"{have[0] / 'anchor.txt'}" if have else "dataset/<lighting>/<object>/anchor.txt"
            print(f"       Add an anchor.txt to each of those folders, same format as {example}:")
            print("         cup1 anchor (636, 663, 40cm)")
            print("         cup2 anchor (1097, 605, 100cm)")
            if not have:
                sys.exit("No object folder has an anchor.txt and no other calibration was "
                         "given; nothing to size.")
            print()
            images = [img for img in images if folder_anchors[img.parent]]

    ground_truth = {} if args.no_eval else load_ground_truth(dataset_root)
    object_sizes = {} if args.no_eval else load_object_sizes(dataset_root)
    print(f"[info] dataset  : {dataset_root} ({len(images)} images in "
          f"{len({i.parent for i in images})} folder(s)) x models {', '.join(model_keys)}")

    # One depth model at a time: build it, sweep the whole dataset, drop it.
    # ``--depth-model all`` is five checkpoints (Depth Anything V2 small/base/large
    # + MiDaS + Marigold); holding them all resident at once does not fit on a
    # 12 GB GPU, so each pipe is released (and VRAM freed) before the next loads.
    yolo_model = make_yolo(args.yolo)

    rows = []
    voided = []
    skipped = 0
    for model_key in model_keys:
        t0 = time.perf_counter()
        try:
            pipe = make_depth_pipe(model_key)
        except (SystemExit, Exception) as exc:  # one bad model must not kill the batch
            print(f"[warn] depth model '{model_key}' could not be loaded, skipping it: {exc}")
            continue
        print(f"[info] loaded   : depth model '{model_key}' in {time.perf_counter() - t0:.1f}s")
        try:
            for img in images:
                rel_parent = img.parent.relative_to(dataset_root)
                group = str(rel_parent).replace(os.sep, "_") or dataset_root.name
                true_distance, lighting = derive_truth(img, dataset_root)
                try:
                    image = Image.open(img).convert("RGB")
                except Exception as exc:
                    print(f"[skip] {img.name}: cannot read ({exc})")
                    skipped += 1
                    continue
                out_dir = args.out / model_key / rel_parent
                anchors = folder_anchors[img.parent]
                anchor_file = folder_anchor_file[img.parent]
                try:
                    rec = size_image(args, image, img, model_key, pipe,
                                     yolo_model, anchors, anchor_file, quiet=True)
                except PipelineSkip as exc:
                    print(f"[skip] {img.name} [{model_key}]: {exc}")
                    skipped += 1
                    continue
                save_outputs(out_dir, img.stem, image, rec["depth"], rec["results"], rec["meta"])
                # Only ever score the object this folder is about. ``--target``
                # wins when given, else the class is inferred from the object id
                # (a01 -> apple, b01/b02 -> bottle, bwl01 -> bowl, c01 -> cup).
                object_id = object_id_for(img, ground_truth.get(f"{lighting}/{img.name}")
                                          or ground_truth.get(img.name))
                expected = args.target or expected_class_for(object_id)
                picked = pick_middle(rec["results"], anchors, expected)
                if picked is None:
                    # The subject was never detected. VOID the row instead of
                    # scoring some other object's distance as if it were the
                    # subject - that is what inflated MAE/RMSE across the board.
                    found = ", ".join(sorted({str(r["label"]) for r in rec["results"]})) or "nothing"
                    reason = (f"no '{expected}' detected" if expected
                              else "no non-anchor object detected")
                    print(f"[void] {img.name} [{model_key}]: {reason} (found: {found})"
                          f" - voided, not scored", flush=True)
                    voided.append({
                        "model": model_key, "filename": img.name, "object_id": object_id,
                        "group": group, "lighting": lighting, "distance_true_m": true_distance,
                        "expected": expected or "", "found": found, "reason": reason,
                    })
                    continue
                row = eval_row(model_key, img, group,
                               None if args.no_eval else true_distance, lighting, picked,
                               ground_truth, anchor_file, object_sizes)
                rows.append(row)
                note = ""
                if row["z_err_cm"] is not None:
                    note = f" (true {row['distance_true_m'] * 100:.0f}cm, err {row['z_err_cm']:+.1f}cm)"
                size_note = ""
                if row["w_err_cm"] is not None or row["h_err_cm"] is not None:
                    parts = []
                    if row["w_err_cm"] is not None:
                        parts.append(f"W{row['w_err_cm']:+.1f}")
                    if row["h_err_cm"] is not None:
                        parts.append(f"H{row['h_err_cm']:+.1f}")
                    size_note = f" [{'/'.join(parts)}cm]"
                print(f"    {img.stem:<16}[{model_key:<8}] {(row['label'] or '-'):<9}"
                      f"Z={row['z_m'] * 100:6.1f}cm{note}{size_note}", flush=True)
        finally:
            pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows and not voided:
        sys.exit("Nothing could be sized; check the anchors, --conf and the dataset.")

    if not rows:
        # Every subject was voided: still write the report so the voided rows are
        # on record instead of the run failing with "nothing could be sized".
        print(f"\n[warn] nothing scored: all {len(voided)} (model, image) pair(s) were voided "
              f"because the subject object was never detected.")
        write_eval(args.out, rows, model_keys, voided)
    elif args.no_eval or not has_truth(rows):
        print_predictions_table(rows, model_keys)
        if args.no_eval:
            print("[info] --no-eval: predictions only; no evaluation written.")
        else:
            print("[info] no ground truth to compare against (no ground_truth.csv and no "
                  "distance in the filenames); printed predictions only.")
    else:
        write_eval(args.out, rows, model_keys, voided)
    print(f"\n[info] sized    : {len(rows)} (model, image) pairs; skipped {skipped}; "
          f"voided {len(voided)} (subject never detected - excluded from the evaluation)")
    return 0


def model_keys_from_spec(spec):
    """Resolve ``--depth-model`` into a list of keys ('all' expands, commas split)."""
    keys = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        keys.extend(BATCH_DEPTH_MODELS if part == "all" else [part])
    return keys or [DEFAULT_DEPTH_MODEL]


def main(argv=None):
    args = parse_args(argv)
    model_keys = model_keys_from_spec(args.depth_model)
    if args.calib == "inverse" and any(is_marigold(MODELS.get(k, k)) for k in model_keys):
        print("[warn] Marigold predicts direct depth (larger = farther), so --calib affine "
              "fits it far better than --calib inverse.\n"
              "       With 'all' the inverse fit will be badly wrong for the Marigold rows: "
              "use --calib affine, or drop Marigold from the list.")
    # --anchor / --anchor-file are resolved per image (see resolve_anchors), so a
    # per-folder anchor.txt only applies to the shots that live in that folder.
    if args.dataset is not None:
        return run_batch(args, model_keys)
    return run_single(args, model_keys)


if __name__ == "__main__":
    raise SystemExit(main())
