#!/usr/bin/env python3
"""Prepare a dataset's calibration anchors: click them, then label them.

Anchors are the fixed pixels that turn relative monocular depth into meters.
They live per object folder - ``dataset/<lighting>/<object>/anchor.txt``, lines
like ``cup1 anchor (636, 663, 40cm)`` - and there is no inheritance: an image
only ever uses the anchor.txt sitting in its own folder.

One file, three modes:

``pick`` (the default) walks the folders that contain images, serves one
reference frame per folder to a local browser page, and turns two clicks into
that folder's anchor.txt. The page supports wheel-zoom (zoom into the cup rim for
pixel accuracy), drag-to-pan, backtracking, and switching to another frame in the
same folder when one shot is blurrier than the rest. Everything else is
server-side: the browser never gets a filesystem path, it posts pixel pairs to a
local-only HTTP server that does the writing.

``label`` is the inverse: it reads each folder's anchor.txt and renders the
markers onto the actual frames, so you can eyeball whether each one still sits
on its cup after a re-shoot, keeping the labelled copies next to - but separate
from - ``dataset/`` (default ``anchordataset/``). A folder without an anchor.txt
is skipped as a whole and reported once. PIL only, so it runs instantly.

``adjust`` makes the lighting modes actually differ. The phone's auto exposure and
white balance flatten them in-camera: real `warm` frames come back almost as
neutral as `white`, and real `dim` frames are only slightly darker. This mode
rewrites every `warm` frame with a stronger yellow cast and every `dim` frame
darker (and flatter), mirroring the dataset into a copy tree (default
``adjusteddataset/``) or over the originals with ``--in-place``. Geometry is
untouched, so each folder's anchor.txt is copied along and the pipeline can run
on the adjusted tree straight away.

    python preprocessing.py                             # pick: dataset/ -> anchor.txt
    python preprocessing.py pick --overwrite --match "*/b1"
    python preprocessing.py pick --dist-cm 40,100 --names cup1,cup2
    python preprocessing.py pick --preview              # draw labelled copies too
    python preprocessing.py label                       # dataset/ -> anchordataset/
    python preprocessing.py label --image photo.jpg     # one file
    python preprocessing.py label --anchor 636,663,0.40 --anchor 1097,605,1.00
    python preprocessing.py label --anchor-file other/anchor.txt
    python preprocessing.py adjust                      # dataset/ -> adjusteddataset/
    python preprocessing.py adjust --only warm --warm-strength 0.9
    python preprocessing.py adjust --only dim --dim-brightness 0.45
    python preprocessing.py adjust --in-place           # rewrite the dataset itself

``pick`` runs when no mode word is given, so the old command lines keep working
(``--overwrite``, ``--match``, ... are read as pick flags).

pick keys: click = place point, Backspace = undo, Enter = save & next,
N = next frame, S = skip folder, Q = quit.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import mimetypes
import re
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# One colour per anchor, in file order. Chosen to pop against white/green/wood
# and to stay distinguishable for colour-blind viewers.
PALETTE = [
    (255, 64, 64),     # red
    (32, 214, 255),    # cyan
    (255, 196, 0),     # amber
    (170, 110, 255),   # violet
    (60, 230, 120),    # green
    (255, 110, 190),   # pink
]

_ANCHOR_RE = re.compile(
    r"^(?P<name>\S+)?\s*(?:anchor)?\s*\(\s*(?P<u>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<v>-?\d+(?:\.\d+)?)\s*,\s*(?P<dist>-?\d+(?:\.\d+)?)\s*(?P<unit>cm|m)?\s*\)",
    re.I,
)


def pillow():
    """Import Pillow on demand; only the labelling side really needs it."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:  # environment guard
        sys.exit("Missing dependency 'pillow'.\nInstall with: pip install pillow")
    return Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------- #
# shared: finding and parsing anchors                                         #
# --------------------------------------------------------------------------- #
def parse_anchor_file(path):
    """Return ``[{name, u, v, distance_m, raw}]`` parsed from one anchor file."""
    path = Path(path)
    if not path.is_file():
        sys.exit(f"Anchor file not found: {path}")
    anchors = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ANCHOR_RE.search(line)
        if not m:
            continue
        dist = float(m.group("dist"))
        if (m.group("unit") or "cm").lower() == "cm":
            dist /= 100.0
        anchors.append({
            "name": (m.group("name") or f"anchor{len(anchors) + 1}").rstrip(":"),
            "u": float(m.group("u")),
            "v": float(m.group("v")),
            "distance_m": dist,
            "raw": line,
        })
    if not anchors:
        sys.exit(f"No anchors parsed from {path}; expected 'name anchor (u, v, 40cm)' lines.")
    return anchors


def cli_anchors(args):
    """Parse repeated ``--anchor u,v,distance_m`` specs."""
    anchors = []
    for spec in args.anchor:
        try:
            u, v, dist = (float(x) for x in spec.replace(" ", "").split(","))
        except ValueError:
            sys.exit(f"--anchor expects 'u,v,distance_m', got '{spec}'")
        if dist <= 0:
            sys.exit(f"--anchor distance must be positive meters, got '{spec}'")
        anchors.append({"name": f"anchor{len(anchors) + 1}", "u": u, "v": v,
                        "distance_m": dist, "raw": spec})
    return anchors


def anchor_file_for(image_path):
    """The ``anchor.txt`` in the image's own folder, or ``None``.

    Anchors are per object folder; there is no inheritance from a parent, so a
    folder without the file is reported and skipped rather than borrowing
    another scene's pixels.
    """
    path = Path(image_path).resolve().parent / "anchor.txt"
    return path if path.is_file() else None


def resolve_anchors(args, image_path, cache=None):
    """``(anchors, source)`` for one image: --anchor > --anchor-file > its anchor.txt.

    ``source`` is the anchor file path (``"cli"`` for CLI anchors) or ``None`` when
    the image's folder has no anchor.txt.
    """
    if args.anchor:
        return cli_anchors(args), "cli"
    cache = {} if cache is None else cache
    if args.anchor_file:
        key = str(Path(args.anchor_file).resolve())
        if key not in cache:
            cache[key] = parse_anchor_file(args.anchor_file)
        return cache[key], Path(args.anchor_file)
    path = anchor_file_for(image_path)
    if path is not None:
        key = str(path.resolve())
        if key not in cache:
            cache[key] = parse_anchor_file(path)
        return cache[key], path
    return [], None


def find_images(root):
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"--dataset is not a directory: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# --------------------------------------------------------------------------- #
# label mode: draw the anchors onto the images                                #
# --------------------------------------------------------------------------- #
def add_label_args(p):
    src = p.add_argument_group("input")
    src.add_argument("--dataset", type=Path, default=_SCRIPT_DIR / "dataset",
                     help="Folder to scan recursively (default: ./dataset).")
    src.add_argument("--image", type=Path, default=None,
                     help="Label a single image instead of scanning the dataset.")
    src.add_argument("--anchor-file", type=Path, default=None, metavar="PATH",
                     help="Force ONE anchor file for every image (skips the per-folder "
                          "anchor.txt discovery).")
    src.add_argument("--anchor", action="append", default=[], metavar="U,V,DIST",
                     help="Anchor 'u,v,distance_m' given on the CLI. Repeat for more; "
                          "overrides --anchor-file and every anchor.txt.")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=_SCRIPT_DIR / "anchordataset",
                     help="Output folder (default: ./anchordataset). Kept separate from --dataset.")
    out.add_argument("--suffix", default="_anchors", help="Filename suffix (default: _anchors).")
    out.add_argument("--jpeg-quality", type=int, default=95, help="Quality for .jpg outputs (default: 95).")

    style = p.add_argument_group("style")
    style.add_argument("--radius", type=int, default=30, help="Marker ring radius in px (default: 30).")
    style.add_argument("--thickness", type=int, default=5, help="Stroke width in px (default: 5).")
    style.add_argument("--font-size", type=int, default=0, help="Label font size (0 = auto).")
    style.add_argument("--no-legend", dest="legend", action="store_false", help="Skip the legend panel.")
    style.add_argument("--no-baseline", dest="baseline", action="store_false",
                       help="Skip the dashed line joining the anchors.")
    p.set_defaults(legend=True, baseline=True)
    return p


def label_defaults(dataset, out):
    """A full label-mode namespace with its usual defaults, for in-process reuse."""
    p = argparse.ArgumentParser(add_help=False)
    return add_label_args(p).parse_args(["--dataset", str(dataset), "--out", str(out)])


def load_font(size):
    """A decent TrueType face if the system has one, else PIL's bitmap default."""
    _, _, ImageFont = pillow()
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def text_size(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top, left, top


def clamp_label(draw, xy, wh, img_size, pad=8):
    """Nudge a label box fully inside the frame."""
    x, y = xy
    w, h = wh
    x = max(pad, min(x, img_size[0] - w - pad))
    y = max(pad, min(y, img_size[1] - h - pad))
    return int(x), int(y)


def draw_marker(draw, u, v, radius, thickness, colour, font, label, sublabel, img_size):
    """Ring + crosshair + a two-line label chip next to the anchor pixel."""
    u, v = float(u), float(v)
    box = [u - radius, v - radius, u + radius, v + radius]
    draw.ellipse(box, outline=colour, width=thickness)
    draw.ellipse([u - thickness // 2, v - thickness // 2,
                  u + thickness // 2, v + thickness // 2], fill=colour)

    arm = int(radius * 1.7)
    for a, b in (((u - arm, v), (u - radius, v)), ((u + radius, v), (u + arm, v)),
                 ((u, v - arm), (u, v - radius)), ((u, v + radius), (u, v + arm))):
        draw.line([a, b], fill=colour, width=max(2, thickness - 2))

    w1, h1, _, _ = text_size(draw, label, font)
    w2, h2, _, _ = text_size(draw, sublabel, font)
    pad_x, pad_y, gap = 8, 5, 3
    chip_w = max(w1, w2) + 2 * pad_x
    chip_h = h1 + h2 + 2 * pad_y + gap

    # Prefer the right-hand side of the ring; flip left near the right edge.
    side = 1 if u + radius + chip_w + 12 < img_size[0] else -1
    cx = u + side * (radius + 10)
    cy = v - chip_h / 2
    cx, cy = clamp_label(draw, (cx if side > 0 else cx - chip_w, cy), (chip_w, chip_h), img_size)

    draw.line([(u + side * radius, v), (cx if side > 0 else cx + chip_w, cy + chip_h / 2)],
              fill=colour, width=max(2, thickness - 2))
    draw.rounded_rectangle([cx, cy, cx + chip_w, cy + chip_h], radius=6,
                           fill=(0, 0, 0), outline=colour, width=2)
    draw.text((cx + pad_x, cy + pad_y), label, fill=(255, 255, 255), font=font)
    draw.text((cx + pad_x, cy + pad_y + h1 + gap), sublabel, fill=colour, font=font)


def draw_legend(draw, anchors, font):
    title = "CALIBRATION ANCHORS"
    rows = []
    for i, a in enumerate(anchors):
        rows.append((PALETTE[i % len(PALETTE)],
                     f"{a['name']}  ({a['u']:.0f}, {a['v']:.0f})  {a['distance_m']:.2f} m"))
    tw, th, _, _ = text_size(draw, title, font)
    rw = max([tw] + [text_size(draw, r, font)[0] for _, r in rows])
    rh = th + 8
    pad, line_h = 12, rh
    box_w, box_h = rw + 2 * pad + 24, pad * 2 + line_h * (len(rows) + 1)
    x0, y0 = pad, pad
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=8,
                           fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    draw.text((x0 + pad, y0 + pad), title, fill=(255, 255, 255), font=font)
    for i, (colour, row) in enumerate(rows):
        y = y0 + pad + line_h * (i + 1)
        draw.rectangle([x0 + pad, y + rh * 0.20, x0 + pad + 12, y + rh * 0.20 + th * 0.7],
                       fill=colour)
        draw.text((x0 + pad + 22, y), row, fill=colour, font=font)


def draw_baseline(draw, anchors, thickness):
    """Dashed line between the first and last anchor (the calibration span)."""
    if len(anchors) < 2:
        return
    (u1, v1), (u2, v2) = (anchors[0]["u"], anchors[0]["v"]), (anchors[-1]["u"], anchors[-1]["v"])
    span = max(1.0, ((u2 - u1) ** 2 + (v2 - v1) ** 2) ** 0.5)
    dash, gap, t = 18, 12, 0.0
    while t < span:
        a = t / span
        b = min(1.0, (t + dash) / span)
        draw.line([(u1 + (u2 - u1) * a, v1 + (v2 - v1) * a),
                   (u1 + (u2 - u1) * b, v1 + (v2 - v1) * b)],
                  fill=(255, 255, 255), width=max(1, thickness // 3))
        t += dash + gap


def label_image(src, dst, anchors, args):
    Image, ImageDraw, _ = pillow()
    with Image.open(src) as im:
        img = im.convert("RGB")
    img_size = img.size
    draw = ImageDraw.Draw(img)
    scale = max(0.6, min(img_size) / 1080.0)
    radius = max(8, int(round(args.radius * scale)))
    thickness = max(2, int(round(args.thickness * scale)))
    font_size = args.font_size or max(16, int(round(min(img_size) / 42)))
    font = load_font(font_size)

    if args.baseline:
        draw_baseline(draw, anchors, thickness)

    for i, a in enumerate(anchors):
        colour = PALETTE[i % len(PALETTE)]
        in_frame = 0 <= a["u"] < img_size[0] and 0 <= a["v"] < img_size[1]
        label = f"{a['name']}  {a['distance_m']:.2f} m"
        sub = f"px ({a['u']:.0f}, {a['v']:.0f})" + ("" if in_frame else "  OUT OF FRAME")
        draw_marker(draw, a["u"], a["v"], radius, thickness, colour, font, label, sub, img_size)
        if not in_frame:
            print(f"    [warn] {a['name']} ({a['u']:.0f},{a['v']:.0f}) lies outside "
                  f"{img_size[0]}x{img_size[1]}")

    if args.legend:
        draw_legend(draw, anchors, font)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        img.save(dst, quality=args.jpeg_quality)
    else:
        img.save(dst)
    return img_size


def output_path(src, out_root, dataset_root, suffix):
    if dataset_root is None:
        return out_root / f"{src.stem}{suffix}{src.suffix or '.png'}"
    try:
        rel = src.relative_to(dataset_root)
    except ValueError:
        rel = Path(src.name)
    return out_root / rel.with_name(f"{src.stem}{suffix}{src.suffix or '.png'}")


def run_label(args):
    if args.image:
        if not args.image.is_file():
            sys.exit(f"--image not found: {args.image}")
        images, dataset_root = [args.image], None
    else:
        if not args.dataset.is_dir():
            sys.exit(f"--dataset is not a directory: {args.dataset}")
        images, dataset_root = find_images(args.dataset), args.dataset
        if not images:
            sys.exit(f"No images (png/jpg/webp/...) found under {args.dataset}")

    target_out = args.out.resolve()
    if dataset_root is not None and target_out == Path(dataset_root).resolve():
        sys.exit("--out must differ from --dataset; keep labelled copies separate.")

    # One anchor.txt per object folder. A folder without one is skipped as a
    # whole and reported once, never filled in from a parent folder.
    cache = {}
    folders = {}
    for src in images:
        folders.setdefault(src.parent, []).append(src)
    resolved = {}
    for folder, imgs in folders.items():
        resolved[folder] = resolve_anchors(args, imgs[0], cache)
    have = sorted(f for f in folders if resolved[f][0])
    missing = sorted(f for f in folders if not resolved[f][0])

    if args.anchor:
        anchors = cli_anchors(args)
        print(f"Forced CLI anchors ({len(anchors)}):")
        for a in anchors:
            print(f"  {a['name']:<8} ({a['u']:>7.1f}, {a['v']:>7.1f})  {a['distance_m']:.2f} m")
    else:
        print(f"Anchor files: {len(have)}/{len(folders)} object folder(s) have an anchor.txt")
        for folder in have:
            anchors, source = resolved[folder]
            print(f"  {source}  ({len(anchors)} anchors, {len(folders[folder])} image(s))")
            for a in anchors:
                print(f"    {a['name']:<8} ({a['u']:>7.1f}, {a['v']:>7.1f})  {a['distance_m']:.2f} m")

    if missing:
        print(f"\n[warn] SKIPPING {len(missing)} object folder(s) - no anchor.txt in them:")
        for folder in missing:
            print(f"         {folder}")
        example = str(Path(have[0] if have else args.dataset) / "anchor.txt")
        print(f"       Add an anchor.txt to each, same format as {example}:")
        print("         cup1 anchor (636, 663, 40cm)")
        print("         cup2 anchor (1097, 605, 100cm)")
        if not have:
            sys.exit("No object folder has an anchor.txt; nothing to label.")

    images = [img for img in images if resolved[img.parent][0]]
    print(f"\nLabelling {len(images)} image(s) -> {args.out}")
    manifest = {"out": str(args.out.resolve()),
                "anchor_files": {str(resolved[f][1]): resolved[f][0] for f in have},
                "images": []}

    wrote = 0
    for src in images:
        anchors, source = resolved[src.parent]
        dst = output_path(src, args.out, dataset_root, args.suffix)
        size = label_image(src, dst, anchors, args)
        manifest["images"].append({"source": str(src), "output": str(dst),
                                   "anchor_file": str(source) if source else "cli",
                                   "anchors": anchors,
                                   "width": size[0], "height": size[1]})
        print(f"  {src} -> {dst}")
        wrote += 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "anchors.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nDone: {wrote} image(s), manifest -> {args.out / 'anchors.json'}")
    return 0


# --------------------------------------------------------------------------- #
# adjust mode: exaggerate the warm / dim light the camera auto-corrected away #
# --------------------------------------------------------------------------- #
LIGHTINGS = ("warm", "dim")


def add_adjust_args(p):
    src = p.add_argument_group("input")
    src.add_argument("--dataset", type=Path, default=_SCRIPT_DIR / "dataset",
                     help="Dataset root to scan recursively (default: ./dataset).")
    src.add_argument("--match", default=None, metavar="GLOB",
                     help="Only folders whose relative path matches this glob, e.g. '*/b1'.")
    src.add_argument("--only", default="warm,dim", metavar="WHICH",
                     help="Lightings to touch, comma separated from warm,dim "
                          "(default: warm,dim; `nv` is never touched).")
    src.add_argument("--warm-token", default="warm", metavar="TOKEN[,TOKEN]",
                     help="Folder/filename token marking warm light (default: warm).")
    src.add_argument("--dim-token", default="dim", metavar="TOKEN[,TOKEN]",
                     help="Folder/filename token marking dim light (default: dim).")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=_SCRIPT_DIR / "adjusteddataset",
                     help="Output folder (default: ./adjusteddataset), mirroring --dataset.")
    out.add_argument("--in-place", dest="in_place", action="store_true",
                     help="Rewrite the images inside --dataset instead of a copy tree.")
    out.add_argument("--jpeg-quality", type=int, default=95,
                     help="Quality for .jpg outputs (default: 95).")
    out.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="Print what would change and write nothing.")

    warm = p.add_argument_group("warm")
    warm.add_argument("--warm-strength", type=float, default=0.65,
                      help="Warm cast strength: 0 = untouched, 1 = strong (default: 0.65).")

    dim = p.add_argument_group("dim")
    dim.add_argument("--dim-brightness", type=float, default=0.55,
                     help="Per-channel brightness gain for dim frames (default: 0.55).")
    dim.add_argument("--dim-contrast", type=float, default=0.9,
                     help="Contrast factor applied after darkening (default: 0.9).")
    p.set_defaults(in_place=False, dry_run=False)
    return p


def adjust_args(args):
    """Validate and normalise the adjust options, exiting on bad input."""
    args.only = [w.strip().lower() for w in args.only.split(",") if w.strip()]
    unknown = [w for w in args.only if w not in LIGHTINGS]
    if unknown:
        sys.exit(f"--only expects warm/dim, got '{', '.join(unknown)}'")
    if not args.only:
        sys.exit("--only is empty; choose warm, dim, or both")
    if not 0 <= args.warm_strength <= 3:
        sys.exit("--warm-strength must be between 0 and 3")
    if not 0 < args.dim_brightness <= 2:
        sys.exit("--dim-brightness must be greater than 0 and at most 2")
    if not 0 <= args.dim_contrast <= 3:
        sys.exit("--dim-contrast must be between 0 and 3")
    args.warm_tokens = {t.strip().lower() for t in args.warm_token.split(",") if t.strip()}
    args.dim_tokens = {t.strip().lower() for t in args.dim_token.split(",") if t.strip()}
    return args


def lighting_of(src, dataset_root, warm_tokens, dim_tokens):
    """``'warm'`` / ``'dim'`` / ``None`` for one image, from its path alone.

    Any parent folder name or filename token (split on ``_ - . space``) that is in
    the warm/dim token sets decides it, so both ``warm/b1/x.png`` and
    ``b1/bottle01_d060_warm.jpg`` are recognised.
    """
    src = Path(src)
    try:
        rel = src.resolve().relative_to(Path(dataset_root).resolve())
    except ValueError:
        rel = Path(src.name)
    tokens = {part.lower() for part in rel.parts[:-1]}
    tokens.update(t for t in re.split(r"[_\-. ]+", rel.stem.lower()) if t)
    if tokens & warm_tokens:
        return "warm"
    if tokens & dim_tokens:
        return "dim"
    return None


def mirror_dir(folder, out_root, dataset_root):
    """The copy-tree folder mirroring ``folder`` under ``out_root``."""
    try:
        rel = Path(folder).resolve().relative_to(Path(dataset_root).resolve())
    except ValueError:
        rel = Path(Path(folder).name)
    return Path(out_root) / rel


def gain_lut(factor):
    """Per-channel lookup table multiplying every value by ``factor``."""
    return [min(255, max(0, int(round(v * factor)))) for v in range(256)]


def scale_channels(img, factors):
    Image, _, _ = pillow()
    return Image.merge("RGB", [c.point(gain_lut(f)) for c, f in zip(img.split(), factors)])


def apply_warm(img, strength):
    """Yellow cast: lift red, nudge green, pull blue down, saturate a little."""
    from PIL import ImageEnhance
    out = scale_channels(img, (1 + 0.28 * strength,
                               1 + 0.07 * strength,
                               1 - 0.24 * strength))
    return ImageEnhance.Color(out).enhance(1 + 0.15 * strength)


def apply_dim(img, brightness, contrast):
    """Low light: everything darker, and a touch flatter."""
    from PIL import ImageEnhance
    out = scale_channels(img, (brightness, brightness, brightness))
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    return out


def save_image(img, dst, jpeg_quality):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        img.save(dst, quality=jpeg_quality)
    else:
        img.save(dst)


def run_adjust(args):
    adjust_args(args)
    Image, _, _ = pillow()
    if not args.dataset.is_dir():
        sys.exit(f"--dataset is not a directory: {args.dataset}")
    images = find_images(args.dataset)
    if not images:
        sys.exit(f"No images (png/jpg/webp/...) found under {args.dataset}")
    root = args.dataset.resolve()
    if not args.in_place and args.out.resolve() == root:
        sys.exit("--out must differ from --dataset; use --in-place to rewrite the originals.")

    plan = []
    for src in images:
        if args.match and not fnmatch.fnmatch(mirror_dir(src.parent, "", root).as_posix(), args.match):
            continue
        lighting = lighting_of(src, root, args.warm_tokens, args.dim_tokens)
        if lighting in args.only:
            plan.append((src, lighting))

    if not plan:
        print(f"No {'/'.join(args.only)} images found under {args.dataset}"
              + (f" matching {args.match}" if args.match else ""))
        return 0

    verb = "Would adjust" if args.dry_run else "Adjusting"
    print(f"{verb} {len(plan)} image(s)"
          + (" in place" if args.in_place else f" -> {args.out}"))
    counts = {name: 0 for name in LIGHTINGS}
    manifest = {
        "out": str(root if args.in_place else args.out.resolve()),
        "in_place": args.in_place,
        "settings": {"warm_strength": args.warm_strength,
                     "dim_brightness": args.dim_brightness,
                     "dim_contrast": args.dim_contrast},
        "images": [],
    }

    for src, lighting in plan:
        dst = src if args.in_place else output_path(src, args.out, args.dataset, "")
        if not args.dry_run:
            with Image.open(src) as im:
                img = im.convert("RGB")
            if lighting == "warm":
                img = apply_warm(img, args.warm_strength)
            else:
                img = apply_dim(img, args.dim_brightness, args.dim_contrast)
            save_image(img, dst, args.jpeg_quality)
        print(f"  [{lighting}] {src} -> {dst}")
        counts[lighting] += 1
        manifest["images"].append({"source": str(src), "output": str(dst),
                                   "lighting": lighting})

    # Geometry is untouched, so the per-folder calibration carries over verbatim
    # and the adjusted tree can be fed straight to test_pipeline.py --dataset.
    anchors_copied = 0
    if not args.in_place and not args.dry_run:
        for folder in sorted({src.parent for src, _ in plan}):
            anchor = folder / "anchor.txt"
            if not anchor.is_file():
                continue
            target = mirror_dir(folder, args.out, args.dataset) / "anchor.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(anchor.read_bytes())
            anchors_copied += 1

    print(f"\nWarm: {counts['warm']}  Dim: {counts['dim']}"
          + (f"  anchor.txt copied: {anchors_copied}" if anchors_copied else ""))
    if args.dry_run:
        print("Dry run: nothing written.")
        return 0

    manifest_path = (args.dataset if args.in_place else args.out) / "adjusted.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest -> {manifest_path}")
    if not args.in_place:
        print(f"Run the pipeline on it with: --dataset {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# pick mode: click the anchors, write each folder's anchor.txt                #
# --------------------------------------------------------------------------- #
def add_pick_args(p):
    p.add_argument("--dataset", type=Path, default=_SCRIPT_DIR / "dataset",
                   help="Dataset root to walk (default: ./dataset).")
    p.add_argument("--match", default=None, metavar="GLOB",
                   help="Only folders whose relative path matches this glob, e.g. '*/b1'.")
    p.add_argument("--overwrite", action="store_true",
                   help="Also redo folders that already have an anchor.txt.")
    p.add_argument("--names", default="cup1,cup2",
                   help="Anchor names, comma separated (default: cup1,cup2).")
    p.add_argument("--dist-cm", default="40,100", metavar="CM,CM",
                   help="True camera distance in cm for each anchor (default: 40,100).")
    p.add_argument("--frame", type=int, default=0, metavar="N",
                   help="Start on this frame index inside each folder (default: 0).")
    p.add_argument("--port", type=int, default=8765, help="Local HTTP port (default: 8765).")
    p.add_argument("--no-open", dest="open_browser", action="store_false",
                   help="Do not launch a browser automatically; just print the URL.")
    p.add_argument("--preview", dest="preview", action="store_true",
                   help="Afterwards draw the labelled copies (same as 'label').")
    p.add_argument("--out", type=Path, default=_SCRIPT_DIR / "anchordataset",
                   help="Preview output folder for --preview (default: ./anchordataset).")
    p.set_defaults(open_browser=True, preview=False)
    return p


def pick_args(args):
    """Normalise the comma-separated pick options, exiting on bad input."""
    args.names = [n.strip() for n in args.names.split(",") if n.strip()]
    try:
        args.dist_cm = [float(x) for x in args.dist_cm.split(",")]
    except ValueError:
        sys.exit(f"--dist-cm expects comma-separated numbers, got '{args.dist_cm}'")
    if len(args.names) != len(args.dist_cm):
        sys.exit(f"--names ({len(args.names)}) and --dist-cm ({len(args.dist_cm)}) "
                 "must have the same length")
    if not args.names:
        sys.exit("--names is empty; need at least one anchor")
    if any(d <= 0 for d in args.dist_cm):
        sys.exit("--dist-cm values must be positive")
    return args


def find_folders(args):
    """``[{folder, rel, images, has_anchor}, ...]`` - folders that contain images."""
    root = args.dataset
    if not root.is_dir():
        sys.exit(f"--dataset is not a directory: {root}")
    folders = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            folders.setdefault(p.parent, []).append(p)
    resolved = root.resolve()
    picked = []
    for folder, images in sorted(folders.items()):
        try:
            rel = folder.resolve().relative_to(resolved).as_posix()
        except ValueError:
            rel = folder.name
        if args.match and not fnmatch.fnmatch(rel, args.match):
            continue
        picked.append({
            "folder": folder,
            "rel": rel or ".",
            "images": images,
            "has_anchor": (folder / "anchor.txt").is_file(),
        })
    return picked


def fmt_cm(value):
    return f"{value:g}cm"


def anchor_lines(points, names, dist_cm):
    return [f"{name} anchor ({int(round(u))}, {int(round(v))}, {fmt_cm(d)})"
            for name, (u, v), d in zip(names, points, dist_cm)]


def write_anchor_file(folder, points, names, dist_cm):
    path = Path(folder) / "anchor.txt"
    path.write_text("\n".join(anchor_lines(points, names, dist_cm)) + "\n")
    return path


class Session:
    def __init__(self, args, folders):
        self.args = args
        self.all_folders = folders
        self.queue = [f for f in folders if args.overwrite or not f["has_anchor"]]
        self.skipped_existing = [f for f in folders if not args.overwrite and f["has_anchor"]]
        self.index = 0
        self.frame = {id(f): min(args.frame, len(f["images"]) - 1) for f in folders}
        self.written = []
        self.skipped = []
        self.done = threading.Event()
        self.lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------- #
    def _maybe_finish(self):
        """Queue exhausted -> shut the server down, after a beat.

        The delay matters: the page fetches /api/state once more to render the
        summary, so tearing the server down inside the request handler would
        kill that fetch.
        """
        if self.index >= len(self.queue) and not self.done.is_set():
            threading.Timer(1.5, self.done.set).start()

    # -- current ---------------------------------------------------------- #
    def current(self):
        if self.index >= len(self.queue):
            return None
        return self.queue[self.index]

    def state(self):
        entry = self.current()
        if entry is None:
            return {
                "done": True,
                "written": [{"folder": str(p), "lines": lines} for p, lines in self.written],
                "skipped": self.skipped,
            }
        frame = self.frame[id(entry)]
        return {
            "done": False,
            "index": self.index,
            "total": len(self.queue),
            "folder": str(entry["folder"]),
            "rel": entry["rel"],
            "anchor_file": str(entry["folder"] / "anchor.txt"),
            "frame": frame,
            "frame_count": len(entry["images"]),
            "image_name": entry["images"][frame].name,
            "image_url": f"/api/image?i={self.index}&f={frame}",
            "names": self.args.names,
            "dist_cm": self.args.dist_cm,
            "written": len(self.written),
            "existing": self._existing(entry),
        }

    def _existing(self, entry):
        path = Path(entry["folder"]) / "anchor.txt"
        if not path.is_file():
            return None
        return path.read_text().splitlines()

    # -- actions ---------------------------------------------------------- #
    def save(self, points):
        with self.lock:
            entry = self.current()
            if entry is None:
                return {"ok": False, "error": "no folder left"}
            if len(points) != len(self.args.names):
                return {"ok": False, "error":
                        f"expected {len(self.args.names)} points, got {len(points)}"}
            for u, v in points:
                if not (0 <= u <= 100000 and 0 <= v <= 100000):
                    return {"ok": False, "error": f"point out of range: {u},{v}"}
            path = write_anchor_file(entry["folder"], points, self.args.names, self.args.dist_cm)
            self.written.append((path, anchor_lines(points, self.args.names, self.args.dist_cm)))
            self.index += 1
            self._maybe_finish()
            return {"ok": True, "path": str(path),
                    "lines": anchor_lines(points, self.args.names, self.args.dist_cm)}

    def skip(self):
        with self.lock:
            entry = self.current()
            if entry is None:
                return {"ok": False, "error": "no folder left"}
            self.skipped.append(str(entry["folder"]))
            self.index += 1
            self._maybe_finish()
            return {"ok": True}

    def next_frame(self):
        with self.lock:
            entry = self.current()
            if entry is None:
                return {"ok": False, "error": "no folder left"}
            self.frame[id(entry)] = (self.frame[id(entry)] + 1) % len(entry["images"])
            return {"ok": True, "frame": self.frame[id(entry)]}

    def finish(self):
        self.index = len(self.queue)
        self.done.set()
        return {"ok": True}


def image_meta(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return list(im.size)
    except Exception:
        return None


def make_handler(session):
    from http.server import BaseHTTPRequestHandler

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # keep the terminal readable
            pass

        # -- replies ------------------------------------------------------ #
        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        # -- GET ---------------------------------------------------------- #
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(session.state())
            if path == "/api/image":
                return self._image()
            return self._send(b"not found", "text/plain", 404)

        def _image(self):
            entry = session.current()
            if entry is None:
                return self._send(b"done", "text/plain", 404)
            frame = session.frame[id(entry)]
            src = entry["images"][frame]
            try:
                data = src.read_bytes()
            except OSError as exc:
                return self._send(str(exc).encode(), "text/plain", 500)
            ctype = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
            size = image_meta(src)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if size:
                self.send_header("X-Image-Size", f"{size[0]}x{size[1]}")
            self.end_headers()
            self.wfile.write(data)

        # -- POST --------------------------------------------------------- #
        def do_POST(self):
            path = self.path.split("?")[0]
            body = self._body()
            if path == "/api/save":
                points = body.get("points") or []
                try:
                    points = [(float(p[0]), float(p[1])) for p in points]
                except (TypeError, ValueError, IndexError):
                    return self._json({"ok": False, "error": "bad points"}, 400)
                return self._json(session.save(points))
            if path == "/api/skip":
                return self._json(session.skip())
            if path == "/api/next-frame":
                return self._json(session.next_frame())
            if path == "/api/quit":
                self._json(session.finish())
                return
            return self._send(b"not found", "text/plain", 404)

    return _H


def serve(session, port):
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def open_browser(url):
    """WSL-friendly: try the Windows side first, then XDG, then webbrowser."""
    for cmd in (["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                 "-NoProfile", "-Command", f"Start-Process '{url}'"],
                ["explorer.exe", url],
                ["xdg-open", url]):
        try:
            if cmd[0].startswith("/") and not Path(cmd[0]).exists():
                continue
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            continue
    try:
        return webbrowser.open(url)
    except Exception:
        return False


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>anchor picker</title>
<style>
  :root { --bg:#0f1117; --fg:#e8e8ea; --dim:#8b8fa3; --accent:#5ee0c0; --warn:#ffb454; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 ui-sans-serif,system-ui,"Segoe UI",sans-serif; height:100vh; display:flex; flex-direction:column; }
  header { padding:10px 16px; border-bottom:1px solid #23262f; display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.02em; }
  .box { color:var(--dim); }
  .box b { color:var(--fg); font-weight:600; }
  #rel { color:var(--accent); font-weight:600; }
  main { flex:1; position:relative; overflow:hidden; cursor:crosshair; background:#07080b; }
  #wrap { position:absolute; left:0; top:0; transform-origin:0 0; will-change:transform; }
  #img { display:block; image-rendering:auto; user-select:none; -webkit-user-drag:none; }
  .marker { position:absolute; border-radius:50%; border:2px solid var(--c);
            box-shadow:0 0 12px var(--c); pointer-events:none;
            transform:translate(-50%,-50%); }
  .marker span { position:absolute; left:50%; top:100%; transform:translate(-50%,4px);
                 background:#000c; color:var(--c); border:1px solid var(--c); border-radius:4px;
                 padding:1px 5px; font-size:11px; white-space:nowrap; font-variant-numeric:tabular-nums; }
  footer { border-top:1px solid #23262f; padding:10px 16px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  button { background:#1b1e26; color:var(--fg); border:1px solid #2e323d; border-radius:7px;
           padding:7px 13px; font:inherit; cursor:pointer; }
  button:hover:not(:disabled) { border-color:#4a5060; background:#222634; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  button.primary { background:var(--accent); color:#06120f; border-color:var(--accent); font-weight:700; }
  #status { margin-left:auto; color:var(--dim); font-variant-numeric:tabular-nums; }
  #err { color:var(--warn); }
  #readout { color:var(--accent); font-family:ui-monospace,monospace; }
  #done { padding:40px; overflow:auto; }
  #done pre { background:#151821; padding:14px; border-radius:8px; overflow:auto; }
  kbd { background:#1b1e26; border:1px solid #2e323d; border-radius:4px; padding:0 5px; font-size:12px; }
</style></head><body>
<header>
  <h1>anchor picker</h1>
  <span class="box"><b id="rel">-</b> <span id="prog"></span></span>
  <span class="box">frame <b id="frame">1/1</b> <span id="imgname"></span></span>
  <span class="box">write to <b id="target">-</b></span>
</header>
<main id="view">
  <div id="wrap"><img id="img" alt=""><div id="overlay"></div></div>
</main>
<footer>
  <button id="undo">Undo</button>
  <button id="clear">Clear</button>
  <button id="nextframe">Next frame (N)</button>
  <button id="skip">Skip folder (S)</button>
  <button id="save" class="primary" disabled>Save &amp; next (Enter)</button>
  <button id="quit">Finish (Q)</button>
  <span id="readout">-</span>
  <span id="err"></span>
  <span id="status"></span>
</footer>
<script>
const COLORS = ['#ff4040','#20d6ff','#ffc400','#aa6eff','#3ce678','#ff6ebe'];
const view = document.getElementById('view');
const wrap = document.getElementById('wrap');
const img = document.getElementById('img');
const overlay = document.getElementById('overlay');
const saveBtn = document.getElementById('save');
const readout = document.getElementById('readout');
const errBox = document.getElementById('err');
let state = null, points = [], scale = 1, tx = 0, ty = 0;
let dragging = false, moved = false, lastX = 0, lastY = 0;

function setErr(msg) { errBox.textContent = msg || ''; }
function fit() {
  const nw = img.naturalWidth, nh = img.naturalHeight;
  if (!nw) return;
  scale = Math.min(view.clientWidth / nw, view.clientHeight / nh) * 0.98;
  tx = (view.clientWidth - nw * scale) / 2;
  ty = (view.clientHeight - nh * scale) / 2;
  apply();
}
function apply() {
  wrap.style.transform = `translate(${tx}px,${ty}px) scale(${scale})`;
  wrap.style.width = img.naturalWidth + 'px';
  wrap.style.height = img.naturalHeight + 'px';
  for (const m of overlay.children) m.style.transform =
    `translate(-50%,-50%) scale(${1 / Math.max(scale, 1e-6)})`;
}
function renderMarkers() {
  overlay.innerHTML = '';
  points.forEach((p, i) => {
    const d = document.createElement('div');
    d.className = 'marker';
    d.style.cssText = `left:${p[0]}px;top:${p[1]}px;width:${26}px;height:${26}px;
      border-color:${COLORS[i % COLORS.length]};--c:${COLORS[i % COLORS.length]}`;
    d.style.borderColor = COLORS[i % COLORS.length];
    d.style.setProperty('--c', COLORS[i % COLORS.length]);
    d.innerHTML = `<span>${state.names[i]} ${Math.round(p[0])},${Math.round(p[1])}</span>`;
    overlay.appendChild(d);
  });
  apply();
  saveBtn.disabled = points.length !== state.names.length;
  document.getElementById('undo').disabled = !points.length;
  document.getElementById('clear').disabled = !points.length;
}
function toImage(clientX, clientY) {
  const r = img.getBoundingClientRect();
  return [(clientX - r.left) * img.naturalWidth / r.width,
          (clientY - r.top) * img.naturalHeight / r.height];
}
async function loadState() {
  const r = await fetch('/api/state');
  state = await r.json();
  setErr('');
  if (state.done) return showDone();
  document.getElementById('rel').textContent = state.rel;
  document.getElementById('prog').textContent = `(${state.index + 1}/${state.total})`;
  document.getElementById('frame').textContent = `${state.frame + 1}/${state.frame_count}`;
  document.getElementById('imgname').textContent = state.image_name;
  document.getElementById('target').textContent = state.anchor_file;
  points = [];
  errBox.textContent = state.existing ? 'folder had an anchor.txt - overwriting' : '';
  img.onload = () => { overlay.style.width = '0'; fit(); renderMarkers(); };
  img.src = state.image_url + '&t=' + Date.now();
  document.getElementById('undo').disabled = true;
  document.getElementById('clear').disabled = true;
  saveBtn.disabled = true;
  document.getElementById('status').textContent = `written ${state.written}`;
}
function showDone() {
  view.innerHTML = `<div id="done"><h2>All folders done</h2>
    <p>Wrote ${state.written.length} anchor.txt file(s). You can close this tab.</p>
    <pre>${state.written.map(w => w.folder + '\n  ' + w.lines.join('\n  ')).join('\n\n') || '(none)'}</pre>
    ${state.skipped.length ? '<p>Skipped: ' + state.skipped.join(', ') + '</p>' : ''}</div>`;
  document.querySelector('footer').style.display = 'none';
  document.querySelector('header').style.display = 'none';
}
// --- interaction ----------------------------------------------------------
view.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!img.naturalWidth) return;
  const factor = Math.exp(-e.deltaY * 0.0016);
  const ns = Math.min(12, Math.max(0.03, scale * factor));
  const k = ns / scale;
  const r = view.getBoundingClientRect();
  tx = (e.clientX - r.left) - ((e.clientX - r.left) - tx) * k;
  ty = (e.clientY - r.top) - ((e.clientY - r.top) - ty) * k;
  scale = ns;
  apply();
}, { passive: false });

view.addEventListener('mousedown', (e) => {
  dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY;
  if (e.button === 1 || e.button === 2) e.preventDefault();
});
window.addEventListener('mousemove', (e) => {
  if (img.naturalWidth) {
    const [u, v] = toImage(e.clientX, e.clientY);
    const inside = u >= 0 && v >= 0 && u < img.naturalWidth && v < img.naturalHeight;
    readout.textContent = inside ? `x ${Math.round(u)}  y ${Math.round(v)}` : '-';
  }
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
  tx += dx; ty += dy; lastX = e.clientX; lastY = e.clientY;
  if (moved) apply();
});
window.addEventListener('mouseup', (e) => {
  const wasDragging = dragging, didMove = moved;
  dragging = false;
  if (!wasDragging || didMove) return;
  if (e.target !== img && e.target !== overlay) return;
  if (!state || state.done) return;
  if (e.button !== 0) return;
  if (points.length >= state.names.length) return;
  const [u, v] = toImage(e.clientX, e.clientY);
  if (u < 0 || v < 0 || u >= img.naturalWidth || v >= img.naturalHeight) return;
  points.push([u, v]);
  renderMarkers();
});
view.addEventListener('contextmenu', (e) => e.preventDefault());

// --- actions --------------------------------------------------------------
async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                               body: JSON.stringify(body || {}) });
  return r.json();
}
async function doSave() {
  if (points.length !== state.names.length) return;
  const res = await post('/api/save', { points });
  if (!res.ok) { setErr(res.error || 'save failed'); return; }
  await loadState();
}
async function doSkip() { await post('/api/skip'); await loadState(); }
async function doNextFrame() { await post('/api/next-frame'); await loadState(); }
async function doQuit() { await post('/api/quit'); await loadState(); }

saveBtn.onclick = doSave;
document.getElementById('undo').onclick = () => { points.pop(); renderMarkers(); };
document.getElementById('clear').onclick = () => { points = []; renderMarkers(); };
document.getElementById('skip').onclick = doSkip;
document.getElementById('nextframe').onclick = doNextFrame;
document.getElementById('quit').onclick = doQuit;

window.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); if (!saveBtn.disabled) doSave(); }
  else if (e.key === 'Backspace') { e.preventDefault(); points.pop(); renderMarkers(); }
  else if (e.key === 'Escape') { points = []; renderMarkers(); }
  else if (e.key === 'n' || e.key === 'N') doNextFrame();
  else if (e.key === 's' || e.key === 'S') doSkip();
  else if (e.key === 'q' || e.key === 'Q') doQuit();
});
window.addEventListener('resize', () => { });
loadState();
</script></body></html>
"""



# --------------------------------------------------------------------------- #
# entry points                                                                #
# --------------------------------------------------------------------------- #
def run_pick(args):
    pick_args(args)
    folders = find_folders(args)
    if not folders:
        sys.exit(f"No image folders found under {args.dataset}"
                 + (f" matching {args.match}" if args.match else ""))

    session = Session(args, folders)
    print(f"Dataset: {args.dataset.resolve()}")
    print(f"Object folders with images: {len(folders)}")
    if session.skipped_existing:
        print(f"  already have anchor.txt (skipped, use --overwrite to redo): "
              f"{len(session.skipped_existing)}")
        for f in session.skipped_existing:
            print(f"    {f['rel']}")
    if not session.queue:
        sys.exit("Nothing to pick: every folder already has an anchor.txt. "
                 "Add --overwrite to redo them.")

    print(f"To label: {len(session.queue)} folder(s)")
    for f in session.queue:
        print(f"    {f['rel']}  ({len(f['images'])} image(s))")
    print("\nAnchors per folder: " + ", ".join(
        f"{n} = {d:g}cm" for n, d in zip(args.names, args.dist_cm)))

    httpd = serve(session, args.port)
    url = f"http://localhost:{args.port}/"
    print(f"\nOpen {url} and click the anchors in order "
          f"({', '.join(args.names)}).")
    if args.open_browser and open_browser(url):
        print("(browser launched)")
    print("Waiting for clicks... Ctrl+C here also quits.\n")

    try:
        session.done.wait()
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        httpd.shutdown()

    print(f"Wrote {len(session.written)} anchor.txt file(s):")
    for path, lines in session.written:
        print(f"  {path}")
        for line in lines:
            print(f"    {line}")
    if session.skipped:
        print(f"Skipped {len(session.skipped)} folder(s):")
        for f in session.skipped:
            print(f"  {f}")
    remaining = len(session.queue) - session.index
    if remaining > 0:
        print(f"\n{remaining} folder(s) were NOT written.")

    if args.preview and session.written:
        print("\nDrawing labelled copies ...")
        run_label(label_defaults(args.dataset, args.out))
    return 0


MODES = ("pick", "label", "adjust")


def build_parser():
    p = argparse.ArgumentParser(
        prog="preprocessing.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog="'pick' is assumed when no mode word is given.",
    )
    sub = p.add_subparsers(dest="mode", required=True, metavar="{pick,label,adjust}")

    pick = sub.add_parser(
        "pick", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="click the anchors in a browser page, write each folder's anchor.txt",
        description="Walk the dataset folder by folder, click each anchor in a local "
                    "browser page, and write that folder's anchor.txt.",
    )
    add_pick_args(pick)

    label = sub.add_parser(
        "label", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="draw every folder's anchors back onto its images",
        description="Read each folder's anchor.txt and render the markers onto the "
                    "actual frames, mirroring them into a separate labelled copy tree.",
    )
    add_label_args(label)

    adjust = sub.add_parser(
        "adjust", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="exaggerate the warm / dim lighting the camera auto-corrected away",
        description="Rewrite every warm frame with a stronger yellow cast and every "
                    "dim frame darker, so the lighting modes actually differ. Pixel "
                    "geometry is untouched and each folder's anchor.txt is copied along.",
    )
    add_adjust_args(adjust)
    return p


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in MODES and argv[0] not in ("-h", "--help")):
        argv.insert(0, "pick")
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "label":
        return run_label(args)
    if args.mode == "adjust":
        return run_adjust(args)
    return run_pick(args)


if __name__ == "__main__":
    raise SystemExit(main())
