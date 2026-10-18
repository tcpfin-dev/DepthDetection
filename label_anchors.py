#!/usr/bin/env python3
"""Draw the calibration anchors onto every dataset image.

Anchors live per object folder (``dataset/<lighting>/<object>/anchor.txt``, lines
like ``cup1 anchor (636, 663, 40cm)``): they are the fixed pixels that turn
relative monocular depth into meters. This script reads the anchor.txt in each
image's own folder, renders the markers on the actual frame so you can eyeball
whether each one still sits on its cup after a re-shoot, and keeps the labelled
copies in a folder next to - but separate from - ``dataset/`` (default:
``anchordataset/``), mirroring the dataset's sub-folders. A folder with no
anchor.txt is skipped as a whole and reported once, with a reminder to add one.

    python label_anchors.py                        # dataset/ -> anchordataset/
    python label_anchors.py --dataset dataset --out anchordataset
    python label_anchors.py --image photo.jpg      # one file
    python label_anchors.py --anchor 636,663,0.40 --anchor 1097,605,1.00  # force, all images
    python label_anchors.py --anchor-file other/anchor.txt               # force one file

Each output is ``<stem>_anchors.png`` plus ``anchors.json`` at the top of the
output folder, recording which anchor file (and which anchors) each image got.
PIL only - no torch, no model download, so it runs instantly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # environment guard
    sys.exit("Missing dependency 'pillow'.\nInstall with: pip install pillow")

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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    return p.parse_args(argv)


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


def load_font(size):
    """A decent TrueType face if the system has one, else PIL's bitmap default."""
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


def draw_legend(draw, anchors, font, img_size, px_per_m):
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
        draw_legend(draw, anchors, font, img_size, None)

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


def main(argv=None):
    args = parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
