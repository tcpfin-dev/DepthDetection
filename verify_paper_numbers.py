#!/usr/bin/env python3
"""Recompute the numbers the paper quotes, straight from outputs/eval_per_image.csv.

    ./venv/bin/python verify_paper_numbers.py [--out outputs]

Prints Table II, the per-object W/Z breakdown of Section V, the depth-error
shares, the mean Z/Z ratios and the back-solved focal length. Every printed
value is the one quoted in paper/paper.tex; a mismatch means the tree changed
underneath the paper.

Nothing here runs a model: it is pure aggregation, so it takes a second and is
safe to run any time after a batch.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

FX_ASSUMED = 1506.9   # 65 deg hfov at 1920 px, the pipeline's default
DAV2 = ("small", "base", "large")
MODELS = ("small", "base", "large", "midas", "marigold")


def load_rows(out_dir: Path) -> list[dict]:
    path = out_dir / "eval_per_image.csv"
    if not path.exists():
        raise SystemExit(f"no {path} - run a batch first (see REPRODUCE.md)")
    rows = []
    for r in csv.DictReader(path.open()):
        for key in ("z_abs_err_cm", "z_err_cm", "z_rel_err_pct", "z_m", "w_m",
                    "h_m", "true_w_m", "true_h_m", "distance_true_m", "w_err_cm",
                    "h_err_cm"):
            if r.get(key) in (None, ""):
                r[key] = None
            else:
                r[key] = float(r[key])
        rows.append(r)
    return rows


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def table2(rows: list[dict]) -> None:
    print("Table II - overall accuracy (475 scored rows, 25 voided)")
    print(f"  {'model':9s}{'n':>4}{'MAE':>7}{'rel%':>7}{'bias':>7}{'RMSE':>7}"
          f"{'>20':>5}{'>30%':>6}{'W%':>7}{'H%':>7}")
    for m in MODELS:
        sub = [r for r in rows if r["model"] == m]
        n = len(sub)
        mae = mean([r["z_abs_err_cm"] for r in sub])
        rel = mean([abs(r["z_rel_err_pct"]) for r in sub])
        bias = mean([r["z_err_cm"] for r in sub])
        rmse = (sum(r["z_abs_err_cm"] ** 2 for r in sub) / n) ** 0.5
        g20 = sum(1 for r in sub if r["z_abs_err_cm"] > 20)
        g30 = sum(1 for r in sub
                  if abs(r["z_err_cm"]) / (r["distance_true_m"] * 100) * 100 > 30)
        w = mean([abs(r["w_err_cm"]) / 100 / r["true_w_m"] * 100 for r in sub])
        h = mean([abs(r["h_err_cm"]) / 100 / r["true_h_m"] * 100 for r in sub])
        print(f"  {m:9s}{n:4d}{mae:7.2f}{rel:7.1f}{bias:+7.2f}{rmse:7.2f}"
              f"{g20:5d}{g30:6d}{w:7.1f}{h:7.1f}")


def depth_shares(rows: list[dict]) -> None:
    dav2 = [r for r in rows if r["model"] in DAV2]
    total = sum(r["z_abs_err_cm"] for r in dav2)
    print("\nSection V - share of total absolute depth error (pooled over Depth Anything V2)")
    for oid in ("b01", "b02"):
        share = sum(r["z_abs_err_cm"] for r in dav2 if r["object_id"] == oid) / total * 100
        print(f"  {oid}: {share:.1f} %")
    both = sum(r["z_abs_err_cm"] for r in dav2 if r["object_id"] in ("b01", "b02")) / total * 100
    print(f"  both bottles: {both:.1f} %   (other three objects: {100 - both:.1f} %)")

    print("\nSection V - box-vs-depth split of width error (pooled over Depth Anything V2)")
    print(f"  {'obj':7s}{'dist%':>7}{'W%':>7}{'H%':>7}{'W-Z':>7}{'box share':>11}")
    for oid in ("a01", "b01", "b02", "bwl01", "c01"):
        sub = [r for r in dav2 if r["object_id"] == oid]
        zd = mean([abs(r["z_err_cm"]) / (r["distance_true_m"] * 100) * 100 for r in sub])
        wd = mean([abs(r["w_err_cm"]) / 100 / r["true_w_m"] * 100 for r in sub])
        hd = mean([abs(r["h_err_cm"]) / 100 / r["true_h_m"] * 100 for r in sub])
        print(f"  {oid:7s}{zd:7.1f}{wd:7.1f}{hd:7.1f}{wd - zd:7.1f}{100 * (wd - zd) / wd:10.0f}%")


def misc(rows: list[dict], out_dir: Path) -> None:
    print("\nMean Z_hat/Z (one multiplicative over-estimate across all models)")
    for m in MODELS:
        sub = [r for r in rows if r["model"] == m]
        print(f"  {m:9s}{mean([r['z_m'] / r['distance_true_m'] for r in sub]):.3f}")

    vals = [FX_ASSUMED * r["w_m"] / r["true_w_m"] for r in rows
            if r["object_id"] in ("a01", "bwl01", "c01") and r["w_m"] is not None]
    med = statistics.median(vals)
    print(f"\nBack-solved focal length (opaque objects, n={len(vals)}): "
          f"median {med:.0f} px, {100 * (1 - med / FX_ASSUMED):.1f} % below {FX_ASSUMED}")

    big = [r for r in rows if r["z_abs_err_cm"] > 20]
    print(f"Rows over 20 cm: {len(big)} total, by lighting {dict(Counter(r['lighting'] for r in big))}")

    voided_path = out_dir / "eval_voided.csv"
    if voided_path.exists():
        voided = list(csv.DictReader(voided_path.open()))
        print(f"Voided rows (subject never detected): {len(voided)} "
              f"by lighting {dict(Counter(r['lighting'] for r in voided))}")
    else:
        print("Voided rows: no eval_voided.csv in this tree")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("outputs"),
                    help="eval tree to read (default: outputs)")
    args = ap.parse_args()
    rows = load_rows(args.out)
    print(f"{len(rows)} scored rows from {args.out / 'eval_per_image.csv'}\n")
    table2(rows)
    depth_shares(rows)
    misc(rows, args.out)


if __name__ == "__main__":
    main()
