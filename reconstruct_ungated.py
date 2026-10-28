#!/usr/bin/env python
"""Rebuild the paper's ungated-selection numbers from the stored detections.

Section "Voided Rows and Evaluation Integrity" claims that an evaluation which
picked the most confident detection between the anchors, with no expected-class
gate, substitutes a reference object on some rows. That old tree no longer
exists, but every ``<stem>_results.json`` in the eval tree keeps ALL detections
for its frame, so the rule can be replayed exactly:

    pick = argmax(conf) over detections whose box centre x lies between the two
           anchor u pixels of that frame (inclusive), no class filter

and compared against the gated row in ``eval_per_image.csv``.

Usage:
    ./venv/bin/python reconstruct_ungated.py --out outputs

It prints the substituted-row count, the MAE on those rows vs the correctly
scored ones, each model's share of squared error, the RMSE inflation and the
worst single row. Those are the numbers quoted in the paper.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict


def true_distance(filename: str) -> float:
    m = re.search(r"_(\d+)cm", filename)
    if not m:
        raise SystemExit(f"cannot read true distance from {filename!r}")
    return int(m.group(1)) / 100.0


def load_gated(tree: str):
    per_image = os.path.join(tree, "eval_per_image.csv")
    voided_path = os.path.join(tree, "eval_voided.csv")
    gated = {(r["model"], r["group"], r["filename"]): r
             for r in csv.DictReader(open(per_image, newline=""))}
    voided = set()
    if os.path.exists(voided_path):
        voided = {(r["model"], r["group"], r["filename"])
                  for r in csv.DictReader(open(voided_path, newline=""))}
    return gated, voided


def collect(tree: str, gated, voided):
    """Replay the ungated rule on every stored results JSON."""
    rows = []
    pattern = os.path.join(tree, "*", "*", "*", "*_results.json")
    for path in sorted(glob.glob(pattern)):
        model = os.path.relpath(path, tree).split(os.sep)[0]
        with open(path) as fh:
            data = json.load(fh)
        stem = os.path.basename(path)[: -len("_results.json")]
        rel = os.path.relpath(path, os.path.join(tree, model)).split(os.sep)
        if len(rel) < 3:
            continue
        key = (model, f"{rel[0]}_{rel[1]}", stem + ".png")
        anchors = data.get("meta", {}).get("calibration", {}).get("anchors") or []
        if len(anchors) < 2:
            continue
        us = [float(a.split(",")[0]) for a in anchors]
        lo, hi = min(us), max(us)
        cands = [d for d in data.get("results", [])
                 if lo <= (d["x1"] + d["x2"]) / 2 <= hi]
        if not cands:
            continue
        pick = max(cands, key=lambda d: d["conf"])
        t = true_distance(stem + ".png")
        rows.append({
            "model": model,
            "key": key,
            "label": pick["label"],
            "conf": pick["conf"],
            "z_m": pick["z_m"],
            "true_m": t,
            "err_cm": abs(pick["z_m"] - t) * 100.0,
            "gated": gated.get(key),
            "voided": key in voided,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs", help="eval tree to replay (default: outputs)")
    args = ap.parse_args()

    gated, voided = load_gated(args.out)
    rows = collect(args.out, gated, voided)
    if not rows:
        raise SystemExit(f"no *_results.json found under {args.out!r}")

    def differs(r):
        g = r["gated"]
        return (r["voided"] or g is None or g["label"] != r["label"]
                or abs(float(g["z_m"]) - r["z_m"]) > 1e-9)

    bad = [r for r in rows if differs(r)]
    good = [r for r in rows if not differs(r)]

    print(f"image pairs with a detection between the anchors: {len(rows)}")
    print(f"substituted by the ungated rule                 : {len(bad)} "
          f"({100 * len(bad) / len(rows):.1f} %)")
    print(f"  labels: {dict(Counter(r['label'] for r in bad))}")
    print(f"MAE on substituted rows : {statistics.fmean(r['err_cm'] for r in bad):6.2f} cm (n={len(bad)})")
    print(f"MAE on correctly scored : {statistics.fmean(r['err_cm'] for r in good):6.2f} cm (n={len(good)})")

    summary_path = os.path.join(args.out, "eval_summary.csv")
    gated_rmse = {}
    if os.path.exists(summary_path):
        gated_rmse = {r["model"]: float(r["rmse_z_cm"])
                      for r in csv.DictReader(open(summary_path, newline=""))}

    print("\nper model                                   ungated   gated   RMSE / SSE share of")
    print("model     n   ungated MAE  ungated RMSE   RMSE gain   substituted rows")
    per_model = defaultdict(list)
    for r in rows:
        per_model[r["model"]].append(r)
    for model in sorted(per_model):
        sub = per_model[model]
        rmse = math.sqrt(statistics.fmean(r["err_cm"] ** 2 for r in sub))
        mae = statistics.fmean(r["err_cm"] for r in sub)
        total_sq = sum(r["err_cm"] ** 2 for r in sub)
        bad_sq = sum(r["err_cm"] ** 2 for r in bad if r["model"] == model)
        gated_val = gated_rmse.get(model)
        gain = f"{100 * (rmse / gated_val - 1):+5.1f} %" if gated_val else "   n/a"
        print(f"{model:9} {len(sub):3d} {mae:12.2f} {rmse:13.2f} {gain:>10} "
              f"{100 * bad_sq / total_sq:10.1f} %")

    worst = max(bad, key=lambda r: r["err_cm"]) if bad else None
    if worst:
        print(f"\nworst substituted row: {worst['label']} {worst['err_cm']:.1f} cm "
              f"({worst['key'][1]}/{worst['key'][2]}, {worst['model']})")


if __name__ == "__main__":
    main()
