#!/usr/bin/env python3
"""Merge several per-model batch runs into ONE eval tree.

Why this exists: ``pipeline.write_eval`` rewrites ``eval_*.csv`` from the rows of
the run in progress, so a five-model table cannot be produced by five sequential
runs into the same ``--out`` (each run would replace the previous model's rows).
Marigold also needs a different ``--calib`` from the regression heads, so the
paper's single table is necessarily assembled from more than one run.

    # four regression heads under the disparity fit, Marigold under its affine fit
    for m in small base large midas; do
      ./venv/bin/python pipeline.py --dataset dataset --depth-model $m \\
          --yolo-size medium --calib inverse --out /tmp/repro/$m
    done
    ./venv/bin/python pipeline.py --dataset dataset --depth-model marigold \\
        --yolo-size medium --calib affine --seed 42 --out /tmp/repro/marigold

    # one tree with all five models and the eval CSVs regenerated
    ./venv/bin/python merge_eval_trees.py --out outputs_new /tmp/repro/*

The per-image artifacts (depth maps, overlays, ``_results.json``) are copied
verbatim, so the merged tree is a complete replacement for a single all-model
run. Rows and voided entries are taken from each tree's own CSVs, so nothing is
re-scored here - this only re-aggregates.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pipeline  # for write_eval + BATCH_DEPTH_MODELS; importing loads torch only

NUMERIC = ("distance_true_m", "conf", "z_m", "z_err_cm", "z_abs_err_cm",
           "z_rel_err_pct", "w_m", "true_w_m", "w_err_cm", "h_m", "true_h_m",
           "h_err_cm")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for raw in csv.DictReader(path.open()):
        row = dict(raw)
        for key in NUMERIC:
            val = row.get(key)
            try:
                row[key] = float(val) if val not in (None, "") else None
            except ValueError:
                row[key] = None
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="destination eval tree")
    ap.add_argument("trees", nargs="+", type=Path, help="per-run output dirs to merge")
    ap.add_argument("--expect", type=int, default=0,
                    help="fail unless the merged tree has this many scored rows "
                         "(e.g. 475 for the paper's dataset)")
    args = ap.parse_args()

    rows, voided = [], []
    # A model may appear in more than one tree (e.g. marigold inverse and
    # marigold affine); take each model WHOLE from the last tree that has it,
    # never a union, or that model's voided rows leak from the rejected run.
    per_tree = []
    for tree in args.trees:
        if not (tree / "eval_per_image.csv").exists():
            raise SystemExit(f"{tree}: no eval_per_image.csv (was the batch aborted?)")
        tree_rows = read_csv(tree / "eval_per_image.csv")
        models = sorted({r["model"] for r in tree_rows})
        per_tree.append((tree, models))
        print(f"[merge] {tree}: {len(tree_rows)} rows, {', '.join(models)}")

    chosen: dict[str, Path] = {}
    for tree, models in per_tree:
        for m in models:
            chosen[m] = tree                      # later tree wins, whole model
    for tree, models in per_tree:
        picked = [m for m in models if chosen[m] == tree]
        if not picked:
            continue
        for d in tree.iterdir():
            if d.is_dir() and d.name in picked:
                shutil.copytree(d, args.out / d.name, dirs_exist_ok=True)
        rows.extend(r for r in read_csv(tree / "eval_per_image.csv") if r["model"] in picked)
        voided.extend(v for v in read_csv(tree / "eval_voided.csv") if v["model"] in picked)
        print(f"[merge]   kept {', '.join(picked)} from {tree}")

    order = [m for m in pipeline.BATCH_DEPTH_MODELS if m in chosen]
    order += [m for m in chosen if m not in order]

    # Later trees win: passing a stale model's tree after a fresh one must not
    # silently double-count that model's rows.
    deduped: dict[tuple, dict] = {}
    for r in rows:
        deduped[(r["model"], r["group"], r["filename"])] = r
    if len(deduped) != len(rows):
        print(f"[merge] note: {len(rows) - len(deduped)} duplicate (model, image) rows "
              f"collapsed, last tree wins")
    rows = list(deduped.values())

    deduped_voided: dict[tuple, dict] = {}
    for v in voided:
        deduped_voided[(v["model"], v["group"], v["filename"])] = v
    if len(deduped_voided) != len(voided):
        print(f"[merge] note: {len(voided) - len(deduped_voided)} duplicate voided rows "
              f"collapsed, last tree wins")
    voided = list(deduped_voided.values())

    pipeline.write_eval(args.out, rows, order, voided=voided)
    print(f"[merge] wrote {args.out}: {len(rows)} scored rows, {len(voided)} voided, "
          f"models {order}")
    if args.expect and len(rows) != args.expect:
        raise SystemExit(f"expected {args.expect} scored rows, got {len(rows)}")


if __name__ == "__main__":
    main()
