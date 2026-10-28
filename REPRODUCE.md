# Reproducing the paper's numbers

Every number in `paper/paper.tex` comes from the eval tree in `outputs/`. This
file lists the exact commands that regenerate it, in the order they were run,
with the output each one must produce. All of it was re-executed from scratch
and diffed against `outputs/`: 475/475 rows and 25/25 voided identical, zero
field differences.

## What you need

- The 100 source frames and all 20 `anchor.txt` files are committed to this repo
  (`dataset/`, 122 tracked files), so a fresh clone can regenerate everything
  below without any external download. Only `outputs/*` is gitignored - the eval
  tree is a build artifact, which is why the recipe rebuilds it.
- `./venv/bin/python` (Python 3.14.8, `torch 2.14.1+cu130`) with
  `transformers`, `ultralytics`, `diffusers`, `scipy`. A CUDA GPU is optional
  but Marigold is unusably slow without one.
- Run everything from the repository root.

## Fixed configuration (what the paper states)

| Setting | Value | Flag |
| --- | --- | --- |
| Detector | YOLOv8m, conf 0.25 | `--yolo-size medium` (the CLI default is **nano** - always pass this) |
| YOLO inference size | 640 px | `--imgsz 640` (default) |
| Calibration, DA-V2 / MiDaS | disparity `1/Z = aD + b` | `--calib inverse` (default) |
| Calibration, Marigold | linear `Z = aD + b` | `--calib affine` |
| Marigold sampler | 10 steps, ensemble 1, 768 px, seed 42 | `--marigold-steps 10 --marigold-ensemble 1 --marigold-resolution 768 --seed 42` |
| Intrinsics | 65 deg hfov (fx = 1506.9 px) | default `CAMERA_HFOV_DEG`; no flag |
| Depth sampling | box median | `--depth-sample box_median` (default) |
| Anchors | per-folder `anchor.txt` | picked up automatically in batch mode |

Two rules that are easy to get wrong and silently change the results:

- **Marigold must not be run under `--calib inverse`.** Its direct-depth output
  makes `1/(aD+b)` near-singular: one object at 0.80 m reads 4.03 m, two extra
  rows void, and full-set MAE goes 9.53 -> 15.34 cm.
- **Never run several models sequentially into the same `--out`.** Each batch
  rewrites `eval_*.csv` from its own rows, so the last model would be the only
  one left in the table. Use one tree per run and merge (step 2).

## 1. Full dataset, the five backbones (Tables II and III)

```sh
S=/tmp/repro && mkdir -p $S

# four regression heads under the disparity fit (deterministic: no seed needed)
for m in small base large midas; do
  ./venv/bin/python pipeline.py --dataset dataset --depth-model $m \
      --yolo-size medium --calib inverse --out $S/$m
done

# Marigold under its native affine fit, seeded
./venv/bin/python pipeline.py --dataset dataset --depth-model marigold \
    --yolo-size medium --calib affine --seed 42 --out $S/marigold
```

Each run ends with a per-distance and per-lighting table; the regression heads
take tens of seconds each and Marigold dominates the wall clock (minutes for
100 images).

## 2. Merge them into one tree (475 rows / 25 voided)

```sh
./venv/bin/python merge_eval_trees.py --out /tmp/repro/merged --expect 475 \
    $S/small $S/base $S/large $S/midas $S/marigold

./venv/bin/python verify_paper_numbers.py --out /tmp/repro/merged
```

`merge_eval_trees.py` copies each model's per-image artifacts and re-aggregates
the eval CSVs through the pipeline's own `write_eval`, so the merged tree is a
drop-in replacement for a single all-model run. Each model is taken whole from
the last tree that provides it.

`verify_paper_numbers.py` prints Table II, the error shares, the mean `Z_hat/Z`
ratios, the back-solved focal length and the voided counts, all read straight
from the CSV. Expected output (this is what the paper quotes):

```
Table II - overall accuracy (475 scored rows, 25 voided)
  model       n    MAE   rel%   bias   RMSE  >20  >30%     W%     H%
  small      95   5.78    8.0  +4.77  10.71   11     7   12.9    8.3
  base       95   5.35    7.4  +4.99   7.83    5     1   11.8    6.1
  large      95   3.63    4.9  +3.21   5.15    0     0    9.3    4.9
  midas      95   9.38   13.1  +5.26  15.91   11    11   15.5   11.1
  marigold   95   9.53   13.6  +4.65  17.18    5     6   18.6   12.0

b01: 61.1 %   b02: 17.7 %   both bottles: 78.8 %  (other three: 21.2 %)
box share of width error: b02 63 %, b01 33 %, apple excess 0.2 points
Z_hat/Z: 1.062 / 1.067 / 1.042 / 1.075 / 1.072
back-solved focal length: median 1483 px, 1.6 % below 1506.9
rows over 20 cm: 32 total, 17 of them night
voided: 25, all night apple frames
```

To install a rerun as the paper's tree: `--out outputs` on the merge step
(the previous CSVs are overwritten; keep a copy first).

## 3. Table I, the calibration ablation (white set, n = 25)

Both fits for every backbone, one tree per cell:

```sh
for m in small base large midas; do
  for c in affine inverse; do
    ./venv/bin/python pipeline.py --dataset dataset/white --depth-model $m \
        --yolo-size medium --calib $c --out $S/t1_${m}_${c}
  done
done
./venv/bin/python pipeline.py --dataset dataset/white --depth-model marigold \
    --yolo-size medium --calib affine --seed 42 --out $S/t1_marigold_affine
./venv/bin/python pipeline.py --dataset dataset/white --depth-model marigold \
    --yolo-size medium --calib inverse --seed 42 --out $S/t1_marigold_inverse
```

Expected MAE (cm) / mean absolute relative error (%):

| model | affine | inverse | AF/INV |
| --- | --- | --- | --- |
| DA-V2-S | 12.16 / 18.1 | 4.45 / 6.1 | 2.73 |
| DA-V2-B | 13.29 / 20.1 | 4.73 / 6.5 | 2.81 |
| DA-V2-L | 12.42 / 18.7 | 3.42 / 4.5 | 3.63 |
| MiDaS | 16.58 / 25.5 | 12.32 / 17.1 | 1.35 |
| Marigold | 6.72 / 9.5 | 11.45 / 15.5 | 0.59 |

Note the `rel` column is the **mean absolute** relative error, matching
Tables II and III; the mean *signed* value is smaller for an inverse fit
(4.46 vs 6.07 on DA-V2-S) and using it silently makes Table I disagree with the
rest of the paper.

## 4. Section V-F, the depth-sampling sweep (night / tall bottle)

```sh
for mode in box_median box_center bottom_center; do
  ./venv/bin/python pipeline.py --dataset dataset/nv/b1 --depth-model small \
      --yolo-size medium --depth-sample $mode --out $S/sweep_$mode
done
```

Expected MAE on the five night frames: box median **28.99 cm** (39.8 %),
box centre **25.70 cm** (34.8 %), bottom centre **13.86 cm** (19.0 %) - a 52 %
cut. Swapping in `--yolo-size nano` reproduces neither the numbers nor the row
count (33.30 / 30.46 / 18.88 cm, and it sizes only 4 of the 5 frames), which is
why the paper specifies the medium detector.

## 5. Figures

`paper/make_paper_figures.py` reads `outputs_clean/`, which is not a directory
in this repo. Point it at the real tree for the duration of the run:

```sh
ln -sfn outputs outputs_clean
./venv/bin/python paper/make_paper_figures.py
rm outputs_clean            # do not commit this symlink
```

Fig. 2 (qualitative montage) and Fig. 3 (error vs distance) are both rebuilt
from the eval tree, so re-run this after any rerun or the figures keep plotting
old numbers. Fig. 1 is a schematic and ignores the data.

## 6. Rebuild the PDF

```sh
cd paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex
```

Six pages, zero errors, no undefined references.

## 7. The ungated-selection numbers (Section "Voided Rows")

The old ungated eval tree no longer exists, so the paragraph that cited it was
replaced with figures replayed from the detections still stored in
`outputs/**/*_results.json` (every frame keeps ALL its detections, not just the
scored one). The rule replayed is the old one: max conf over detections whose box
centre lies between the frame's two anchor u pixels, no class filter.

```sh
./venv/bin/python reconstruct_ungated.py --out outputs
```

Expected output (these are the numbers in the paper):

```
image pairs with a detection between the anchors: 485
substituted by the ungated rule                 : 35 (7.2 %)
MAE on substituted rows :  29.27 cm (n=35)
MAE on correctly scored :   6.35 cm (n=450)

model     n   ungated MAE  ungated RMSE   RMSE gain   share of SSE
base       97         6.13         10.15    +29.7 %       57.5 %
large      97         4.93          9.71    +88.6 %       79.1 %
marigold   97        10.77         19.57    +13.9 %       27.3 %
midas      97        11.61         21.71    +36.5 %       48.2 %
small      97         6.55         13.06    +22.0 %       51.6 %

worst substituted row: knife 132.8 cm (nv_a1/a01_60cm.png, midas)
```

## Scope and caveats

- Deterministic claims: a fresh-process rerun reproduces `z_m`, `w_m`, `h_m` and
  `z_err_cm` exactly for both a regression head (95/95 rows) and Marigold under
  its seed (95/95 affine, 93/93 inverse).
- Marigold's own paper uses 50 steps and a 10-member ensemble; at 10/1 its
  9.53 cm is a floor on its accuracy, not a compute-matched comparison.
- The detection stage itself is deterministic: a fresh yolov8m run reproduces
  `label`, `conf` and every box to 1e-9, so the 475 scored / 25 voided split is
  stable (all 25 misses are the night apple folder).
- The paper no longer quotes the *original* ungated run (51 of 500 rows,
  39.5 vs 6.5 cm, a 668 cm knife). That tree is gone; step 7 gives the
  replayable replacement, which describes the same failure mode with smaller
  magnitudes because the anchors have been re-clicked since.
