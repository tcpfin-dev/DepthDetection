# AGENTS.md

## What this project is

Early-stage research scaffold for the proposal in `Research_Proposals.docx`:
**Zero-Shot Monocular Depth Estimation for Automated Crop/Object Sizing in
Low-Contrast Environments**. The end goal is a plug-and-play Python pipeline
that (a) runs a pre-trained monocular depth model on a single RGB image,
(b) combines the depth map with a YOLOv8 bounding box, and (c) converts
relative depth into metric width/height/volume using camera intrinsics.

No model training happens anywhere in this project (that is a deliberate
constraint of the proposal). Foundation models are used zero-shot.

Current state: the combined entry point is `test_pipeline.py` at the
**repository root** — one command runs a zero-shot depth model (Depth Anything
V2 by default, MiDaS via `--depth-model midas`) + YOLOv8, applies the geometric
metric-conversion pipeline from the proposal (Part 1) and sizes **every**
detected object, writing annotated images plus `_results.json` / `_results.csv`
into `outputs/` at the root. It fits `Z = alpha*D + beta` from measured anchors
(`--calib inverse` instead fits the disparity `1/Z = alpha*D + beta`, which
interpolates much better between anchors because monocular depth is roughly
inverse-depth), or derives Z from EXIF intrinsics and a known `--object-height`,
and falls back to the assumed `--hfov`. The `test/` directory holds the earlier testing-phase
standalone scripts: the model **smoke scripts** `test_depth_anything.py` and
`test_yolo.py`, plus `test_sizing.py` (which sizes a single object).

## Setup

The checked-in `.venv/` is essentially empty (contains only `pip`) and runs
**Python 3.14.8**. There is no `requirements.txt`, `pyproject.toml`, or
`setup.py` — dependencies are documented only in the script docstrings.

```bash
# Depth Anything V2 script
pip install torch transformers pillow numpy
# GPU torch build:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# YOLOv8 script
pip install ultralytics        # pulls torch + torchvision automatically
```

**Gotcha:** Python 3.14 is very new; CPU/GPU wheels for `torch`/`transformers`
may not exist yet. If installs fail, create a separate venv on Python 3.11/3.12
rather than debugging the existing `.venv`. (Verified working: `torch 2.14.1`
ships `cp314` wheels.)

**Gotcha:** on a headless host `ultralytics` imports `cv2`, and stock
`opencv-python` needs GUI libs (`libxcb.so.1`). Install
`opencv-python-headless` instead (it satisfies Ultralytics' `cv2` import).

## Running

All scripts are CLI entry points run directly with Python. They are
**not** pytest tests despite the `test_` filename prefix — there are no test
functions, no `pytest` dependency, and `pytest` collects nothing. Run them
directly:

```bash
python test/test_depth_anything.py --image photo.jpg
python test/test_yolo.py --image photo.jpg --model yolov8s.pt --conf 0.4
python test/test_sizing.py --image photo.jpg --target bottle --distance 0.5
python test/test_sizing.py --image photo.jpg --target bottle \
    --anchor 2048,1400,0.5 --anchor 3900,300,2.0 --fx 3215   # affine fit

# Combined pipeline (sizes every detection, writes JSON + CSV)
python test_pipeline.py --image photo.jpg --target bottle --distance 0.5
python test_pipeline.py --image photo.jpg --target bottle \
    --anchor 2048,1400,0.5 --anchor 3900,300,2.0        # affine fit
python test_pipeline.py --image photo.jpg --target bottle --object-height 0.20
python test_pipeline.py --image photo.jpg --target bottle --calib inverse \
    --anchor 1463,935,0.2 --anchor 1229,308,1.0          # disparity (1/Z) fit
python test_pipeline.py --image photo.jpg --target bottle --depth-model midas \
    --object-height 0.20                                # MiDaS control baseline

# Batch mode: scan a dataset folder, auto-read each folder's anchor.txt, run
# every image, write an evaluation.  --depth-model all = small + base + large.
python test_pipeline.py --dataset dataset --depth-model all --calib inverse
python test_pipeline.py --dataset dataset/white --depth-model large --conf 0.2
python test_pipeline.py --dataset dataset --yolo-size medium --calib inverse

# Anchors (one script, three modes): click them instead of reading pixels off
# in Paint, then draw them back to check them
python preprocessing.py                         # pick = per-folder anchor.txt writer
python preprocessing.py pick --overwrite --match '*/b1'
python preprocessing.py label                   # draw them back onto the images

# Lighting: warm -> yellower, dim -> darker, because the phone's auto exposure
# and white balance flattened both
python preprocessing.py adjust                  # dataset/ -> adjusteddataset/
python preprocessing.py adjust --only warm --warm-strength 0.9
python preprocessing.py adjust --in-place       # rewrite dataset/ itself
```

`colab_marigold.py` is the Colab depth runner: one file, **no arguments**, no
`test_pipeline.py` dependency. Upload it with the `dataset/` folder beside it
(`/content`) and run `!python colab_marigold.py`; the only input it looks for is
that `dataset/` next to the script (no archives, no Drive mount, no fallback to
cwd images - a missing folder is a hard error). It installs
`diffusers`/`accelerate` if missing, loads `prs-eth/marigold-depth-v1-0` (fp16 on
a GPU), runs it over every image, writes
`<out>/<lighting>/<object>/<image>_depth.npy` + `_depth.png` (grayscale) +
`_depth_color.png` (magma) + `_photo.png` (thumb) into `marigold_outputs/`, then
renders and inline-displays a `photo | depth` montage (`preview.png`) and
`summary.json`. Tuning lives in the constants at the top of the file
(`MODEL_ID` / `STEPS` / `ENSEMBLE` / `RESOLUTION` / `OUT_DIR`), not in flags, and
existing `_depth.npy` files are reused on a re-run.

Anchors live **per object folder**: `dataset/<lighting>/<object>/anchor.txt`, and
there is no inheritance - an image only ever uses the `anchor.txt` sitting in its
own folder. `preprocessing.py pick` writes them: it walks the folders that contain
images, serves one reference frame per folder to a local browser page, and turns
two clicks into `cup1 anchor (636, 663, 40cm)` / `cup2 anchor (1097, 605, 100cm)`
lines. Wheel-zoom for pixel accuracy, `--names` / `--dist-cm` for other anchors,
`--overwrite` to redo folders that already have the file, `--preview` to run
`preprocessing.py label` afterwards. That `label` mode is the inverse: it reads
the same file and draws the markers back onto every image (into `anchordataset/`
by default, mirroring the dataset).

`preprocessing.py adjust` exists because the phone's auto exposure and white
balance flatten the lighting modes: real `warm` frames come back almost as
neutral as `white`, and real `dim` frames are only slightly darker. It rewrites
every `warm` frame with a stronger yellow cast (`--warm-strength`, default 0.65)
and every `dim` frame darker and flatter (`--dim-brightness` 0.55,
`--dim-contrast` 0.9), and leaves `white` and `nv` alone. The lighting is read
from the folder name or a filename token (`_warm`, `_dim`), the output mirrors
`--dataset` into `adjusteddataset/` with the same filenames, and each folder's
`anchor.txt` is copied along because pixel geometry is untouched - so
`test_pipeline.py --dataset adjusteddataset` works immediately. `--in-place`
rewrites the dataset itself, `--only warm|dim` narrows what is touched,
`--dry-run` only prints the plan. Batch mode
(`--dataset <root>`) walks the folder recursively, checks
each object folder for that file, and a folder without one is **skipped as a
whole** and reported once with a reminder to add it (a half-shot dataset still
runs instead of dying). It then runs each image through every requested model
(reusing each model across the batch) and writes the results under
`<out>/<model>/<relative path>/` plus eval files at the top of `<out>`:
`eval_per_image.csv`, `eval_summary.csv`, `eval_by_lighting.csv`,
`eval_by_distance.csv`, `eval_by_object.csv`. `--anchor` / `--anchor-file` force
one calibration on every image and `--distance` / `--object-height` calibrate
without anchors at all, so those flags bypass the per-folder requirement.
`--yolo-size nano|small|medium|large|xlarge` selects the YOLOv8 backbone
(default `nano`; `--yolo <weights>` overrides it). `medium` detects the dataset's
small objects (apple, b02 bottle) far more reliably than `nano`, while `large`
has not consistently helped.
The evaluation is computed on the
**middle object** (the detection whose box centre lies between the two anchor
pixels), with MAE / RMSE of Z in cm and the relative %; ground truth comes from
a `ground_truth.csv` (Section 8 schema) if one sits next to the dataset, otherwise
from the filename (`..._50cm` / `..._d050_...`) and the lighting folder/token.
Each image's `_results.json` meta records `calibration.anchor_file`, and
`eval_per_image.csv` has an `anchor_file` column, so a per-folder calibration is
always traceable.

**Object sizes (W/H ground truth) come from `dataset/objects.txt`.** It is the
measure-once table for the whole set, parsed by `load_object_sizes()`:

```text
target objects:
a01 W:7.5cm,H:7cm
b01 W:6.2cm,H:24cm
b02 W:7.5cm,H:25cm
bwl01 W:12cm,H:6cm
c01 W:12cm, H:8cm

anchors
cups W:12cm, H:8cm
cups without handle W:8cm, H:8cm
```

Only the `target objects:` section is read; those W/H become `true_w_m` /
`true_h_m` in `eval_per_image.csv` (and the `w_err_cm` / `h_err_cm` columns +
the `[W+1.1/H+0.8cm]` tag on each per-image line). Values from a
`ground_truth.csv` row still win when that row carries its own `true_w_m` /
`true_h_m`. The lookup is keyed by `object_id` (from the CSV row or the filename
prefix), accepting both the short id (`c01`) and the spelled-out prefix
(`cup01` / `bottle01` -> `b01`, `bowl01` -> `bwl01`). The file is found in the
dataset folder, or up to three of its ancestors, so `--dataset dataset/white/c1`
still picks up `dataset/objects.txt`.

**The `anchors` section is calibration, not ground truth, and is skipped.** The
reference cups are the fixed 0.40 / 1.00 m constants in each folder's
`anchor.txt`. Note the special case: **only the b01 folders use the handle-less
anchor cup pair** (`cups without handle W:8cm, H:8cm`); every other object folder
(the a1 / b2 / bwl1 / c1 scenes) uses the normal with-handle cups
(`cups W:12cm, H:8cm`). Either way they are just the two calibration anchors at
0.40 / 1.00 m, so they are never scored objects and must not be added to the
target table.

Ground truth is **optional** - without it the batch still sizes every image and
prints the model outputs (Z / W / H); it never errors. `--no-eval` skips the
evaluation deliberately and prints predictions only. The cup anchors are not
ground truth: they are the fixed calibration constant that turns relative depth
into meters (and `--distance` / `--object-height` work when there are no cups).

`test_sizing.py` and `test_pipeline.py` need a calibration anchor (a
tape-measured distance) to turn relative depth into meters: `--distance <m>` for
a single anchor (assumes `beta = 0`) or two or more `--anchor u,v,distance_m` for
a full fit. `test_pipeline.py`'s `--calib` picks what is fitted: `affine`
(default) fits `Z = alpha*D + beta`; `inverse` fits the disparity
`1/Z = alpha*D + beta`, which is much more accurate for objects that sit
*between* the anchors (monocular depth is roughly inverse-depth, not linear in
distance). `test_pipeline.py` additionally accepts `--object-height <m>`
(known real height of the `--target` class, so no depth anchor is needed).
Camera focal length comes from `--fx`/`--fy`, else EXIF
(`FocalLengthIn35mmFilm`), else an assumed `--hfov` (65 deg).

If `--image` is omitted each script generates a deterministic synthetic image
(seeded RNG) so it doubles as an environment/dependency smoke test. Expect
meaningless depth output and near-zero YOLO detections on synthetic images.

`test_pipeline.py` writes its outputs (gray + colorized depth maps, annotated
overlay image) to `outputs/` next to it at the **repository root** by default
(overridable with `--out`), plus `<image>_results.json` and `<image>_results.csv`
(one row per detection: box, relative depth, `Z`, `W`, `H`, area). The `test/`
scripts keep their own `test/outputs/`. The YOLO script writes its downloaded
`.pt` weights into the **current working directory** on first run, so run it from
a predictable directory.

## Conventions / gotchas

- **Depth Anything V2 checkpoint IDs must end in `-hf`** (e.g.
  `depth-anything/Depth-Anything-V2-Small-hf`). The non-`-hf` repos ship custom
  model code and are not loadable by the Transformers
  `pipeline("depth-estimation", ...)` used here. The `MODELS` dict maps
  `small`/`base`/`large` to the correct IDs; `midas`/`midas-small` map to
  Transformers-compatible MiDaS (DPT) checkpoints (`Intel/dpt-hybrid-midas`,
  `Intel/dpt-swinv2-tiny-256`) selected with `--depth-model midas`. Live in
  `test_pipeline.py`; the Depth Anything V2 default is unchanged.
- Device selection is automatic: CUDA if `torch.cuda.is_available()` else CPU
  (`device=0` vs `-1` for the HF pipeline; `0` vs `"cpu"` for Ultralytics).
- Scripts use a `try/except ImportError` guard that prints a
  `pip install ...` hint and exits, rather than crashing with a traceback.
- Each script defines its own `make_synthetic_image`, `load_image`, and
  `parse_args`; there is no shared module. If you add shared utilities, create
  a package rather than importing across these scripts via path hacks.
- Depth maps are normalized min-max for visualization only; raw relative depth
  is `result["predicted_depth"]`.
- **Depth orientation:** the `*-hf` checkpoints emit relative depth in an
  *inverse-depth* convention — **larger value = closer** to the camera (verified
  on the reference image: near floor ~5.8, far wall ~0). The affine
  `Z = alpha*D + beta` in `test_sizing.py` learns the sign, so nothing
  hard-codes it.
- Metric scaling is implemented in `test/test_sizing.py` (the `Z = alpha*D +
  beta` affine fit and the `W/H = pixels * Z / f` back-projection).

## Repository context

- Not a git repository, and `.crush/.gitignore` ignores `*` (Crush internal
  state dir — leave it alone).
- `Research_Proposals.md` is the source-of-truth spec (execution plan, the
  metric-conversion formulas, and the candidate model list: Depth Anything V2,
  Marigold, MiDaS v3.1). Read it directly; it is a verbatim markdown
  transcription of `Research_Proposals.docx` (the original binary is kept for
  reference only).
- Planned stack from the proposal: Python, PyTorch, Hugging Face Transformers,
  and OpenCV.
