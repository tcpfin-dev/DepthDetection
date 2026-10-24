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

# Anchors: click them instead of reading pixels off in Paint
python pick_anchors.py                          # per-folder anchor.txt writer
python pick_anchors.py --overwrite --match '*/b1'
python label_anchors.py                         # draw them back onto the images
```

Anchors live **per object folder**: `dataset/<lighting>/<object>/anchor.txt`, and
there is no inheritance - an image only ever uses the `anchor.txt` sitting in its
own folder. `pick_anchors.py` writes them: it walks the folders that contain
images, serves one reference frame per folder to a local browser page, and turns
two clicks into `cup1 anchor (636, 663, 40cm)` / `cup2 anchor (1097, 605, 100cm)`
lines. Wheel-zoom for pixel accuracy, `--names` / `--dist-cm` for other anchors,
`--overwrite` to redo folders that already have the file, `--preview` to run
`label_anchors.py` afterwards. It is the inverse of `label_anchors.py`, which
reads the same file and draws the markers back onto every image. Batch mode
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
pixels), with MAE / RMSE of Z in cm and the relative %; ground truth comes from a
`ground_truth.csv` (Section 8 schema) if one sits next to the dataset, otherwise
from the filename (`..._50cm` / `..._d050_...`) and the lighting folder/token.
Each image's `_results.json` meta records `calibration.anchor_file`, and
`eval_per_image.csv` has an `anchor_file` column, so a per-folder calibration is
always traceable.

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
