# Image Collection Plan

Field guide for the **100-image custom dataset** required by
`Research_Proposals.md` (Phase 1, Steps 1-2). The pipeline
(`test_pipeline.py`) is already built; what is missing is the dataset + the
physical ground truth it is evaluated against.

Goal: 100 camera photos of **five objects across three classes** - three bottles
(clear, blue, opaque), one opaque cup and one opaque bowl - spanning different
**sizes/colors**, **distances**, and **lighting conditions**, with the true
camera-to-object distance and real dimensions tape-measured for every shot.

All 100 are **real captures, no augmentation**: 5 objects x 5 middle distances x
4 lighting modes = 100 distinct smartphone photos. `Research_Proposals.md` asks
for 100 photographs taken with a phone, measured by hand per shot, so derived
(cropped / filtered) images cannot stand in for them; Section 11 records the
augmentation options only as a rejected alternative.

**Design: two fixed cup anchors + one moving object.** Two cups are taped to the
floor at all times - **cup1 on the 0.40 m mark, cup2 on the 1.00 m mark** - and
they never move, so their pixel positions are identical in all 100 images
(re-verify per lighting mode, Section 5 step 7 - a mode that uses a different
lens, crop or resolution shifts them). They are used as the pipeline's two
metric anchors:

```bash
--anchor <cup1_u>,<cup1_v>,0.40 --anchor <cup2_u>,<cup2_v>,1.00
```

Only the object in the **middle** moves, to 0.50 / 0.60 / 0.70 / 0.80 / 0.90 m.
Everything the paper evaluates (Z error, W/H error, RMSE/MAE by distance and by
lighting) is computed on that middle object, which always sits *between* the two
anchors so the calibration interpolates instead of extrapolates.

Anchors live **per object folder** (`dataset/<lighting>/<object>/anchor.txt`) and
are **required**: each object folder must contain one, an image never inherits an
anchor.txt from a parent folder, and the pipeline / `label_anchors.py` skip a
folder that has none (warning once, listing the folders) instead of guessing.

---

## 1. Equipment checklist

- camera
- Tape measure / ruler (metric).
- Tripod or a fixed rest so camera height and orientation never change.
- Floor tape or marks to fix the distance positions.
- **Two anchor cups** (any two easy-to-see cups) - permanently on the 0.40 m and
  1.00 m marks for the whole session. Keep these separate from the **c01** cup,
  which is a measured object that moves to the middle marks.
- The 5 objects: 3 bottles (clear, blue, opaque), 1 opaque cup (c01), 1 opaque bowl (bwl01).
- Paper/notebook or a laptop to log ground truth **as you shoot**.

---

## 2. The capture matrix (100 images)

One image = one `(object, middle distance, lighting)` combination. The two
anchors are constant, so they are not an axis.

| Axis | Levels | Count |
|---|---|---|
| Objects (3 bottles + 1 cup + 1 bowl) | b01, b02, b03, c01, bwl01 | 5 |
| Middle-object distance | 0.50, 0.60, 0.70, 0.80, 0.90 m | 5 |
| Lighting (real, 4 modes) | white, warm, dim, nv | 4 |

**5 x 5 x 4 = 100.**

The object set is deliberately mixed: the three bottles keep the class-level
control the earlier plan had, while the cup and the bowl add a different
geometry (short and wide, curved rim, big planar top) that stresses the W/H
back-projection on non-bottle shapes. Middle distance and lighting are the
variables the paper reports against; because the objects now differ in class,
report Z/W/H error **per object** as well as overall - the batch run writes
`eval_by_object.csv` for exactly that.

---

## 3. Object selection (the 5 objects)

The set is fixed at **5 objects** (three bottles, one cup, one bowl); the 4
lighting modes make up the rest of the 100. Inspect them first and give each a
role so the set varies **deliberately**, because the novelty claim is
*low-contrast* scenes:

| id | object | label | role / why it is in the set |
|---|---|---|---|
| b01 | normal plastic bottle | `bottle` | clear / transparent bottle - hardest contrast case |
| b02 | blue plastic bottle | `bottle` | dark, saturated plastic - colour/contrast change on a bottle |
| b03 | opaque bottle | `bottle` | normal opaque bottle (easy bottle control) |
| c01 | opaque cup | `cup` | new class: short, curved, visible interior - different aspect ratio |
| bwl01 | opaque bowl | `bowl` | new class: wide, low, large planar top - hardest W/H case |

`label` is the controlled class recorded in the ground truth (`bottle`, `cup`,
`bowl`); all three are real COCO classes, so YOLO can name them.

**The c01 cup is NOT one of the two anchor cups.** The anchors stay taped on the
0.40 m / 1.00 m marks for the whole session; c01 is a separate cup that is placed
on the middle marks (0.50 ... 0.90 m) exactly like the bottles and the bowl.

Fill in each object's real colour and character once you have it in hand; the
point is that the five differ (class, contrast, transparency, size), not that
they match the labels exactly. If two of the bottles turn out too similar, note
it in `notes` so the eval can flag the duplication.

Measure each object **once**, before shooting:

- Real width `W` (m)
- Real height `H` (m)

- **bottle / cup**: `W` = body width at mid-height (ignore a handle), `H` =
  base-to-rim / base-to-cap height.
- **bowl**: `W` = outer diameter (widest visible span), `H` = base-to-rim
  height. The bowl's top opening is much wider than a bottle's, so `W` will be
  the largest in the set - that is the point of including it.

Re-use those numbers for every shot of that object. Do **not** re-measure per
image.

---

## 4. Lighting conditions (4 real modes)

The light has four modes; all four are real captures, no synthetic filtering.
This is the axis that takes the set from 25 to 100:

| token | mode | what it is |
|---|---|---|
| `white` | fluorescent / cool white | bright, even, neutral - the easy case |
| `warm` | warm light | orange cast, lower colour temperature - tests colour bias |
| `dim` | dim | low light, flat and noisy - the low-contrast case the proposal cares about |
| `nv` | light **off** -> camera night vision | IR-illuminated, near-monochrome |

`warm` is the **real** version of a warm-cast augmentation, which is why no
colour filter is needed anywhere in this plan.

**`dim` and `nv` are different conditions - do not merge them.** `dim` is the
real low-light capture (the proposal's "low-contrast" case) and stays in the
normal colour domain. `nv` is a different *domain*: with the light off the
camera turns on its IR illuminator, so the scene is actively IR-lit and comes
back near-monochrome. The depth model was trained on RGB, so `nv` is
out-of-distribution and its error will likely be the largest; report it as its
own track.

**The anchors do not move because of the light.** Pixel projection depends only
on the rig (camera position, orientation, lens, framing), so on a locked tripod,
with the same lens and the same output resolution, the cups land on exactly the
same pixels in all four modes, `nv` included. What *can* move them is the
camera app, not the illumination:

- a **different lens / sensor** used for night vision (a dedicated IR or mono
  camera has a different field of view) - shifts every pixel;
- a **crop / digital zoom / stabilisation crop** applied in one mode only;
- a **different output resolution** between modes - the pixel grid changes, so
  `anchor.txt` needs rescaling by `new_w / old_w`, `new_h / old_h`.

IR light itself only affects focus and brightness, not pixel positions. So: note
the resolution each mode writes, confirm the framing once per mode (Section 5
step 7), and if only the resolution changed, rescale that folder's anchors
instead of re-measuring them. Anchors are stored per shot folder
(`dataset/<lighting>/<object>/anchor.txt`), so a mode that shifts pixels only
needs ITS folder's file edited - the other folders keep their own anchors.

Keep exposure / white balance on **auto** and lock nothing, but do not
post-process the originals; the depth model should see the real photo.

If you add more conditions later (side-lit/hard shadow, backlit), keep them as
extra levels and expand the matrix.

**Watch the anchors as the light changes.** The cups stay on their marks and the
camera never moves, so the anchor *pixels* are constant, but each mode's sampled
depth shifts a little. Do not re-aim the camera; just note the mode in `lighting`
and let the eval report the four tracks separately.

---

## 5. Fixed rig setup (do once per session)

1. Lock the camera in the tripod and never move it again.
2. Tape **cup1** to the 0.40 m mark and **cup2** to the 1.00 m mark, both
   fully visible, off to the sides so the middle object can never occlude them.
3. Set the middle-object marks at 0.50 / 0.60 / 0.70 / 0.80 / 0.90 m.
4. Keep the phone resolution, zoom and orientation fixed (no digital zoom, no
   crop) - the anchor pixels are only valid for this exact framing.
5. Take one reference shot and record the **anchor pixels** (the centre of each
   cup's visible body, not its rim or its handle):

```text
cup1 (0.40 m mark): pixel (u, v) = (     ,     )
cup2 (1.00 m mark): pixel (u, v) = (     ,     )
```

6. Write those two lines into that folder's `anchor.txt`
   (`dataset/<lighting>/<object>/anchor.txt`) - every shot folder carries its
   own copy, so a re-shot scene never inherits another scene's pixels:

```text
cup1 anchor (569, 721, 40cm)
cup2 anchor (1270, 577, 100cm)
```

7. Verify the rig by running the pipeline once (Section 10) and confirming the
   two cups come back at 0.40 m and 1.00 m. If they do not, that folder's anchor
   pixels are in the wrong place - fix them now, before shooting the rest of the
   folder. `python label_anchors.py` renders the anchors into
   `anchordataset/<same path>/` so you can see where they actually land. Repeat
   the check once per lighting mode (Section 4): a mode that uses a different
   lens, an in-app crop, or a different output resolution shifts the anchor
   pixels out from under that folder's `anchor.txt`, so catch it here, not after
   100 shots.

Only the **middle object** moves for the rest of the session; measure its
distance from the camera lens, tape-taut, at each mark.

---

## 6. Per-shot ground truth to record

For every image capture these (this is what RMSE/MAE are computed against):

- `filename`
- `object_id` (b01 | b02 | b03 | c01 | bwl01)
- `label` (bottle | cup | bowl - the controlled class of the middle object)
- `distance_m` (tape-measured camera -> **middle object**)
- `true_w_m`, `true_h_m` (from Section 3)
- `lighting` (white | warm | dim | nv)

The two anchors are **not** logged per row - they are constants of one shot
folder and live in that folder's `anchor.txt` / the `notes`, so the CSV stays one
row per image. The pipeline stamps `calibration.anchor_file` into every
`_results.json` and logs an `anchor_file` column in `eval_per_image.csv`, so a
row always points back at the file that calibrated it.

---

## 7. File naming

Use a systematic name so the log and the pipeline outputs line up:

```
bottle01_d050_white.jpg      # b01, 0.5 m, cool/fluorescent
bottle02_d080_warm.jpg       # b02, 0.8 m, warm light
cup01_d090_nv.jpg            # c01, 0.9 m, lights off (night vision)
bowl01_d070_dim.jpg          # bwl01, 0.7 m, dim light
```

Pattern: `<object><ID>_d<distance_cm>_<lighting>.jpg`

`<object>` is `bottle`, `cup` or `bowl` and `<ID>` is that object's 2-digit
number, so `b01` -> `bottle01`, `c01` -> `cup01`, `bwl01` -> `bowl01`. The
prefix is also what the batch eval falls back to for `object_id` when the
`ground_truth.csv` column is empty.

Distances in cm, zero-padded to 3 digits (`050`, `060`, `070`, `080`, `090`),
and `<lighting>` is one of `white`, `warm`, `dim`, `nv`.

---

## 8. `ground_truth.csv` schema

Create this next to the images (one row per image). The `filename` lines up with
the per-image `_results.csv` the pipeline writes, which contains:
`label, conf, x1, y1, x2, y2, depth, z_m, w_m, h_m, area_m2`.

Join on `filename`, then match the pipeline row whose box sits in the middle
(between the two anchor pixels) - that is the row being evaluated.

```csv
filename,object_id,label,distance_m,true_w_m,true_h_m,true_volume_ml,lighting,augmented,notes
bottle01_d050_white.jpg,b01,bottle,0.50,0.068,0.21,,white,false,
bottle02_d060_dim.jpg,b02,bottle,0.60,0.062,0.21,,dim,false,
bottle03_d070_warm.jpg,b03,bottle,0.70,0.070,0.22,,warm,false,
cup01_d080_nv.jpg,c01,cup,0.80,0.080,0.095,,nv,false,
bowl01_d090_white.jpg,bwl01,bowl,0.90,0.150,0.060,,white,false,
```

`object_id` is the stable key the batch eval groups on (`eval_by_object.csv`).
If the column is missing or blank the pipeline falls back to the filename prefix
(`bottle01_d050_white.jpg` -> `bottle01`), so keep the Section 7 pattern.

`augmented` stays `false` for **all 100** rows - every image is a real capture.
The column is kept only so a derived/filtered image (Section 11) could be
flagged and reported separately if one were ever added.

---

## 9. Shooting checklist (per image)

1. Confirm the camera has not moved and both cups are on their marks (0.40 /
   1.00 m) and visible.
2. Place the correct object (b01 ... bwl01) on the target floor mark (0.50 ...
   0.90 m), centered, so it does not cover either cup.
3. Set the lighting mode to the current cell (white | warm | dim | nv); in `nv`
   wait for night vision to settle before shooting.
4. Confirm the object is fully in frame at the fixed camera position.
5. Take the shot; keep the original file.
6. Rename it per the pattern in Section 7.
7. Log the ground-truth row immediately (Section 8).
8. Move to the next cell of the matrix.

Do a 5-image dry run first, run it through `test_pipeline.py` once, and confirm
the empty-scene anchors are exact and the middle object is detected before
committing to all 100.

---

## 10. Running the pipeline on the dataset

One command per image, with the two cup anchors and the settings that worked on
the reference rig (`--calib inverse` interpolates between anchors; `base` is
noticeably more accurate than the default `small` at these distances):

```bash
python test_pipeline.py --image bottle01_d050_white.jpg --calib inverse \
    --depth-model base --conf 0.2 \
    --anchor 569,721,0.40 --anchor 1270,577,1.00
```

Notes for this dataset:

- Use `--conf 0.2`: the cup / bowl and some bottles only score just over 0.2 as
  their COCO class, so the default 0.25 silently drops the very object being
  measured. Because the set mixes `bottle`, `cup` and `bowl`, do **not** pin
  `--target bottle` for the dataset run - size all detections and pick the
  middle box (`pick_middle` does that automatically in batch mode).
- COCO has `bottle`, `cup` and `bowl`, so the middle object should normally come
  back labelled correctly. It can still mis-label a low-contrast or `nv` frame
  (a bowl read as `basket`/`sink`, a cup as `bowl`). Record the detector's label
  in `notes`; the ground-truth `label` stays the controlled class.
- `--target` is still useful for a single-image check on one plain bottle
  (`--target bottle`), just not for a mixed-class dataset pass.
- Report Z/W/H error per object as well as overall: the batch run writes
  `eval_by_object.csv` next to `eval_by_lighting.csv` / `eval_by_distance.csv`,
  so the bowl (widest W) and the cup (shortest H) each get their own track.
- `--depth-model base` vs the default `small` changes Z by ~5 cm at 0.7 m, so
  keep the same model for the whole dataset and report which one was used.
- Both anchors are given in pixels, so YOLO failing to detect a cup in low light
  (`warm` / `nv`) does **not** break the calibration.

---

## 11. Augmentation: not used (reference only)

The 100 images are **all real captures** (Section 2: 5 objects x 5 distances x 4
lighting modes). Augmentation is **not** used to build the dataset and is **not**
a substitute for real photos: `Research_Proposals.md` requires 100 *photos taken
via smartphone* (Step 2: "Capture 100 **distinct** images") with the distance and
sizes *manually measured for each*, and a derived image is neither a distinct
capture nor newly measured. Keep `augmented=false` on every row.

This section is kept only as a record of what was considered and why it was
rejected, in case a reviewer asks. Even if it had been allowed it would have
been weak:

- Derived images share the 25 real setups, so the effective sample size stays
  small and a reviewer will question the generalization claim.
- MDE foundation models are largely illumination-invariant, so a global
  brightness/contrast change may barely move the depth map -> a flat,
  uninformative robustness result. The four **real** lighting modes in Section 4
  cover that axis for real, which is stronger evidence.

### 11.1 Safe vs unsafe transforms (reference)

Only **photometric** transforms are safe: they change pixel *values*, never
pixel *positions*, so the cup anchors (`anchor.txt`), `distance_m`, and
`true_w_m` / `true_h_m` all stay valid.

| Safe (geometry-preserving) | Unsafe (moves the anchors) |
|---|---|
| brightness / contrast, gamma | hflip, crop, resize |
| exposure + white-balance / color-temp shift | rotation, perspective / keystone |
| warm cast (`yellow`), grayscale (`mono`) | any scaling or digital zoom |
| sensor / ISO noise (Gaussian) | |
| slight Gaussian blur / defocus | mirroring maps `u -> W-1-u`, so `anchor.txt` is wrong |
| JPEG re-compression (low quality) | |
| vignette | |

Never apply an unsafe transform without transforming the anchor pixels by the
same matrix.

### 11.2 The two color filters (rejected)

Documented only because they were considered. Neither is used, because the
Section 4 lighting modes already give the useful part of both for real:

- **`yellow`** (warm color cast) is redundant with the real **`warm`** light
  mode - you shoot that colour shift with the actual lamp.
- **`mono`** (black & white) is not a real illumination condition, and the
  **`nv`** night-vision mode already provides a real near-monochrome capture.

If one is ever added: they are per-pixel color transforms, so they are
geometry-preserving and need no anchor change. Apply at most **one** per real
image so the ablation stays interpretable - do not stack `yellow` + `mono` +
noise on the same shot - and never count it toward the 100 (see the top of this
section).

**`yellow` - warm color cast.** Stands in for tungsten / sodium-vapor light or
yellowed plastic. Superseded by the real `warm` mode (Section 4), which captures
the same colour shift with an actual lamp.

```python
img[..., 2] *= 0.70   # B down
img[..., 1] *= 0.92   # G slightly down
# R unchanged -> warm/yellow cast
```

**`mono` - black & white.** Luminance replicated back to 3 channels. Not a real
illumination condition (cameras here shoot color), so it is a *color-invariance
ablation*: "does the pipeline still size correctly without color?" Expect the
depth map to move more than under `yellow`, since MDE models lean on
luminance/color structure.

```python
gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
img = np.stack([gray, gray, gray], axis=-1)
```

Bookkeeping for both: set `augmented=true`, copy `distance_m` and
`true_w_m` / `true_h_m` from the parent row unchanged, and record the filter as
the `lighting` value (`yellow` or `mono`; `warm` is reserved for the real warm
mode) or in `notes`. Name the files as `<parent>_<filter>.jpg`, e.g.
`bottle01_d050_white_yellow.jpg`.

---

## 12. Tooling still to build

- [x] `--anchor-file <path>` in `test_pipeline.py` so the two cup anchors can be
      read straight from `anchor.txt` instead of being retyped per image.
- [x] Per-folder anchors: every shot folder carries its own `anchor.txt`
      (`dataset/<lighting>/<object>/anchor.txt`) and the pipeline resolves the
      nearest one per image, so one re-shot scene cannot skew the rest.
      `label_anchors.py` renders them into `anchordataset/` for a visual check.
- [x] Batch mode in `test_pipeline.py` to process a whole dataset folder in one
      run: `--dataset <root>` scans recursively, auto-discovers each folder's
      `anchor.txt`, and `--depth-model all` runs small + base + large. Evaluation
      is written alongside: `eval_per_image.csv`, `eval_summary.csv`,
      `eval_by_lighting.csv`, `eval_by_distance.csv`, `eval_by_object.csv`
      (MAE / RMSE of Z on the middle object), so no separate `eval_accuracy.py`
      is needed.
- [ ] A check that flags an image where a cup's sampled depth drifted far from
      its session baseline (mis-aimed camera or moved cup).

### One-command batch run (current)

```bash
# whole dataset, three depth models, auto anchors + auto eval:
python test_pipeline.py --dataset dataset --depth-model all --calib inverse
```

Results land under `outputs/<model>/<lighting>/<object>/`, with the evaluation at
the top of `outputs/`. Ground truth is read from `ground_truth.csv` if present,
otherwise the true distance is parsed from the filename and the lighting from the
folder.

