# Wrapup: Zero-Shot Monocular Depth Sizing — Full Dataset Run (corrected)

Generated from `outputs_clean/` — 5 depth backbones x 100 images, scored after the
evaluation was fixed to only ever score the object each folder is about.

Reproduce exactly:

```bash
python test_pipeline.py --dataset dataset --depth-model all --calib inverse \
    --yolo yolov8m.pt --out outputs_clean
```

| item | setting |
|---|---|
| dataset | `dataset/` — 4 lighting modes x 5 object folders x 5 distances (0.5-0.9 m) = 100 images |
| calibration | `--calib inverse`, 2 anchors per folder (0.40 m and 1.00 m cups) from each folder's `anchor.txt` |
| YOLO | `yolov8m.pt` for every model, conf 0.25 |
| depth models | Depth-Anything-V2 small / base / large (`-hf`), `Intel/dpt-hybrid-midas`, `prs-eth/marigold-depth-v1-0` |
| depth sampling | `--depth-sample box_median` (median over the interior 60 % of the box) |
| hardware | RTX 4070 12 GB |
| camera | 1920x1080, `fx=fy=1506.9` from an assumed 65 deg HFOV (no usable EXIF) |

**Scored vs voided:** 475 rows scored, **25 voided**. A row is voided when the
subject class is never detected in that image — it is excluded from MAE, RMSE,
every grouped table and the outlier counts, and logged in `eval_voided.csv`.
Every scored row's `label` now equals its folder's object class (0 mismatches).

Metrics: **MAE Z** in cm, **MAE rel** in %, **W% / H%** = relative error of the
sized width / height against `dataset/objects.txt`.

---

## 0. Headline

**Depth-Anything-V2-Large wins clearly** at 3.63 cm MAE (4.9 %), with **zero**
detections worse than 20 cm or 30 %. Base (5.35 cm) and small (5.78 cm) follow,
then Marigold (8.55) and MiDaS (9.38).

Three things dominate the error budget:

1. **b01 (the tall thin bottle)** — **61 %** of all absolute Z error, and almost
   entirely a one-directional "too far" bias. The main research finding and the
   main thing left to fix.
2. **`nv` (night)** — 1.8x the error of the best lighting mode (7.29 vs 4.01 cm,
   Depth-Anything pooled), and home to 14 of the 30 remaining outliers.
3. **Marigold and MiDaS are not competitive** on Z, and Marigold's number is not
   reproducible between runs (section 8).

Without b01 the ranking flattens dramatically — **small (1.95 cm) actually beats
large (2.23 cm)** on the other four objects.

---

## 1. Overall performance by model

| model | n | void | MAE Z (cm) | MAE rel (%) | bias (cm) | RMSE Z (cm) | >20 cm | >30 % | W err (%) | H err (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| **large** | 95 | 5 | **3.63** | **4.9** | +3.21 | **5.15** | **0 %** | **0 %** | **9.3** | **4.9** |
| base | 95 | 5 | 5.35 | 7.4 | +4.99 | 7.83 | 5 % | 1 % | 11.8 | 6.1 |
| small | 95 | 5 | 5.78 | 8.0 | +4.77 | 10.71 | 12 % | 7 % | 12.9 | 8.3 |
| marigold | 95 | 5 | 8.55 | 12.1 | -3.12 | 11.62 | 3 % | 2 % | 14.3 | 13.5 |
| midas | 95 | 5 | 9.38 | 13.1 | +5.26 | 15.91 | 12 % | 12 % | 15.5 | 11.1 |

- **Depth-Anything-V2 sweeps every column.** Small/base/large beat both baselines
  on Z accuracy and on width/height sizing. Clearest result in the study.
- **Large is the only model with no gross failures.** 0 % of its detections
  exceed 20 cm or 30 %, and its RMSE (5.15) sits close to its MAE — errors are
  small and evenly spread rather than a few catastrophes. Small's RMSE is nearly
  2x its MAE (10.71 vs 5.78), i.e. a handful of bad rows drive it.
- **Base beats small overall** (5.35 vs 5.78) because it handles b01 and nv
  better; small only wins on the easy objects. If cost is not the constraint,
  take large; if it is, base is a better default than small.
- **Every model over-estimates distance** except Marigold. Mean `Zpred/Ztrue`:
  1.062 (small), 1.067 (base), 1.042 (large), 1.075 (midas), 0.949 (marigold).
  A single multiplicative correction would buy ~4-7 points of relative error for
  free, since W and H scale with Z.
- **Marigold beats MiDaS** here (8.55 vs 9.38) but at diffusion-sampling cost and
  with non-reproducible variance — see section 8. MiDaS is not competitive on
  anything except being the least biased on the b01 bottle.

**On the easy subset (everything except b01):**

| model | n | MAE Z (cm) | MAE rel (%) | W err | H err |
|---|---|---|---|---|---|
| **small** | 75 | **1.95** | **2.8** | 6.2 % | 4.6 % |
| large | 75 | 2.23 | 3.0 | 6.0 % | 4.2 % |
| base | 75 | 3.08 | 4.2 | 7.1 % | 3.9 % |
| marigold | 75 | 7.66 | 11.0 | 12.0 % | 12.6 % |
| midas | 75 | 9.23 | 12.9 | 15.6 % | 10.8 % |

Small edges out large here. Large's overall win is earned entirely on the bottle
and at night. On disk small's weights are 99 MB against large's 1.34 GB (13x).

---

## 2. Performance by lighting condition

MAE Z (cm) / MAE rel (%) per cell.

| lighting | small | base | large | midas | marigold |
|---|---|---|---|---|---|
| **dim** | 4.47 / 6.1 % | 4.66 / 6.3 % | **2.90 / 3.9 %** | 5.42 / 8.0 % | 9.41 / 13.4 % |
| **white** | 4.45 / 6.1 % | 4.73 / 6.5 % | 3.42 / 4.5 % | 12.32 / 17.1 % | 6.39 / 9.4 % |
| **warm** | 5.16 / 7.2 % | 5.08 / 7.1 % | 3.72 / 5.1 % | 6.93 / 10.1 % | 7.94 / 11.5 % |
| **nv** | 9.84 / 13.9 % | 7.33 / 10.3 % | 4.69 / 6.6 % | 13.73 / 18.3 % | 10.95 / 14.7 % |

Ranking (Depth-Anything-V2 pooled): **dim (4.01) < white (4.20) < warm (4.65)
<< nv (7.29 cm)**.

- **`dim` is marginally the easiest condition, not the hardest.** Consistent
  across the Depth-Anything family (and only just ahead of white). Plausible
  mechanism: darkening the frame puts a bright object on a dark background, the
  cleanest figure/ground separation for a monocular depth model. The A/B that
  would confirm it is `preprocessing.py adjust --in-place` vs the unadjusted
  originals.
- **`white` and `warm` are 0.6 cm apart.** The yellow cast costs little on Z but
  slightly more on width (warm W 10.5 % vs white 9.8 % for small).
- **`nv` is the failure mode.** 1.8x the error of dim for large, 2.2x for small,
  holding 14 of the 30 remaining outliers. Large degrades gracefully (4.69 cm,
  0 outliers); small (9.84 cm, 12 % over 30 %) and MiDaS (13.73 cm) do not.
- Measured global luminance says contrast is **not** the driver: nv has the
  highest pixel std (52.3) and dim the lowest (21.2), yet nv is worst and dim
  best. nv frames are high-contrast because they are **clipped** (p95 luminance
  208.8, p5 28.8) — blown highlights, crushed shadows, lost mid-tones.

---

## 3. Performance by object category

MAE Z (cm) / rel (%). a01 has n=15 per model because its 5 night images are
voided — the apple is never detected there (section 7).

| object | true W x H | small | base | large | midas | marigold |
|---|---|---|---|---|---|---|
| **bwl01** bowl | 12.0 x 6.0 cm | **1.23 / 1.9 %** | 1.73 / 2.3 % | 1.67 / 2.2 % | 7.13 / 9.0 % | 8.74 / 12.5 % |
| **c01** cup | 12.0 x 8.0 cm | 2.42 / 3.5 % | 1.43 / 2.2 % | **1.21 / 1.8 %** | 3.77 / 5.1 % | 6.30 / 9.1 % |
| **a01** apple | 7.5 x 7.0 cm | **1.57 / 2.2 %** | 2.95 / 3.9 % | 2.37 / 3.1 % | 4.63 / 6.7 % | 9.20 / 13.2 % |
| **b02** bottle (short) | 7.5 x 25.0 cm | **2.49 / 3.6 %** | 6.17 / 8.4 % | 3.73 / 4.8 % | 20.23 / 29.0 % | 6.77 / 9.6 % |
| **b01** bottle (tall/thin) | 6.2 x 24.0 cm | 20.12 / 27.3 % | 13.89 / 19.4 % | **8.86 / 12.4 %** | 9.97 / 14.2 % | 11.92 / 16.5 % |

Share of total absolute Z error (Depth-Anything-V2 pooled):

| object | share |
|---|---|
| **b01** | **61.1 %** |
| b02 | 17.7 % |
| a01 | 7.4 % |
| c01 | 7.2 % |
| bwl01 | 6.6 % |

- **Broad, squat objects are solved.** Bowl and cup sit at 1.2-2.4 cm / under
  4 % for every Depth-Anything variant. Genuinely usable for sizing.
- **Thin objects are not.** b01 alone is 61 % of the error in the whole study,
  and it is a bias problem, not a precision problem (section 5).
- **The apple is easy when it is detected at all** (1.57 cm for small), which
  makes its night detection failure the more striking.
- **MiDaS has an inverted object profile.** Worst model on b02 by a wide margin
  (20.23 cm, 29 %) where Depth-Anything is comfortable, yet the *least biased* on
  b01. Its errors are distributed differently, not simply larger.

---

## 4. Width/height sizing accuracy

The actual deliverable is dimensions, not Z.

| model | W err | H err |
|---|---|---|
| **large** | **9.3 %** | **4.9 %** |
| base | 11.8 % | 6.1 % |
| small | 12.9 % | 8.3 % |
| marigold | 14.3 % | 13.5 % |
| midas | 15.5 % | 11.1 % |

- **Height is consistently more accurate than width** for every model (roughly
  half the relative error). Expected: height is the long axis of most of these
  objects, so a box error is proportionally smaller, and vertical edges are
  cleaner against a table surface.
- **Width error > Z error on both bottles** — a box-tightness signature, not a
  depth signature. Points of W error beyond the Z error, pooled over models:

  | object | Z err | W err | extra W from box | extra H |
  |---|---|---|---|---|
  | b02 | 11.1 % | 18.6 % | **+7.5** | -0.7 |
  | b01 | 18.0 % | 25.4 % | **+7.4** | -3.4 |
  | bwl01 | 5.6 % | 6.9 % | +1.3 | +0.5 |
  | c01 | 4.4 % | 5.2 % | +0.8 | +0.3 |
  | a01 | 5.8 % | 6.1 % | +0.3 | +2.3 |

  So the bottle width error is roughly 40 % box, 60 % depth. The apple's is
  entirely depth — do not go box-hunting there. Height shows no such excess
  anywhere, i.e. vertical box edges are reliable.

---

## 5. Where the error comes from: the b01 bottle

b01 is 61 % of the error budget and it is **systematic**. Signed Z error by
distance, positive = predicted too far:

| model | 0.5 m | 0.6 m | 0.7 m | 0.8 m | 0.9 m |
|---|---|---|---|---|---|
| small | +7.95 | +11.64 | +25.42 | +23.75 | +31.85 |
| base | +7.98 | +10.87 | +14.26 | +14.97 | +21.37 |
| large | +5.34 | +6.72 | +8.92 | +9.26 | +14.05 |
| midas | +0.67 | -1.75 | -0.45 | +1.50 | +1.06 |
| marigold | +1.15 | +2.98 | -2.59 | +7.04 | +14.90 |

For every Depth-Anything model the signed bias ≈ the MAE (large b01: MAE 8.86,
bias +8.86) — **essentially 100 % of the b01 error is one-directional
over-estimation**, growing with distance. At 0.9 m small thinks the bottle sits
at ~1.32 m.

**Diagnosis: depth sampling, not the depth model.** The default
`--depth-sample box_median` takes the median over the interior 60 % of the box.
For a tall (24 cm) thin (6.2 cm) object a slightly loose box pulls background
columns in on both sides; background is farther, so the median drifts away. b01
is also the loosest detection in the set (confidence down to 0.67).

**Verified fix.** `dataset/nv/b1`, small model, inverse calibration, same anchors:

| depth-sample | MAE Z (cm) | MAE rel (%) |
|---|---|---|
| box_median (what this run used) | 28.54 | 39.1 |
| box_center | 26.12 | 35.5 |
| **bottom_center** | **17.03** | **23.0** |

`bottom_center` samples the ground-contact point, geometrically the right place
for a standing object and immune to the loose-box problem: a **40 % MAE
reduction on the object that is 61 % of the error**, for free. It should be the
default for tall objects, or selected by aspect ratio.

Caveat: that sweep used the default YOLO nano (it labels some b01 frames
`vase`), so the absolute numbers are not directly comparable to the yolov8m run.
The ordering is the finding.

---

## 6. Error vs distance

MAE Z (cm) across all objects:

| model | 0.5 m | 0.6 m | 0.7 m | 0.8 m | 0.9 m |
|---|---|---|---|---|---|
| small | 3.01 | 4.18 | 6.47 | 6.29 | 8.93 |
| base | 2.86 | 3.66 | 5.75 | 6.17 | 8.32 |
| large | 1.86 | 2.18 | 3.42 | 4.45 | 6.24 |
| midas | 5.85 | 6.49 | 9.94 | 11.44 | 13.20 |
| marigold | 4.69 | 7.91 | 10.03 | 10.32 | 9.82 |

- Absolute error grows monotonically with distance for every model except
  Marigold. Relative error grows much more slowly than absolute error (~5-7 %
  for large), the expected signature of a correctly scaled inverse-depth fit: the
  geometry is sound, resolution and sampling are the limits.
- **Effective working range on this rig: 0.5-0.8 m** (large stays under 4.5 cm).
  At 0.9 m it starts to slip. Nothing here supports sizing beyond 1 m.
- The non-monotonic 0.8 m point seen in the uncorrected evaluation is gone — it
  was b01 rows being scored against an anchor cup.

---

## 7. What was voided, and why that matters

**25 of 500 (model, image) pairs voided — every one of them apple-at-night.** No
other folder had a single void.

| image | expected | what YOLOv8m actually found |
|---|---|---|
| a01_50cm nv | apple | cup |
| a01_60cm nv | apple | cup, knife |
| a01_70cm nv | apple | cup |
| a01_80cm nv | apple | bowl, cup |
| a01_90cm nv | apple | cup, person |

The apple is not in the detection list on any night frame — a01 is the only
object with a 0/5 detection rate in any lighting mode. Those rows are recorded in
`eval_voided.csv` and excluded from every metric.

**This also removed a silent-failure mode.** The pipeline previously called
`pick_middle()` with no class filter, so it scored whatever was most confident
between the two anchors — the calibration cups themselves, or a knife, bowl or
dining table. Because a row's ground-truth distance comes from the filename, that
produced guaranteed 10-50 cm "errors" with nothing to do with the depth model.
That is what inflated the original `eval_*.csv` (appendix).

The fix in `test_pipeline.py`:

- `pick_middle(results, anchors, expected)` drops any detection whose box
  contains an anchor click (the reference cups are clicks *on* those cups, so
  containment is exact), filters to the expected class, and **returns None** when
  the subject was never detected.
- The expected class is derived per folder from the object id
  (`expected_class_for`: a01 -> apple, b01/b02 -> bottle, bwl01 -> bowl,
  c01 -> cup), or from `--target` when given.
- A voided row is skipped, counted, and written to `eval_voided.csv`;
  `eval_summary.csv` gained an `n_voided` column.

---

## 8. Caveats before this goes in a paper

- **Marigold is not reproducible as configured.** The diffusion pipe is called
  without `generator=` or fixed latents, so every run starts from different
  noise. Its number moved from 17.46 (original run) to 8.55 (this run) — part
  voiding, part pure sampling variance. It also ran at 10 steps / ensemble 1 /
  768 px rather than the paper's 50 / 10. **Fix the seed and the schedule before
  quoting any Marigold number.**
- **`--calib inverse` is the wrong fit for Marigold** — the script warns about
  this itself: Marigold predicts direct depth (larger = farther), so the inverse
  fit suits it poorly. MiDaS and Marigold should ideally run with `--calib
  affine`. Keeping inverse here preserves comparability with the original run; it
  does not give Marigold its best shot.
- **One camera, one assumed FOV.** `fx` comes from an assumed 65 deg HFOV because
  EXIF had nothing usable. Real intrinsics would move every absolute number.
- **One shot per condition.** 5 distances x 4 lightings x 5 objects, one image
  each. No repeated captures, so no measurement of shot-to-shot variance.
- **b01's folders use the handle-less cups (8x8 cm)** while every other folder
  uses the 12x8 cm pair. Both are only the 0.40 / 1.00 m calibration constants, so
  this is fine, but the folders are not identical setups.
- The `nv` failure looks like highlight clipping. Until that is tested with an
  exposure bracket, "monocular depth fails at night" is not separable from "this
  camera blew the highlights at night".

---

## 9. Recommendations

Ordered by expected value per unit of work.

1. **Switch depth sampling to `bottom_center`** (or make it aspect-dependent).
   Verified 40 % MAE improvement on the object that is 61 % of your error. Costs
   nothing; do this next.
2. **Fix Marigold's seed and schedule** (or drop it from the comparison). As it
   stands its number is not defensible in either direction. Also run it with
   `--calib affine`.
3. **Apply a scale correction.** Every model is off by a consistent +4 to +7 %
   (Marigold -5 %). One constant fitted on the calibration anchors removes it and
   improves W/H for free, since both scale with Z.
4. **Lead with large** for the headline — the only model with zero gross
   failures — but be explicit that small beats it by 0.3 cm on four of five
   objects at 13x smaller weights. That is a real deployment finding.
5. **Tighten boxes on the two bottle classes.** ~40 % of their width error is box
   geometry that depth cannot fix. Segmentation or tighter box regression would
   help; nothing else on this list will.
6. **Report the object profile, not just the aggregate.** 61 % of the aggregate
   error comes from one object. "Bowls and cups to ~2 %, bottles need work" is the
   honest and more useful claim.
7. **Re-test night with an exposure bracket** to separate clipping from any
   inherent limitation, and only then decide whether to invest in night handling.

## 10. What I would test next

- Full re-run with `--depth-sample bottom_center` to confirm the b01 fix
  generalises beyond the 5 nv images it was measured on.
- Per-light `--depth-sample` sweep — warm may prefer a different mode than nv.
- Marigold at 50 steps / ensemble 10 / fixed seed on a subset, to price its
  accuracy against its runtime fairly.
- Box-tightening experiment on b01/b02 (small segmentation model, or a regression
  head) to attack the remaining +7.4 points of width error.
- A night capture with bracketed exposure, and a night capture under continuous
  artificial light, to test the clipping hypothesis.

---

## Appendix: what the original numbers said, and why they were wrong

The first run's `outputs/eval_*.csv` reported much worse numbers:

| model | old MAE Z (raw) | corrected MAE Z | old RMSE | corrected RMSE |
|---|---|---|---|---|
| small | 7.23 | **5.78** | 14.11 | 10.71 |
| base | 6.83 | **5.35** | 11.56 | 7.83 |
| large | 5.66 | **3.63** | 11.19 | 5.15 |
| midas | 12.13 | **9.38** | 22.14 | 15.91 |
| marigold | 17.46 | **8.55** | 64.68 | 11.62 |

Cause: 51 of 500 rows (10.2 %) scored the wrong object — 36 x cup (the
calibration anchors themselves), 5 x knife, 5 x bowl, 5 x dining table. Those
rows averaged **39.51 cm** error against 6.50 cm for correct rows, and supplied
60-90 % of each model's squared error. One row alone — `a01_60cm` nv picking a
`knife` and predicting 668 cm — was 88.4 % of Marigold's entire squared error.

The distortion was worst in the per-object table: raw a01 was 18.20 cm MAE and
looked like the worst object; correctly scored it is 2.29 cm pooled and third
best. Marigold's height error roughly halved (47.4 % -> 13.5 %).

The old per-image rows are still in `outputs/` for reference. `outputs_clean/` is
the authoritative set.
