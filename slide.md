# Mid-Term Presentation — Slide Content

---

## 1. Title Slide

**Title:**
Zero-Shot Monocular Depth Estimation for Automated Crop/Object Sizing in Low-Contrast Environments

**Subtitle:**
Turning a single RGB image into real-world width, height, and area with YOLOv8, Depth Anything V2, and a geometric mapping engine

**Presenter line:**
[Your Name] · [Program / Department] · [Date]
Mid-Term Progress Review

---

## 2. Problem & Motivation

**Headline:**
Measuring object size in the field is still manual, slow, and inconsistent.

**The problem:**
- Crop grading, inventory, packaging, and robotic picking all need distance and physical size.
- Manual tape measurement does not scale and is impossible for a moving or remote camera.
- Depth hardware is expensive, fragile, short-range, and needs calibration.
- Ordinary RGB cameras are already everywhere.

**The gap:**
- Most depth benchmarks use controlled indoor scenes, not low-contrast real captures.
- There is no simple pipeline that reports real-world size error across distance and lighting.

**Approach:**
- Frozen YOLOv8 detector + frozen Depth Anything V2 depth model.
- Two known anchor distances turn relative depth into metric measurements.
- No model training, no task-specific fine-tuning.

**Research question:**
How accurately can a zero-shot depth model, paired with a detector and minimal anchor calibration, recover object distance and size across distance and illumination?

---

## 3. Introduction to YOLO & Depth Model

**Headline:**
Two frozen pre-trained models, each doing one job.

**YOLOv8 — object detection:**
- Finds each object and returns a bounding box, class, and confidence.
- Pre-trained on COCO; the relevant classes here are `bottle`, `cup`, and `bowl`.
- Frozen and used zero-shot, with no fine-tuning.

**Depth Anything V2 — monocular depth:**
- Converts one RGB image into a relative near/far depth map.
- The output is relative, not metric; it needs calibration to become meters.
- Three sizes, all tested here:

| Variant | Trade-off |
|---|---|
| Small | Fastest, lower accuracy |
| Base | Balanced |
| Large | Slowest, highest accuracy |

- MiDaS is included as a classical control baseline.

**Why combine them:**
- YOLO answers *what* and *where*; depth answers *near or far*; the mapping engine turns both into measurements.

---

## 4. Pipeline Architecture

**Headline:**
An end-to-end pipeline combines off-the-shelf perception with a geometric mapping engine.

**Pipeline stages:**
1. **Capture:** one ordinary RGB photo.
2. **Detect:** YOLOv8 finds each object and draws its box.
3. **Estimate depth:** Depth Anything V2 produces a relative near/far map.
4. **Map to metric:** the geometric mapping engine converts model output into real measurements.
5. **Deliver:** annotated image, depth visualization, per-object measurements, and batch evaluation.

**What the geometric mapping engine does:**
- **Calibrate:** learns the depth-to-distance scale from two reference cups at 0.40 m and 1.00 m.
- **Locate:** assigns a real-world distance to every detected object.
- **Size:** converts the object's pixel footprint at that distance into centimeters.
- **Batch mode:** each scene uses its own anchor file; one command runs a single image or the full dataset.

---

## 5. Experiment Setup / Data Collection

**Headline:**
A controlled capture matrix isolates distance and lighting as the variables.

**Capture matrix (100 images):**

| Axis | Levels | Count |
|---|---|---|
| Objects | 3 bottles, 1 cup, 1 bowl | 5 |
| Distance | 0.50 to 0.90 m, in 10 cm steps | 5 |
| Lighting | white, warm, dim, night vision | 4 |
| **Total** | **5 × 5 × 4** | **100** |

**Rig and ground truth:**
- Camera locked on a tripod; height, zoom, and resolution never change.
- Two reference cups fixed at 0.40 m and 1.00 m.
- Measured object placed at 0.50, 0.60, 0.70, 0.80, or 0.90 m.
- Distance and true width/height recorded with a tape measure.

**Lighting modes:**

| Mode | Condition |
|---|---|
| White | Bright, even reference case |
| Warm | Orange colour cast |
| Dim | Low light, flat and noisy |
| Night vision | Lights off, IR illuminator, near-monochrome |

**Current progress:**
- 40 pilot images collected: cup and bottle across four lighting modes and five distances.
- This deck reports the white-light cup slice: 5 distances × 3 depth models.

---

## 6. Evaluation — Depth Model Results (Distance)

**Headline:**
Depth Anything V2 Large averages **0.76 cm error (1.12%)** on the white-light cup sequence.

**Setup:** Cup c01, white light, 0.50 to 0.90 m (n = 5 per model). YOLOv8n, 65° hfov, inverse-disparity calibration on the 0.40 / 1.00 m anchors.

| Model | MAE | Mean error | Width error | Height error |
|---|---|---|---|---|
| Small | 2.07 cm | 3.00% | 0.38 cm / 3.2% | 0.39 cm / 4.9% |
| Base | 1.16 cm | 1.82% | 0.25 cm / 2.1% | 0.30 cm / 3.7% |
| Large | **0.76 cm** | **1.12%** | **0.24 cm / 2.0%** | **0.23 cm / 2.9%** |

Size error is against a tape-measured 12 × 8 cm cup. RMSE: 2.39 / 1.30 / 0.96 cm.

**Distance error by range (absolute):**

| Model | 0.50 m | 0.60 m | 0.70 m | 0.80 m | 0.90 m |
|---|---|---|---|---|---|
| Small | 2.4 cm | 1.5 cm | 1.4 cm | 0.8 cm | 4.2 cm |
| Base | 1.9 cm | 1.2 cm | 0.9 cm | 0.2 cm | 1.6 cm |
| Large | 1.1 cm | 0.6 cm | **0.02 cm** | 0.3 cm | 1.7 cm |

**Key findings:**
1. **Bigger model, better numbers.** MAE falls 2.07 → 1.16 → 0.76 cm; Large holds size error under 3%.
2. **0.70 to 0.80 m is the sweet spot.** Large is near-exact at 0.70 m and within 0.3 cm at 0.80 m.
3. **Edges fail predictably.** All models read short below 0.70 m and long at 0.90 m (Small +4.2 cm). All 5 images detected; reference cups tracked to ~1 cm.

**Bottom line:** Large gives sub-centimeter distance error with no training, no stereo, no depth sensor.

---

## 7. Comparing Depth Model Results (White vs Night Vision)

**Headline:**
Night vision hurts the detector far more than the depth model. Four of five shots still size within ~2 cm; the 11 cm average comes from one missing cup.

**Setup:**
Same rig, cup c01 at 0.50 to 0.90 m, IR illuminator with the lights off. Same anchors, same inverse calibration as the white run.

**Distance error, white vs night vision (MAE, cm):**

| Model | White (all 5) | NV (all 5) | NV, 0.50 to 0.80 m |
|---|---|---|---|
| Small | 2.07 | 12.18 | 2.96 |
| Base | 1.16 | 10.92 | **1.36** |
| Large | **0.76** | 11.51 | 2.08 |

**NV error by distance (cm):**

| Model | 0.50 m | 0.60 m | 0.70 m | 0.80 m | 0.90 m |
|---|---|---|---|---|---|
| Small | 3.2 | 3.7 | 4.1 | 0.8 | miss* |
| Base | 2.4 | 1.4 | 1.4 | 0.2 | miss* |
| Large | 2.6 | 2.5 | 1.4 | 1.9 | miss* |

\* c01 was not detected at 0.90 m, so the pipeline sized the 0.40 m reference cup instead (Z = 40.8 cm, 49 cm error). That single frame is what pushes the NV average from ~2 cm to ~11 cm.

**Detector behaviour:**

| | White | Night vision |
|---|---|---|
| Subject detected | 5 / 5 | 4 / 5 |
| Subject confidence | 0.67 to 0.81 | 0.30 to 0.75 |
| Wrong-class boxes | 3 (`dining table`) | 9 (`toilet`, all on the 1.00 m reference cup) |

**Key findings:**
1. **Depth survives the dark, detection does not.** On the four shots it caught, Large moves from 0.53 cm (white, same range) to 2.08 cm, still centimetre-scale.
2. **IR falloff is the failure mode.** c01 confidence falls 0.77 (white 0.50 m) to 0.30 (NV 0.80 m) to 0.07 at 0.90 m, where it only appears as `bowl` / `toilet` below a 0.10 threshold. Lowering `--conf` does not rescue it, it just adds junk boxes.
3. **The ranking flips.** Base beats Large in NV (1.36 vs 2.08 cm): IR strips the texture cues Large relies on.
4. **Overlabelling is systematic, not random.** The far reference cup is read as `toilet` at 0.31 to 0.40 conf in 9 of 15 frames.

**Bottom line:** Night vision needs a detector-side fix (better IR illumination, a class whitelist, or a detector tuned on IR). The depth and calibration stage is not the bottleneck.

Data: `outputs_nv/eval_summary.csv`, `outputs_nv/eval_by_distance.csv`, `outputs_nv/eval_per_image.csv`.

---

## 8. Comparing Depth Model Results (Opaque vs Transparent)

**Headline:**
Transparency, not model size, is the limit. Large scores 0.76 cm on the opaque cup but 8.09 cm on the clear bottle at the same distances (1.12% vs 11.14%).

**Setup:**
Bottle b01 (6.2 × 24 cm, clear plastic) against the cup c01 (12 × 8 cm, opaque). Both white light, 0.50 to 0.90 m, same rig, anchors and calibration. The detector is clean here: 5 / 5 shots, confidence 0.81 to 0.93, so this isolates the depth model.

**Mean distance error, same rig and range (cm):**

| Model | Opaque cup | Transparent bottle | Cup % | Bottle % |
|---|---|---|---|---|
| Small | 2.07 | 15.30 | 3.00% | 20.38% |
| Base | 1.16 | 11.63 | 1.82% | 15.92% |
| Large | **0.76** | **8.09** | **1.12%** | **11.14%** |

**Bottle error by distance (cm):**

| Model | 0.50 m | 0.60 m | 0.70 m | 0.80 m | 0.90 m |
|---|---|---|---|---|---|
| Small | 5.2 | 6.1 | 19.0 | 20.4 | 25.8 |
| Base | 7.0 | 6.1 | 12.1 | 11.4 | 21.5 |
| Large | **4.6** | **6.1** | **7.3** | **7.4** | **15.1** |

**Size error (true 6.2 × 24 cm bottle, Large):** width 1.16 cm / 18.6%, height 1.71 cm / 7.1% (cup was 2.0% and 2.9%).

**Key findings:**
1. **Roughly 10× the error on the same rig.** Every model degrades the same way, so it is the object, not the configuration.
2. **Every prediction reads long, on all 15 shots.** The bottle inherits the depth of the wall behind it: in the depth map it barely separates from the background, while the opaque cup pops out clearly.
3. **The 0.60 m frame is the smoking gun.** All three models return almost identical values there (0.6607, 0.6608, 0.6607 m). Models that differ 2× on opaque objects agree to 0.2 mm here, so the bias is scene-driven, not model noise.
4. **Error grows with distance, and width suffers most.** Large goes 4.6 cm at 0.50 m to 15.1 cm at 0.90 m; width error reaches 18.6% because the narrow box depends entirely on a Z that is already biased.

---

## 9. Conclusion

**Key findings:**
1. **Sub-centimetre accuracy, no training.** Large reaches **0.76 cm MAE (1.12%)** distance error and under 3% size error on the opaque cup in white light, single RGB image, two anchors, two frozen models.
2. **Model size scales predictably.** MAE falls 2.07 → 1.16 → 0.76 cm; accuracy peaks at 0.70–0.80 m.
3. **Depth survives the dark; detection does not.** NV depth holds ~2 cm on the four frames caught; the 11 cm average comes from one missed cup, not depth failure.
4. **Transparency is the dominant error.** Clear bottle costs ~10× the opaque cup (8.09 vs 0.76 cm) on identical rig and calibration; all models converge to the same wrong value at 0.60 m, so the bias is scene-driven.


**Next steps:**
1. Complete the full 100-image matrix across all lighting × distance × object combinations.
2. Fix the NV detector side: better IR illumination, class whitelist, or IR-tuned detector.
3. Add transparency-aware correction (segmentation cue or background-disparity subtraction).
4. Test calibration sensitivity to anchor placement and spacing.
5. Write a short paper documenting the project

## 10. Q&A
**Questions and discussion.**
