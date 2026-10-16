# Project Proposal: Computer Vision & Depth Detection Track

**Project Title:** Zero-Shot Monocular Depth Estimation for Automated
Crop/Object Sizing in Low-Contrast Environments

## Project Details

We will leverage a pre-trained monocular depth estimation model to map
2D images into spatial depth maps without fine-tuning the model's weights. By
combining this depth map with a lightweight bounding-box object detector (like
YOLOv8), we will build a geometric pipeline to estimate the physical
dimensions (volume/bounding area) of target objects from a single standard RGB
camera feed.

- **Core Software Tools:** Python, PyTorch, Hugging Face Transformers (for
  depth model access), and OpenCV.
- **The Novelty Hook:** Instead of a novel architecture, the "paper
  contribution" lies in benchmarking how zero-shot foundation models generalize
  to a specific, challenging, un-mined physical environment compared to costly
  physical depth hardware.

## Work Requirements

- **A Custom 100-Image Dataset:** A targeted set of 100 photos taken via
  smartphone of a specific object class under diverse lighting (e.g., indoor
  potted plants, small machinery components, or grocery items on a shelf).
- **Physical Ground Truth:** must manually measure the physical
  distance and sizes of these items with a ruler/tape measure to serve as the
  evaluation baseline.
- **No Training Infrastructure:** A simple laptop or a free Google Colab tier
  is completely sufficient since no model training is occurring.

## Step-by-Step Execution Plan

### Phase 1: Setup & Ground Truth Collection (Weeks 1-2)

- **Step 1:** Establish the problem environment (e.g., measuring the volume of
  desktop packages or sizing plants).
- **Step 2:** Capture 100 distinct images, meticulously recording the exact
  camera distance and real-world dimensions for each.
- **Step 3:** Set up a Python script to pass these images through a pre-trained
  zero-shot depth model to extract raw relative depth arrays.

### Phase 2: Geometric Mapping Logic (Weeks 3-4)

- **Step 4:** Implement a standard trigonometric conversion pipeline using known
  camera intrinsic matrices to translate pixel depth values into metric units
  (meters/centimeters).
- **Step 5:** Combine the depth profile with an object detection mask to
  calculate the real-world width and height of the isolated object.
- **Step 6:** Calculate error matrices (Root Mean Square Error, Mean Absolute
  Error) between the computed values and the manual ground truth measurements.

### Phase 3: Validation Table & Drafting (Weeks 5-6)

- **Step 7:** Generate an evaluation table showing performance drops/gains under
  distinct lighting conditions or object distances.
- **Step 8:** Draft a 3-to-4-page conference short paper. The methodology
  section will easily write itself, focusing entirely on the geometric mapping
  applied to the foundation model outputs.

## Target Results & Deliverables

- **A Geometric Code Pipeline:** A plug-and-play Python script converting
  relative 2D depth predictions into metric size boundaries.
- **Performance Evaluation Matrix:** A clear data breakdown validating exactly
  where zero-shot depth estimation models succeed or fail in specialized
  real-world scenarios.
- **Short Paper Draft:** A highly targetable manuscript optimized for an IEEE
  regional conference with rapid-review tracks or application-focused signal
  processing events like IEEE ICIP, complete with abstract, geometric mapping
  formulas, evaluation tables, and references.

## Part 1: Metric Depth Conversion Formulation

A standard Monocular Depth Estimation (MDE) foundation model outputs a relative
depth map ($D_{\text{rel}}$) where values typically range from 0 (closest) to 1
(farthest), or represent unscaled inverse depth. To convert these relative pixel
values into physical metric dimensions (meters), must apply a
geometric camera model using camera intrinsic parameters.

Assuming the camera is calibrated using the standard pinhole camera model, the
transformation involves two phases: linear alignment and back-projection.

### Phase 1: Affine Alignment to Metric Depth

If the model outputs relative depth $D_{\text{rel}}(u, v)$ at pixel coordinates
$(u, v)$, it must be mapped to absolute metric depth $Z(u, v)$ using a linear
transformation based on known ground-truth anchor points:

$$Z(u,v) = \alpha \cdot D_{\text{rel}}(u,v) + \beta$$

How finds $\alpha$ and $\beta$:  samples a few known
physical distances from their ground-truth data (e.g., the center pixel distance
measured manually with a tape measure) and solves this simple linear system
using least-squares regression.

### Phase 1b: Inverse-Depth (Disparity) Alignment (implementation note)

Monocular depth models such as Depth Anything V2 emit values proportional to
*inverse* depth (disparity $1/Z$), not to $Z$ — consistent with the note above
that they "represent unscaled inverse depth". Fitting the affine model directly
on $Z$ therefore warps any object that lies *between* two calibration anchors,
because the true relationship is close to linear in $1/Z$, not in $Z$.

Fitting the disparity instead is far more accurate:

$$\frac{1}{Z(u,v)} = \alpha \cdot D_{\text{rel}}(u,v) + \beta
\quad\Longrightarrow\quad
Z(u,v) = \frac{1}{\alpha \cdot D_{\text{rel}}(u,v) + \beta}$$

$\alpha$ and $\beta$ are found the same way (least-squares over the tape-measured
anchors); only the target of the regression changes from $Z$ to $1/Z$.

Implementation: `test_pipeline.py --calib inverse`. The default, `--calib
affine`, keeps the linear-in-$Z$ form of Phase 1 above.

Measured on a real capture (bottle on a counter, anchors at 0.2 m and 1.0 m):

| model | $Z$ | $W$ | $H$ |
| --- | --- | --- | --- |
| `affine` ($Z = \alpha D + \beta$) | 82.7 cm | 10.7 cm | 31.4 cm |
| `inverse` ($1/Z = \alpha D + \beta$) | 53.6 cm | 6.9 cm | 20.3 cm |
| tape-measured ground truth | 60 cm | - | 23 cm |

Anchoring $Z$ directly at the true distance (`--distance 0.6`) reproduces the
ground truth ($W = 7.8$ cm, $H = 22.8$ cm), confirming that the residual error
sits in the depth-model alignment and not in the pinhole geometry of Phases 2-3.

### Phase 2: 3D Back-Projection for Physical Width and Height

Once the true metric depth $Z$ is known for a target pixel, its physical 3D
coordinates $(X, Y, Z)$ in meters are recovered using the camera's focal lengths
($f_x, f_y$) and principal point ($c_x, c_y$):

$$X = \frac{(u - c_x) \cdot Z}{f_x}$$

$$Y = \frac{(v - c_y) \cdot Z}{f_y}$$

### Phase 3: Sizing the Target Object

To find the physical width ($W$) and height ($H$) of an object bounded by
pixels $(u_1, v_1)$ and $(u_2, v_2)$ at a relatively uniform depth $Z$:

$$W = \frac{|u_2 - u_1| \cdot Z}{f_x}$$

$$H = \frac{|v_2 - v_1| \cdot Z}{f_y}$$

**Student Setup Shortcut:** If the exact camera matrix is unknown,
can use the smartphone's Exif metadata to extract the focal length in pixels, or
use a standard checkerboard calibration script in OpenCV
(`cv2.calibrateCamera`) which takes under an hour to set up.

## Part 2: Open-Source Depth Foundation Models

These state-of-the-art models are completely open-source, require no training,
and can be integrated into a Python script via Hugging Face or GitHub instantly:

### 1. Depth Anything V2 (Highly Recommended)

- **Developer:** TikTok / University of Hong Kong
- **Why it fits a 6-week timeline:** It is the current gold standard for
  zero-shot monocular depth estimation. It is incredibly fast, lightweight, and
  offers pristine boundary definitions even on complex geometries or low-contrast
  agricultural scenes.
- **Instant Access:** Available directly on Hugging Face Transformers
  (`pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small")`).

### 2. Marigold

- **Developer:** ETH Zürich
- **Why it fits a 6-week timeline:** Marigold leverages a latent diffusion
  architecture (built on top of Stable Diffusion). While slower to compute than
  Depth Anything, it produces remarkably high-fidelity, fine-grained structural
  depth details. It is perfect if target objects have micro-textures
  or intricate edges.
- **Instant Access:** Available via the Diffusers library (`MarigoldPipeline`).

### 3. MiDaS (v3.1)

- **Developer:** Intel ISL
- **Why it fits a 6-week timeline:** The classic baseline model for zero-shot
  depth. It runs efficiently on standard laptop CPUs without needing a dedicated
  GPU. It serves as an excellent low-overhead control benchmark to compare the
  newer models against.
- **Instant Access:** Can be loaded with two lines of code via PyTorch Hub
  (`torch.hub.load("intel-isl/MiDaS", "MiDaS_small")`).
