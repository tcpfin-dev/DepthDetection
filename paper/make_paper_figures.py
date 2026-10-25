"""Generate the figures for the short paper from outputs_clean/.

Run with the project venv:
    ./venv/bin/python paper/make_paper_figures.py
"""

import csv
import os
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figs")
CLEAN = os.path.join(ROOT, "outputs_clean")
os.makedirs(OUT, exist_ok=True)

MODELS = ["small", "base", "large", "midas", "marigold"]
LABEL = {"small": "DA-V2-S", "base": "DA-V2-B", "large": "DA-V2-L",
         "midas": "MiDaS (DPT-Hybrid)", "marigold": "Marigold"}
COLOR = {"small": "#4c72b0", "base": "#55a868", "large": "#c44e52",
         "midas": "#8172b3", "marigold": "#b8860b"}
MARK = {"small": "o", "base": "s", "large": "D", "midas": "^", "marigold": "v"}
OBJECTS = ["bwl01", "c01", "a01", "b02", "b01"]
OBJ_LABEL = {"bwl01": "bowl", "c01": "cup", "a01": "apple",
             "b02": "bottle\n(short)", "b01": "bottle\n(tall, clear)"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})


def load_rows():
    with open(os.path.join(CLEAN, "eval_per_image.csv")) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- figure 1
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 1.45))
    ax.set_xlim(0, 20.0)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # (x, w, y, h, title, detail, sub)
    boxes = [
        (0.05, 3.05, 1.55, 1.30, "single RGB\nframe", "", "1920 x 1080"),
        (3.60, 3.00, 1.55, 1.30, "YOLOv8m", "frozen", "box, class, conf"),
        (3.60, 3.00, 0.15, 1.00, "depth model", "frozen", "median, inner 60 %"),
        (7.80, 3.45, 0.95, 1.90, "anchor fit", "$1/Z=\\alpha D+\\beta$",
         "cups at 0.40 / 1.00 m"),
        (11.80, 3.55, 0.95, 1.90, "back-projection", "$W=(u_2-u_1)Z/f_x$",
         "$H=(v_2-v_1)Z/f_y$"),
        (16.05, 3.80, 0.95, 1.90, "per-object output", "$Z$, $W$, $H$, area",
         "map + CSV / JSON"),
    ]
    for x, w, y, h, title, detail, sub in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=0.8, edgecolor="#333333", facecolor="#f4f4f4"))
        ax.text(x + w / 2, y + h * (0.74 if sub else 0.55), title,
                ha="center", va="center", fontsize=7.2, weight="bold")
        if detail:
            ax.text(x + w / 2, y + h * 0.46, detail, ha="center", va="center",
                    fontsize=7.0)
        ax.text(x + w / 2, y + h * 0.20, sub, ha="center", va="center",
                fontsize=6.0, color="#555555")

    def arrow(p, q, color="#333333", ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.8, color=color, linestyle=ls,
                                     shrinkA=0, shrinkB=0))

    arrow((3.12, 2.20), (3.58, 2.20))
    arrow((3.12, 2.20), (3.58, 0.65))
    arrow((6.62, 2.20), (7.78, 2.20))
    arrow((6.62, 0.65), (7.78, 1.35))
    arrow((11.27, 1.90), (11.78, 1.90))
    arrow((15.37, 1.90), (15.98, 1.90))

    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_qualitative():
    cases = [
        ("white/c1/c01_50cm", "cup, 0.50 m",
         "success: $\\hat{Z}$=0.49 m (-1.1 cm)"),
        ("nv/b1/b01_90cm", "tall clear bottle, 0.90 m",
         "bias: $\\hat{Z}$=1.07 m (+16.8 cm)"),
    ]
    ratio = 9 / 16
    col_w = 1.12
    fig_w = 3 * col_w + 2 * 0.05
    row_h = col_w * ratio
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, 2 * (row_h + 0.34) + 0.40))
    for c, h in enumerate(["RGB frame", "relative depth", "sized output"]):
        x = 0.03 + c * (col_w + 0.05) + col_w / 2
        fig.text(x / fig_w, 0.988, h, fontsize=6.8, ha="center", va="top")
    for r, (rel, name, cap) in enumerate(cases):
        base = os.path.join(CLEAN, "large", rel)
        imgs = [os.path.join(ROOT, "dataset", rel + ".png"),
                base + "_depth_color.png",
                base + "_pipeline.png"]
        for c, p in enumerate(imgs):
            ax = axes[r, c]
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(p)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.4)
                s.set_color("#666666")
        axes[r, 0].annotate(f"{name}  |  {cap}", xy=(0, 1.05),
                            xycoords="axes fraction", fontsize=6.0, ha="left",
                            va="bottom", color="#222222")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.845, bottom=0.02,
                        wspace=0.03, hspace=0.40)
    fig.savefig(os.path.join(OUT, "fig2_qualitative.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_results(rows):
    by_dist = defaultdict(lambda: defaultdict(list))
    by_obj = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d = float(r["distance_true_m"])
        by_dist[r["model"]][d].append(abs(float(r["z_err_cm"])))
        by_obj[r["model"]][r["object_id"]].append(abs(float(r["z_rel_err_pct"])))

    fig, ax1 = plt.subplots(figsize=(3.4, 1.85))

    for m in MODELS:
        ds = sorted(by_dist[m])
        ax1.plot(ds, [np.mean(by_dist[m][d]) for d in ds], marker=MARK[m],
                 ms=3.2, lw=1.2, color=COLOR[m], label=LABEL[m])
    ax1.set_xlabel("true object distance (m)")
    ax1.set_ylabel("MAE of $Z$ (cm)", labelpad=1)
    ax1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax1.set_ylim(0, 17)
    ax1.grid(alpha=0.25, lw=0.5)
    
    ax1.legend(frameon=False, loc="upper left", handlelength=1.6, ncol=2,
           columnspacing=0.8, fontsize=6.6)

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig3_distance.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_qualitative()
    fig_results(load_rows())
    for f in sorted(os.listdir(OUT)):
        print(os.path.join(OUT, f), os.path.getsize(os.path.join(OUT, f)))
