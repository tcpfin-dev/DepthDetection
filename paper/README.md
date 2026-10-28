# Short paper (IEEE conference format)

`paper.tex` is the short paper manuscript for the project described in
`../AGENTS.md`. `paper.pdf` is the built result (6 pages: 5 pages of content
plus a references page).

## Build

```sh
cd paper
pdflatex paper
pdflatex paper          # second pass for cross-references
```

`IEEEtran.cls` is vendored in this folder, so no TeX Live publisher packages are
required. If LaTeX is missing on a fresh box:

```sh
pacman -S texlive-latex texlive-latexrecommended texlive-fontsrecommended
```

## Figures

`figs/` is generated, not hand-made. Regenerate from the evaluation CSVs:

```sh
cd .. && ./venv/bin/python paper/make_paper_figures.py
```

| file | content | source |
|---|---|---|
| `fig1_pipeline.png` | method schematic (drawn with matplotlib) | — |
| `fig2_qualitative.png` | mug success / tall-bottle failure, RGB + depth + sized output | `outputs_clean/large/...` and `dataset/...` |
| `fig3_distance.png` | MAE vs true distance, all five models | `outputs_clean/eval_per_image.csv` |

## Numbers

Every table and every figure value comes from `outputs/`. `../REPRODUCE.md` has
the full, verified recipe; the short version is that a single `--depth-model all`
run is **wrong for this paper**, because Marigold needs its native affine fit
while the regression heads need the disparity fit, and each batch rewrites the
eval CSVs:

```sh
for m in small base large midas; do
  ./venv/bin/python pipeline.py --dataset dataset --depth-model $m \
      --yolo-size medium --calib inverse --out /tmp/repro/$m
done
./venv/bin/python pipeline.py --dataset dataset --depth-model marigold \
    --yolo-size medium --calib affine --seed 42 --out /tmp/repro/marigold
./venv/bin/python merge_eval_trees.py --out outputs --expect 475 /tmp/repro/*
```

`../verify_paper_numbers.py` re-derives every quoted value from
`outputs/eval_per_image.csv` (Table II, the 61 %/17.7 % error shares, the
box-vs-depth split, the mean Z_hat/Z ratios, the back-solved focal length, the
voided count). Run it after any rerun: numbers computed across models go stale
whenever one model's rows change.

## Before submitting

- Author block: Huang JunLin (first) and Y. F. Tang, both School of Engineering,
  Nanyang Polytechnic, Singapore, with their e-mail addresses. Keep this block
  and any submission metadata in sync.
- Add the venue's copyright/notice block if the call for papers requires one.
- The PDF is 6 pages: 5 of content plus a references page (`\clearpage` before
  `thebibliography` keeps the list off the content pages). A page-limited venue
  needs the figure/caption compression pass that was undone for readability.
