# VFE-board QC peak finding (ML) — Runbook (detail), steps 1–9: archive, dataset, training, report

**Main detail:** `RUNBOOK_vfe_peak_ml-DETAIL.md` — overview, conventions, gotchas, changelog.
**Card:** `RUNBOOK_vfe_peak_ml.md` — commands only.

### 1. Environment
**Known-good command:**
```bash
micromamba run -n $ENV pip install --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
micromamba install -n $ENV -c conda-forge pyyaml -y
```
**Expect:** this prints `True`:
```bash
micromamba run -n $ENV python -c "import torch,yaml;print(torch.cuda.is_available())"
```
numpy, matplotlib, scipy and scikit-learn are already in the env.

### 2. List and unpack the archive's metadata
**Known-good command:**
```bash
tar -tf ~/Downloads/results.tar > <members.txt> && mkdir <yaml_dir> && tar -xf ~/Downloads/results.tar -C <yaml_dir> --no-same-owner --no-same-permissions --wildcards '*.yaml'
```
**Expect:** one line per archive member in `<members.txt>`; the per-run result YAMLs under
`<yaml_dir>/results/<run>/`. `<yaml_dir>` is a new, empty folder outside the repo.

### 3. First look at real traces
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.real_first_look <yaml_dir> ~/Downloads/results.tar <extract_dir> diagnostics/<date>_real_first_look.png <members.txt>
```
**Expect:** a PNG of shots where the finder found all four peaks and shots where it found
three, the finder's marks overlaid (apex markers, start–end shaded).

### 4. Build the real dataset
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.real_data <yaml_dir> ~/Downloads/results.tar <members.txt> data
```
**Expect:** progress lines, then `data/real_shots.npz  shots=… good=… runs=… P0 labelled=…`
(P0 labelled a few shots below good). Traces are
extracted once into `data/raw/` (skipped on rerun). The npz holds X (N, 1, 5120) in amps,
P (N, 5) landmark bins, a 0.01 ms bin width, and the finder's raw output for every shot
(`finder_t_ms`, `finder_lab`), failures included.

### 5. Train on real traces
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.train_real --arch unet --out outputs/real/unet
```
Options: `--arch unet|transformer`, `--base B --levels L` (U-Net width/depth),
`--d D --tdepth T` (transformer width/layers), `--seed S`, `--epochs`, `--sigma`.
**Expect:** a `train … | val … | hard …` split line, 40 epoch lines, then metrics; writes
`metrics.json` (argmax and sub-bin `val_subbin` scores, plus the clean shots of the
failure runs), `model.pt`, `predictions.npz` into the `--out` folder.

### 6. Compare model sizes
**Known-good command:**
```bash
for cfg in "u16 16 3" "u32 32 3" "u64 64 3" "u32l4 32 4" "u32l5 32 5"; do set -- $cfg; micromamba run -n $ENV python -m src.train_real --arch unet --base $2 --levels $3 --out outputs/real/sweep/$1; done && for cfg in "t256x4 4" "t256x8 8"; do set -- $cfg; micromamba run -n $ENV python -m src.train_real --arch transformer --d 256 --tdepth $2 --out outputs/real/sweep/$1; done
for s in 1 2; do for cfg in "<tag> <base> <levels>"; do set -- $cfg; micromamba run -n $ENV python -m src.train_real --arch unet --base $2 --levels $3 --seed $s --out outputs/real/sweep_seed/$1_s$s; done; done
micromamba run -n $ENV python -m diagnostics.real_sweep_summary
```
**Expect:** the seven seed-0 runs take about 55 min in total on the RTX 4070 (2 min for
base 16 up to 12 min for base 64); the second line re-trains the top configs with new
seeds. The summary prints one row per folder under `outputs/real/sweep*/`: parameters,
training time, seed, and argmax / sub-bin agreement with the labels.

### 7. Failure shots
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.real_hard_cases outputs/real/unet outputs/real/transformer diagnostics/<date>_real_hard_cases.png
```
**Expect:** per failure run, the models' A1 time and its shot-to-shot spread, and a PNG of
the failure shots with each model's marks.

### 8. ML vs finder PDF
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.report_real outputs/real/<date>_ml_vs_finder.pdf outputs/real/sweep/u32l4 data/archive_yaml
```
The second argument is the U-Net folder for pages 1–6 (default `outputs/real/unet`); the
third is the result-YAML folder from step 10, used for the trace zooms.
**Expect:** a 7-page PDF (summary, agreement, clean shots, failure shots, repeatability,
confidence, model size), plus three pages when `outputs/real/finder_vs_ml/shots.csv` from
step 11 exists (today's finder vs the network: per-landmark table and Vin bars, P0 zooms,
A1 zooms), so 10 pages. The summary and sweep rows are printed. Every page names the
unsigned algorithm steps; page 1 names the U-Net's width, depth, size and seed.

### 9. Full write-up (LaTeX note)
**Known-good command:**
```bash
micromamba run -n $ENV python -m writeup.make_figures && cd writeup/src && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs vfe_peak_ml_writeup.tex && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs vfe_peak_ml_writeup.tex
```
Needs steps 5 (`outputs/real/transformer`) and 6 (`outputs/real/sweep/u32l4` and the seed
re-runs).
**Expect:** 8 PNG paths and
`perpeak_table.tex`, `sweep_table.tex`, `numbers.tex` (`writeup/plots/real/`), no assertion
error; `writeup/outputs/vfe_peak_ml_writeup.pdf`, 11 pages, real data only;
`grep -E "LaTeX Warning|Overfull"` on the log shows only one 2.6 pt overfull line (Method,
read-out paragraph). Every page footer names the unsigned steps.

