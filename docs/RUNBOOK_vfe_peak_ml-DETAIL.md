# VFE-board QC peak finding (ML) — Runbook (detail)

**Last updated:** 2026-11-11
**Card:** `RUNBOOK_vfe_peak_ml.md` — commands only.
**Where it runs:** this machine, `$ENV` micromamba env, NVIDIA RTX 4070 Laptop GPU (8 GB).
Every command runs from the repository root (`python -m src...` imports the repo's packages).

## Overview

Replaces the hard-coded peak finder of the LVR board tester. Each 4-peak test shot is an
input-current trace (scope channel C4) with five landmarks: P0 the peak just before A1,
then P1–P4 = A1, B1, A2, B2. Steps 2–9 work on the tester's archive of `.trc` traces and
result YAMLs: the finder's own output on each shot is the training label for P1–P4, P0
comes from a rule; a 1D heat-map network is trained and scored split by run, compared with
the finder in a PDF, and written up in a LaTeX note.

Real data and everything derived from it stays local and git-ignored; the public repo
carries code only.

## Steps

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
micromamba run -n $ENV python -m src.report_real outputs/real/<date>_ml_vs_finder.pdf outputs/real/sweep/u32l4
```
The second argument is the U-Net folder for pages 1–6 (default `outputs/real/unet`).
**Expect:** a 7-page PDF (summary, agreement, clean shots, failure shots, repeatability,
confidence, model size) and the summary and sweep rows printed. Every page names the
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
error; `writeup/outputs/vfe_peak_ml_writeup.pdf`, 16 pages; `grep -E "LaTeX Warning|Overfull"`
on the log shows only one 2.6 pt overfull line in the abstract. Every page footer names the
unsigned steps.

## Conventions

- **Exactly five landmarks.** The model always returns five positions; a doubtful peak is
  signalled by low heat-map confidence, never by a shorter output.
- **Output:** five per-sample heat-map channels, position = argmax, confidence = max sigmoid.
  Chosen over regressing five numbers so QC gets a per-peak confidence.
- **Input scaling:** per-trace median subtraction, then division by max |value|. Every
  inference path reuses `src.model.normalise`.
- **Real labels = the finder's output.** The finder's `time_ms` is the highest raw sample
  inside the peak and `amplitude_A` the current there (checked against the traces);
  `start_ms` is its onset.
- **P0 = the peak just before A1** (not reported by the finder), by rule in
  `src.real_data.p0_before_a1`: from the last bin before A1 below 0.4 × A1's height, walk back
  while the current falls (to the dip), then while it rises; the first local maximum is P0.
  Steps of up to 5 mA against the direction are tolerated. No peak, or a walk more than 3 ms
  before A1 → P0 is NaN and the shot is neither trained on nor scored. The rule needs A1, so
  finder-failure shots have no P0 label. Checked with `diagnostics/real_p0_rule.py`.
- **Real landmark map:** P0 = peak before A1, P1 = A1, P2 = B1, P3 = A2, P4 = B2 (4-peak test).
- **No simulated data.** Toy generator, toy models, deck and toy figures were removed
  (commit 6ef0dc9); only real traces are used.
- **Real binning:** traces stored at 200 ns are averaged in blocks of 50 into 10 µs bins and
  padded to 5120.
- **Split by run, never by shot** — shots inside a run are near-copies. Runs containing a
  finder failure are held out entirely. Training applies random shifts of up to 3 ms so the
  network learns shape, not position.
- **Compare models across seeds.** The same seed on the GPU does not reproduce bit for bit
  — identical configs give measurably different scores, so a single-run difference may be noise.
- **Sweep folder naming:** seed-0 runs go to `outputs/real/sweep/<tag>/`, extra seeds to
  `outputs/real/sweep_seed/<tag>_s<seed>/`. `src/report_real.py` groups runs by the tag
  before `_s`, so the tag itself must not contain `_s`.
- **Report model = U-Net base 32, 4 levels, seed 0** (`outputs/real/sweep/u32l4`), chosen
  on three seeds: it varied least across seeds of the two configs re-trained. Seed 0 is used
  because it was trained before the seed comparison, not picked from it.
- **`.trc` format facts** (LeCroy WAVEDESC): block lengths are seven int32 at offsets 36–60,
  `WAVE_ARRAY_1` is the one at 60; `SPARSING_FACTOR` (int32 at 136) multiplies
  `HORIZ_INTERVAL`, because the header keeps the full-rate interval for sparse records.
  Not every shot's trace is in the archive — always filter by the `tar -tf` listing.
- **Archive handling:** the downloaded tar is untrusted — extract into a new empty folder
  with `--no-same-owner --no-same-permissions`, read YAML with `safe_load` only, run nothing
  from it.
- **Real data stays local.** `data/`, `outputs/real/` and `diagnostics/*_real_*` are
  git-ignored; the public repo gets code only, and docs carry no real results or run IDs.
- **Write-up numbers are macros.** Every real-data number in
  `writeup/src/vfe_peak_ml_writeup.tex` is a `\newcommand` generated into
  `writeup/plots/real/numbers.tex`, and the real tables are generated `.tex` rows read with
  `\csname @@input\endcsname` (plain `\input` inside a tabular breaks `\bottomrule`). The
  `.tex` source therefore holds no real results; `writeup/plots/real/` and `writeup/outputs/`
  are git-ignored.
- **Algorithm steps are unsigned.** No `docs/algorithm_*.md` exists yet; any deliverable
  names the unsigned steps (dataset build and P0 rule, run split, training, peak readout).

## Gotchas

- **Symptom:** metrics quoted after a rerun are identical to the previous run, wall-clock
  time included.
  **Cause:** the rerun crashed partway and left the earlier metrics file in place.
  **Fix:** confirm the run's log reached its final print before reading any metrics file;
  identical wall-clock times across runs are the tell.
- **Symptom:** `UserWarning: CUDA initialization: ... Error 101: invalid device ordinal`, and
  the run continues on CPU.
  **Cause:** GPU state left by a previous process in the same session.
  **Fix:** harmless for inference, but check the `device` field in the written metrics before
  quoting a timing as a GPU timing.
- **Symptom:** a P0 rule with fixed windows (dip = lowest bin in A1 − 1.5 ms, P0 = highest
  bin in dip − 0.4 ms) puts P0 on the down-slope on hundreds of shots, mostly at 9 V.
  **Cause:** the P0 → A1 gap varies with input voltage (median 1.5–1.9 ms).
  **Fix:** the walking rule in `p0_before_a1`; no fixed windows.
- **Symptom:** a training run is cut off (machine switched off) and leaves an empty `--out`
  folder.
  **Cause:** files are written only after the last epoch.
  **Fix:** rerun that config; finished folders are complete and need no rerun.
- **Symptom:** `Command 'python' not found`, or `No module named diagnostics` / `src`.
  **Cause:** the shell has no bare `python`, or the command ran outside the repository root.
  **Fix:** `cd` to the repo root and use `micromamba run -n $ENV python -m ...`.

## Changelog

- 2026-11-09 — Runbook created: toy generator, CNN heat-map model, hybrid and ablation, six
  run commands, conventions (five peaks, heat-map output, normalisation, independent
  rise/decay, truth = pulse centre), seven gotchas.
- 2026-11-10 — P0 placed 10–60 samples ahead of P1; retrained. Added `src/refine.py` (apex
  reporting + guarded snap), the two tuning sweeps, and the lab deck in `talk/`.
- 2026-11-10 — Toy labels moved from the pulse centre to the visible apex; invisible-landmark
  draws redrawn. P0 gross miss 2.3 % → 0.8 %; all-five-within-2 0.876 → 0.768. P0 loss
  weighting tried first and did nothing; weights left neutral.
- 2026-11-10 — ±2-sample study on apex labels: target width and training-length sweep
  (`diagnostics/2026-11-10_sigma_sweep/`) and sub-sample decoding gave nothing better than
  0.768; oracle readers score 0.569–0.639. On toys the apex is defined to about 3 samples.
- 2026-11-11 — Real track added: `src/trc.py`, `src/real_data.py`, `src/train_real.py`
  (U-Net, configurable-depth `PeakUNetN`, transformer, sub-bin readout, seeds),
  `src/report_real.py`, `diagnostics/real_first_look.py`, `real_hard_cases.py`,
  `real_sweep_summary.py`; PyYAML added to the env. Settled: finder = raw sample maximum,
  landmark map P0 = inrush and P1–P4 = A1..B2, run-level split, real data local only.
- 2026-11-11 — Split into card + detail; consolidated the duplicated Gotchas and Changelog
  sections; refreshed toy numbers in steps 3–5 to the apex-label run. Pruned gotchas the
  current code can no longer produce: install log masked by `|| true`, pre-spike window
  `high - low < 0`, matplotlib `boxplot(labels=)`, `exp` overflow in `spiky_pulse`, hybrid
  scan-recall ceiling under centre labels, labels on the rising edge. Moved `.trc` header
  facts (WAVE_ARRAY_1 offset, SPARSING_FACTOR) into Conventions.
- 2026-11-11 — Step 16 now carries the sweep training loop and the extra-seed re-runs;
  sweep folder naming added to Conventions.
- 2026-11-11 — Step 18: PDF gains a model-size page (7 pages) and takes the U-Net folder as
  a second argument; report model switched from base 16 to base 32 / 4 levels.
- 2026-11-11 — Step 19 added: full LaTeX write-up (`writeup/make_figures.py`,
  `writeup/src/vfe_peak_ml_writeup.tex`), real numbers as generated macros. Step 9's
  hybrid panel y-range fixed (60–85 %; the old 80–92 % left the bars off-axis).
- 2026-11-11 — Simulated data removed: toy steps 2–11, toy conventions and three toy gotchas
  dropped, steps renumbered 1–9. P0 redefined from the inrush spike to the peak just before
  A1 (walking rule); `normalise` moved to `src/model.py`; all models retrained on the new
  labels.
