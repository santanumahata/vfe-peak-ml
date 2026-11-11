# VFE-board QC peak finding (ML) — Runbook (detail)

**Last updated:** 2026-11-11
**Card:** `RUNBOOK_vfe_peak_ml.md` — commands only.
**Where it runs:** this machine, `$ENV` micromamba env, NVIDIA RTX 4070 Laptop GPU (8 GB).
Every command runs from the repository root (`python -m src...` imports the repo's packages).

## Overview

Replaces the hard-coded peak finder of the LVR board tester. Each 4-peak test shot is an
input-current trace (scope channel C4) with five landmarks: P0 the inrush spike, then
P1–P4 = A1, B1, A2, B2, each sitting on a floor that never returns to its starting level,
with ripple and small structure in between that must not be reported. Two tracks:

- **Toy track** (steps 2–11): a generator written from a verbal description, a 1D U-Net
  heat-map model trained on it, a hybrid scan-and-rank variant, apex/precision studies,
  a results PDF and a lab deck.
- **Real track** (steps 12–18): the tester's archive of `.trc` traces and result YAMLs. The
  finder's own output on each shot is the training label; models are trained and scored
  split by run, and compared with the finder in a PDF.

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

### 2. Generator self-check (toys)
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.toy_waveforms
```
**Expect:** one line `ok  peaks: [...] apex: [...]`, five strictly increasing positions and
apex heights rising from P0 to P4. Labels are the visible apex of the noiseless waveform;
draws with an invisible landmark, or a P0–P1 apex gap outside 10–60 samples, are redrawn
(about 42 % of raw draws).

### 3. Train the toy CNN
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.train
```
Sweep overrides: `--sigma S --epochs E --n-train N --out DIR`.
**Expect:** `data (8000, 1, 1024) (1000, 1, 1024) ... on cuda`, 25 epoch lines, about
2 min on the 4070. Writes `outputs/metrics.json`, `outputs/peak_unet.pt`,
`outputs/val_eval.npz`. Current toy result: all five within 2 samples 0.768, within 4
samples 0.890, gross miss (>15 samples) 0.020.

### 4. Hybrid stage 2 (toys, not the default)
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.hybrid
```
**Expect:** a stage-2 candidate table, then a `cnn_only` and a `hybrid` line; writes
`outputs/metrics_hybrid.json`, `outputs/hybrid_eval.npz`. The hybrid scores below
CNN-only (0.722 vs 0.768 all-five-within-2).

### 5. Decoder ablation (toys)
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.hybrid_ablation
```
**Expect:** four rows (cnn_only, dp_monotone, dp_plain, greedy), the count of traces where
the CNN answer is in the candidate set (1000/1000) and where the CNN chain breaks monotone
apex (1/1000). Writes `outputs/ablation.json`.

### 6. Apex snap tuning (toys)
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.refine_sweep && micromamba run -n $ENV python -m diagnostics.refine_gated
```
**Expect:** `outputs/refine_sweep.json`, `outputs/refine_gated.json`; the baseline line must
match `outputs/metrics.json`, or the model file is stale. `src/refine.py::predict` is the
toy inference entry point (position, confidence, apex sample, apex height in board units,
snap flag); defaults window 3, gain 0.05, smooth 1, snap only below confidence 0.50.

### 7. Apex precision checks (toys)
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.apex_ambiguity && micromamba run -n $ENV python -m diagnostics.decode_compare outputs
```
**Expect:** oracle readers (told where each peak is) at 0.569 raw / 0.639 smoothed
all-five-within-2, 74 % of traces with an apex flat to one noise sigma 3+ samples away;
writes `outputs/apex_ambiguity.json`. Sub-sample decoding matches argmax within 0.3 %.

### 8. Toy results PDF
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.report
```
**Expect:** `wrote outputs/vfe_peak_ml_report.pdf`. The stage-2 page is skipped with a
printed notice if step 4 or 5 has not been run.

### 9. Slide figures
**Known-good command:**
```bash
micromamba run -n $ENV python talk/make_plots.py
```
**Expect:** one absolute path per figure in `talk/plots/` (anatomy, heatmap_demo, perpeak,
methods, gallery, p0_errors, snap, precision).

### 10. Lab deck
**Known-good command:**
```bash
cd talk/src && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs vfe_peak_ml.tex && pdflatex -interaction=nonstopmode -output-directory=../outputs vfe_peak_ml.tex
```
**Expect:** `talk/outputs/vfe_peak_ml.pdf`, 17 pages;
`grep -c Overfull ../outputs/vfe_peak_ml.log` is 0.

### 11. Rebuild order after a generator change
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.train && micromamba run -n $ENV python -m src.hybrid && micromamba run -n $ENV python -m diagnostics.hybrid_ablation && micromamba run -n $ENV python -m diagnostics.refine_sweep && micromamba run -n $ENV python -m diagnostics.refine_gated && micromamba run -n $ENV python -m diagnostics.apex_ambiguity && micromamba run -n $ENV python talk/make_plots.py
```
**Expect:** every stored toy number and figure regenerated from the new generator; then
steps 8 and 10.

### 12. List and unpack the archive's metadata (real)
**Known-good command:**
```bash
tar -tf ~/Downloads/results.tar > <members.txt> && mkdir <yaml_dir> && tar -xf ~/Downloads/results.tar -C <yaml_dir> --no-same-owner --no-same-permissions --wildcards '*.yaml'
```
**Expect:** one line per archive member in `<members.txt>`; the per-run result YAMLs under
`<yaml_dir>/results/<run>/`. `<yaml_dir>` is a new, empty folder outside the repo.

### 13. First look at real traces
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.real_first_look <yaml_dir> ~/Downloads/results.tar <extract_dir> diagnostics/<date>_real_first_look.png <members.txt>
```
**Expect:** a PNG of shots where the finder found all four peaks and shots where it found
three, the finder's marks overlaid (apex markers, start–end shaded).

### 14. Build the real dataset
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.real_data <yaml_dir> ~/Downloads/results.tar <members.txt> data
```
**Expect:** progress lines, then `data/real_shots.npz  shots=… good=… runs=…`. Traces are
extracted once into `data/raw/` (skipped on rerun). The npz holds X (N, 1, 5120) in amps,
P (N, 5) landmark bins, a 0.01 ms bin width, and the finder's raw output for every shot
(`finder_t_ms`, `finder_lab`), failures included.

### 15. Train on real traces
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.train_real --arch unet --out outputs/real/unet
```
Options: `--arch unet|transformer`, `--base B --levels L` (U-Net width/depth),
`--d D --tdepth T` (transformer width/layers), `--seed S`, `--epochs`, `--sigma`.
**Expect:** a `train … | val … | hard …` split line, 40 epoch lines, then metrics; writes
`metrics.json` (argmax and sub-bin `val_subbin` scores, plus the clean shots of the
failure runs), `model.pt`, `predictions.npz` into the `--out` folder.

### 16. Compare model sizes
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.real_sweep_summary
```
**Expect:** one row per folder under `outputs/real/sweep*/`: parameters, training time,
seed, and argmax / sub-bin agreement with the finder.

### 17. Failure shots
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.real_hard_cases outputs/real/unet outputs/real/transformer diagnostics/<date>_real_hard_cases.png
```
**Expect:** per failure run, the models' A1 time and its shot-to-shot spread, and a PNG of
the failure shots with each model's marks.

### 18. ML vs finder PDF
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.report_real outputs/real/<date>_ml_vs_finder.pdf
```
**Expect:** a 6-page PDF (summary, agreement, clean shots, failure shots, repeatability,
confidence) and the summary rows printed. Every page names the unsigned algorithm steps.

## Conventions

- **Exactly five landmarks.** The model always returns five positions; a doubtful peak is
  signalled by low heat-map confidence, never by a shorter output.
- **Output:** five per-sample heat-map channels, position = argmax, confidence = max sigmoid.
  Chosen over regressing five numbers so QC gets a per-peak confidence.
- **Input scaling:** per-trace median subtraction, then division by max |value|. Every
  inference path reuses `src.train.normalise`.
- **Toy labels = visible apex** of the noiseless waveform (`label_at_apex`). Each toy peak is
  a pulse plus a floor step centred on the same point, so the summed curve tops out after
  the pulse centre; the old centre labels sat on the rising edge.
- **Real labels = the finder's output.** The finder's `time_ms` is the highest raw sample
  inside the peak and `amplitude_A` the current there (checked against the traces);
  `start_ms` is its onset. P0 (inrush) is not reported by the finder: it is the highest bin
  between −5 and +3 ms.
- **Real landmark map:** P0 = inrush, P1 = A1, P2 = B1, P3 = A2, P4 = B2 (4-peak test).
- **Real binning:** traces stored at 200 ns are averaged in blocks of 50 into 10 µs bins and
  padded to 5120.
- **Split by run, never by shot** — shots inside a run are near-copies. Runs containing a
  finder failure are held out entirely. Training applies random shifts of up to 3 ms so the
  network learns shape, not position.
- **Compare models across seeds.** The same seed on the GPU does not reproduce bit for bit
  — identical configs give measurably different scores, so a single-run difference may be noise.
- **`.trc` format facts** (LeCroy WAVEDESC): block lengths are seven int32 at offsets 36–60,
  `WAVE_ARRAY_1` is the one at 60; `SPARSING_FACTOR` (int32 at 136) multiplies
  `HORIZ_INTERVAL`, because the header keeps the full-rate interval for sparse records.
  Not every shot's trace is in the archive — always filter by the `tar -tf` listing.
- **Archive handling:** the downloaded tar is untrusted — extract into a new empty folder
  with `--no-same-owner --no-same-permissions`, read YAML with `safe_load` only, run nothing
  from it.
- **Real data stays local.** `data/`, `outputs/real/` and `diagnostics/*_real_*` are
  git-ignored; the public repo gets code only, and docs carry no real results or run IDs.
- **Default toy path is CNN-only.** `src/hybrid.py` stays as a switch, not the production path.
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
- **Symptom:** a generator change leaves `outputs/ablation.json`, `metrics_hybrid.json`,
  `refine_*.json`, `apex_ambiguity.json` and the slide figures silently stale.
  **Cause:** they are all keyed to the trained toy model and the generator config.
  **Fix:** run step 11, then steps 8 and 10.
- **Symptom:** snapping every peak onto the nearest maximum of the noisy trace drops toy
  all-five-within-2 from 0.768 to 0.570.
  **Cause:** labels are the clean apex; the noisy-trace maximum is a worse estimate of it
  than the heat-map argmax.
  **Fix:** keep the gain and confidence guards in `src/refine.py`.
- **Symptom:** median |Δt| reads exactly 0.00 for toy P1–P4.
  **Cause:** apex labels and argmax predictions are both whole samples.
  **Fix:** report within-k fractions, not the median.
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
