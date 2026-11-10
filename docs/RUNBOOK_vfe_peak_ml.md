# VFE-board QC peak finding (ML) — Runbook

**Last updated:** 2026-11-09
**Where it runs:** this machine, `$ENV` micromamba env, NVIDIA RTX 4070 Laptop GPU (8 GB)

## Overview

Replaces a hard-coded peak finder for VFE-board QC voltage traces. Each good trace holds
exactly five peaks, P0–P4: a train of many small spikes whose last member is P0, then P1–P4
rising monotonically in amplitude, each decaying only partway so the inter-peak floor keeps
climbing, with small irregular structure sitting on those floors that must not be reported.
A toy generator produces labelled training traces; a 1D U-Net emits one per-sample heatmap
channel per peak; a results PDF carries the method, examples and metrics. As of this writing
**no real board data has been used** — every number below is measured on toys.

## Steps

### 1. Environment

**Known-good command:**
```bash
micromamba run -n $ENV pip install --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
```
**Expect:** `torch 2.15.0+cu126`, and this must print `True`:
```bash
micromamba run -n $ENV python -c "import torch;print(torch.cuda.is_available())"
```
numpy 2.5.2, matplotlib 3.11.1, scipy 1.18.1, scikit-learn 1.9.1 are already present in that env.

### 2. Generator self-check

**Known-good command:**
```bash
micromamba run -n $ENV python src/toy_waveforms.py
```
**Expect:** one line `ok  peaks: [...] apex: [...]` with five strictly increasing positions and
apex values increasing from P0 to P4. The script asserts ordering, P4 > P1, and that the floor
between P1 and P2 sits above the pre-P0 level.

### 3. Train the CNN (stage 1)

**Known-good command:**
```bash
micromamba run -n $ENV python -m src.train
```
**Expect:** `data (8000, 1, 1024) (1000, 1, 1024) ... on cuda`, 25 epoch lines, ~70–85 s total on
the 4070. Writes `outputs/metrics.json`, `outputs/peak_unet.pt`, `outputs/val_eval.npz`.
Current result on 1000 held-out toys: median |Δt| 0.29–0.41 samples per peak, all five peaks
within 2 samples on 88.9 % of traces, 0.8 % of traces with a gross miss (>15 samples), nearly
all of them P0 confused with the pre-spike train.

### 4. Hybrid stage 2 (optional, currently not the default)

**Known-good command:**
```bash
micromamba run -n $ENV python -m src.hybrid
```
**Expect:** a stage-2 table of roughly 170 candidates per trace (`PROMINENCE = 0.010`,
`PRIOR_MIN = 0.05`, `MATCH_TOL = 3.0`), then two comparison lines.
Writes `outputs/metrics_hybrid.json`, `outputs/hybrid_eval.npz`. Current result: hybrid 0.832–0.858
all-five-within-2 versus 0.889 for CNN-only, i.e. **worse**. Keep CNN-only as the default path.

### 5. Decoder ablation

**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.hybrid_ablation
```
**Expect:** four rows (cnn_only, dp_monotone, dp_plain, greedy), the count of traces where the
CNN answer is inside the candidate set (currently 1000/1000) and where the CNN chain violates
monotone apex (6/1000). Writes `outputs/ablation.json`.

### 6. Build the PDF

**Known-good command:**
```bash
micromamba run -n $ENV python -m src.report
```
**Expect:** `wrote outputs/vfe_peak_ml_report.pdf`, six pages: method/results text, toy gallery,
validation predictions with heatmaps, per-peak error histograms, training curves, stage-2
comparison. The stage-2 page is skipped with a printed notice if step 4 or 5 has not been run.

## Conventions

- **Exactly five peaks.** The model always returns five positions; a missing peak is signalled only
  by low heatmap confidence, never by a shorter output.
- **Output representation:** five per-sample heatmap channels, position = argmax, confidence =
  max sigmoid. Chosen over direct regression of five scalars so QC gets a per-peak confidence.
- **Targets:** Gaussian bumps, σ = 4 samples, trained with `BCEWithLogitsLoss`.
- **Input scaling:** per-trace median subtraction then division by max |value|, so absolute volts
  never matter. Any inference path must reuse `src.train.normalise` verbatim.
- **Peak shape:** rise width and decay tau are drawn independently, then tilted by one per-trace
  `shape_bias` in [0.5, 2.0], so slow-rise/fast-fall and fast-rise/slow-fall both occur.
- **Truth definition:** the generator's pulse centre t₀, *not* the sample maximum. These differ by
  ~0.5 samples systematically, because the floor step and the previous peak's decay displace the
  observed apex. **Open question for real data:** if the existing hard-coded finder reports the
  sample maximum, it is not measuring the same quantity and the two cannot be compared directly.
- **Default path is CNN-only.** `src/hybrid.py` stays in the tree as a switch to re-test on real
  traces, not as the production path.
- **Done criteria used so far:** median |Δt| ≤ 2 samples per peak (met), and no decoy structure
  ranked above a true peak (not met — 0.8 % gross-miss rate remains).

## Gotchas

- **Symptom:** `pip install` reported success, but `torch.cuda.is_available()` stayed `False` and
  `torch.__version__` stayed `2.15.0+cpu`.
  **Cause:** the install command redirected its log to a path built from `$PWD/../../...` that did
  not exist, so the shell failed before pip ran, and a trailing `|| true` swallowed the error.
  **Fix:** never mask an install's exit status; log into the session scratchpad with an absolute
  path, and verify afterwards with the `torch.cuda.is_available()` one-liner in step 1.

- **Symptom:** `ValueError: high - low < 0` from `rng.uniform` in `make_waveform`.
  **Cause:** widening `jitter_frac` to 0.50 let P0 land within a few samples of the trace start, so
  the pre-spike window `(2.0, pos[0] - 6.0)` inverted.
  **Fix:** clamp peak positions to `lo_edge = max(40, 0.03 * n_samples)` in `draw_peak_positions`,
  and guard the window with `pre_hi = max(4.0, pos[0] - 6.0)`.

- **Symptom:** `TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'`.
  **Cause:** matplotlib 3.11 renamed the argument.
  **Fix:** use `tick_labels=` instead of `labels=`.

- **Symptom:** `RuntimeWarning: overflow encountered in exp` from `spiky_pulse`.
  **Cause:** the exponential-decay branch is evaluated over the whole time axis before `np.where`
  discards it, so samples far left of the apex produce a huge positive exponent.
  **Fix:** clip both exponents, `np.clip(..., -700, 0)` and `np.clip(..., 0, 700)`.

- **Symptom:** metrics quoted after a rerun were identical to the previous run, wall-clock time
  included.
  **Cause:** the rerun had crashed partway; `outputs/metrics.json` was still the earlier file, and
  the chained `&&` report step had also left a partially written PDF.
  **Fix:** read the run's log and confirm it reached its final print before reading any metrics
  file; treat identical wall-clock times across runs as the tell that nothing was rewritten.

- **Symptom:** `UserWarning: CUDA initialization: ... Error 101: invalid device ordinal`, and the
  run silently continues on CPU.
  **Cause:** GPU state left by a previous process in the same session.
  **Fix:** harmless for inference-only scripts, but check the `device` field in the written metrics
  before quoting any timing number as a GPU timing.

- **Symptom:** the hybrid stage lost badly to CNN-only (0.509 all-five-within-2 on first attempt).
  **Cause:** two compounding issues — the scan threshold `prominence=0.004` admitted ~248 noise
  maxima per trace, and scan maxima are systematically displaced from the pulse centre, so
  scan-only candidate recall (true peak within 3 samples) caps at 0.954 even there — see
  `scan_only_sweep` in `outputs/ablation.json`. On a separate 300-trace draw the same sweep showed
  P0 recall collapsing from 0.967 at prominence 0.004 to 0.23 at 0.08 (that per-peak breakdown is
  not persisted; rerun the sweep if you need it).
  **Fix:** make the candidate set the union of scan maxima and the CNN's own heatmap maxima, which
  lifts recall to 0.997 and puts the CNN answer in the set 1000/1000 times. Note this only removed
  the ceiling; the stage-2 re-ranker still underperforms the heatmap it consumes, so the hybrid
  remains off by default.

## Changelog

- 2026-11-09 — Runbook created. Records the toy generator, the CNN heatmap model, the hybrid stage
  and its ablation, the six run commands with expected output, the conventions settled this session
  (five peaks, heatmap output, σ = 4, normalisation, independent rise/decay, truth = pulse centre),
  and seven gotchas hit while building it.

## Apex reporting and the snap guard

`src/refine.py` is the inference entry point: `predict(model, X_raw)` returns, per
peak, the position, the network confidence, the apex sample, the apex height in
the board's own units, and whether the guard moved the position. It reapplies
`src.train.normalise` itself, so callers pass raw traces.

**Known-good command (tuning sweeps):**
```bash
micromamba run -n $ENV python -m diagnostics.refine_sweep
micromamba run -n $ENV python -m diagnostics.refine_gated
```
**Expect:** `outputs/refine_sweep.json` and `outputs/refine_gated.json`; the
baseline line must match `outputs/metrics.json`, or the model file is stale.

Defaults: window 3, gain 0.05, smooth 1, snap only below confidence 0.50.

## Gotchas (continued)

- **Symptom:** snapping every peak to the nearest local maximum drops all-five-within-2
  from 0.876 to 0.276.
  **Cause:** truth is the pulse centre; the noisy sample maximum is a worse estimate
  of it than the heatmap argmax.
  **Fix:** keep the gain and confidence guards. Under *sample-maximum* truth the
  ranking reverses (0.272 network-only vs 0.668 always-snap), so settle the
  definition before tuning anything here.
- **Symptom:** a generator change silently invalidates `outputs/ablation.json`,
  `metrics_hybrid.json`, `refine_*.json` and every slide figure.
  **Cause:** they are all keyed to the trained model and the generator config.
  **Fix:** after any `ToyConfig` edit rerun, in order: `src.train`, `src.hybrid`,
  `diagnostics.hybrid_ablation`, `diagnostics.refine_sweep`,
  `diagnostics.refine_gated`, `talk/make_plots.py`, then the deck.

## Changelog
- 2026-11-10 — P0 now sits 10--60 samples ahead of P1 (`p0_gap`, `peak_window`
  raised to 0.14); retrained. All-five-within-2 0.889 -> 0.876, gross miss
  0.008 -> 0.025, both from P0/P1 confusion. Added `src/refine.py` (apex
  reporting + guarded snap) and the two tuning sweeps. Added `talk/` with the
  lab deck.
- 2026-11-10 — Labels moved from the pulse centre to the visible apex of the noiseless
  waveform (`label_at_apex`, `apex_labels`); draws with an invisible landmark, or a
  P0-P1 apex gap outside 10--60, are redrawn (42% of draws). Before: on 20% of P1-P4
  and 34% of P0 labels the label sat >2 samples off the apex, up to 40 late, and on 37%
  of draws P0 had no apex at all. Retrained: P0 gross 2.3% -> 0.8%; all-five-within-2
  0.876 -> 0.768, within-4 0.890. Loss weighting of the P0 channel was tried first and
  did nothing (0.924 -> 0.923); weights left neutral in `src/train.py`.

## Gotchas (labels)

- **Symptom:** a training label sits on the rising edge, below the visible apex.
  **Cause:** each main peak is a pulse plus a floor step centred on the same point;
  the sum keeps rising after it.
  **Fix:** label at the apex of the noiseless waveform (`apex_labels`).
- **Symptom:** median |dt| reads exactly 0.00 for P1-P4.
  **Cause:** apex labels and argmax predictions are both whole samples.
  **Fix:** report within-k fractions, not the median.
