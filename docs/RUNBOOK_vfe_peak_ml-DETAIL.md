# VFE-board QC peak finding (ML) — Runbook (detail)

**Last updated:** 2026-11-13
**Card:** `RUNBOOK_vfe_peak_ml.md` — commands only.
**Where it runs:** this machine, `$ENV` micromamba env, NVIDIA RTX 4070 Laptop GPU (8 GB).
Every command runs from the repository root (`python -m src...` imports the repo's packages).
Working copy: `~/Documents/vfe_peak_ml/vfe-peak-ml` — a clone of the repo with the local,
git-ignored `data/`, `outputs/`, `writeup/plots/real/`, `writeup/outputs/` and real
diagnostics PNGs copied in.

## Overview

Replaces the hard-coded peak finder of the LVR board tester. Each 4-peak test shot is an
input-current trace (scope channel C4) with five landmarks: P0 the peak just before A1,
then P1–P4 = A1, B1, A2, B2. Steps 2–9 work on the tester's archive of `.trc` traces and
result YAMLs: the finder's own output on each shot is the training label for P1–P4, P0
comes from a rule; a 1D heat-map network is trained and scored split by run, compared with
the finder in a PDF, and written up in a LaTeX note. Steps 10–14 put the network next to the
tester's own finder: re-run today's finder on the held-out shots, run the tester GUI on this
laptop, and replay an archived run in the GUI with the finder's and the network's peaks side
by side and a per-peak review that records which one is right. Step 15 runs today's finder
on every shot of the dataset (the label source for the next model); step 16 compares the
tester's current limits with the distribution over the production cards.

Real data and everything derived from it stays local and git-ignored; the public repo
carries code only.

Steps 17–21 retrain on the finder's own P0 and on synthetic merged peaks, compare that
model with today's finder and show every held-out shot of one test type in the tester GUI.

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

### 10. Result YAMLs into the repo's data folder
**Known-good command:**
```bash
mkdir -p data/archive_yaml && tar -xf ~/Downloads/results.tar -C data/archive_yaml --no-same-owner --no-same-permissions --wildcards 'results/*/*.yaml'
```
**Expect:** one folder per run, `data/archive_yaml/results/<run>/<run>.yaml`. `data/` is
git-ignored and survives reboots, unlike `/tmp`. Steps 8, 11 and 13 take `data/archive_yaml`
as their YAML root. The archive is untrusted: YAMLs are read with `safe_load` only and
nothing from it is run.

### 11. Today's finder vs the network on the held-out shots
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.finder_vs_ml data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester
```
`--limit N` runs the first N shots only (smoke test).
**Expect:** missing Vin (C1) traces extracted into `data/raw/` in one tar pass, progress every
100 shots (about 10 min for all held-out and failure shots), the summary printed, and
`outputs/real/finder_vs_ml/shots.csv` (per shot and landmark: network time and confidence,
today's finder time, a disagree flag at 0.02 ms, the archived finder time) plus
`summary.json`, each with a `.meta` sidecar. `finder_exceptions` is 0, and on clean shots the
self-check shows today's finder reproducing the archived A1–B2 times.

### 12. Tester GUI on this laptop
**Known-good command:**
```bash
~/Documents/vfe_peak_ml/run_lvr_tester.sh --selftest; tail -3 ~/Documents/vfe_peak_ml/lvr_tester_gui.log
```
Desktop icon `~/.local/share/applications/lvr-tester.desktop` (copy on `~/Desktop`) runs the
same launcher without arguments.
**Expect:** a `selftest OK — IDLE, START blocked by: …` line. START stays blocked without
instruments, so nothing can be run, confirmed or uploaded from the laptop.

### 13. Build a replay folder from an archived run
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.make_replay_run data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --run <run>
```
`--out` defaults to `~/Documents/vfe_peak_ml/replay/<run>`.
**Expect:** one line per shot (Vin, shot, network vs finder per landmark) and the output
folder. Under `<out>/campaign_replay/`: `shots/shot_NNNNN_V<vin>V.npz` (the layout the
tester's runner writes), `campaign_peaks.csv` and `campaign_shot_metrics.csv` from today's
finder, `ml_peaks.csv` from the network, each with a `.meta` sidecar. An existing
`review.csv` is kept and a WARNING printed.

### 14. Replay in the GUI, side by side, with the peak review
**Known-good command:**
```bash
~/Documents/vfe_peak_ml/run_lvr_tester.sh --selftest --replay ~/Documents/vfe_peak_ml/replay/<run>; tail -2 ~/Documents/vfe_peak_ml/lvr_tester_gui.log; ~/Documents/vfe_peak_ml/run_lvr_tester.sh --replay ~/Documents/vfe_peak_ml/replay/<run>
```
Icon: `~/.local/share/applications/lvr-tester-replay.desktop` (copy on `~/Desktop`).
**Expect:** `replay OK — <n> traces, hint: …` in the log, then the window. Each "Trace <v> V"
entry shows the finder (left) and the network (right) on a shared time axis, zoomed to the
landmarks; **Zoom out** toggles the full trace. A network landmark more than 0.02 ms from the
finder's is red. The review strip under the plots lists each flagged landmark of the shot
with Finder / ML / Neither and Prev / Next; **F / M / N** work after a click into the plot;
Neither waits for a click on the true position (Esc cancels). Each choice is written to
`campaign_replay/review.csv` at once.

### 15. Today's finder on every shot
**Known-good command:**
```bash
micromamba run -n $ENV python -W ignore -m diagnostics.finder_all_shots data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --workers 12 > outputs/real/finder_all/full_run.log 2>&1
```
`--limit N` runs the first N shots only; `-W ignore` keeps scipy's `PeakPropertyWarning`
lines, one per worker and shot, out of the log. Create `outputs/real/finder_all/` first when
redirecting into it.
**Expect:** `<N> shots, <n> Vin traces extracted` (missing C1 traces pulled into `data/raw/`
in one tar pass), a progress line every 250 shots (about 30 s per 250 with 12 workers), then
`shots … peaks found per shot {…} P0 kind {…}` and the path of
`outputs/real/finder_all/shots.csv` with its `.meta`. One row per shot in npz order (`idx` =
row of `data/real_shots.npz`): peak count, Vin crossing, P0 time / kind (`turnover` or
`shoulder`) / amplitude, and for each of the four peaks apex time, amplitude, pulse start,
pulse end and pulse level; blank = no answer. Check: with `--limit 16`, the P0–P4 times of a
held-out `idx` equal those in step 11's `shots.csv`.

### 16. Limits against the production-card distribution
**Known-good command:**
```bash
mkdir -p ~/Documents/vfe_peak_ml/limits_study/data/records && tar -xf ~/Downloads/results.tar -C ~/Documents/vfe_peak_ml/limits_study/data/records --no-same-owner --no-same-permissions --wildcards 'results/*_P1_B[12]_*/*.yaml'
cd ~/Documents/vfe_peak_ml/limits_study && micromamba run -n $ENV python diagnostics/2026-11-13_limits_distribution.py data/records ../lvr-tester/configuration/test_types/incoming_A.yaml diagnostics/2026-11-13_limits_distribution
```
Runs from `~/Documents/vfe_peak_ml/limits_study/`, not the repo root: the script, the
extracted records and the outputs live there, outside both repositories.
**Expect:** `boards kept: … runs skipped: … incomplete points: …`, one line per skipped run
(an earlier run of a retested board) and per incomplete voltage point, then the two output
paths. The CSV has one row per quantity and voltage (17 quantities × 4 voltages): number of
boards, min / median / max, mean, std, robust sigma (1.4826 × MAD), the current window, boards
outside it, and the margin from each window edge to the nearest board. The PDF has one page
per quantity, four voltage panels, current limits dashed. Both carry a `.meta` sidecar and
name the unsigned steps.

### 17. P0 label from the finder
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.relabel_p0_finder
```
**Expect:** one line `data/real_shots_p0finder.npz shots=… labelled=… newly_labelled_p0=…
moved_over_0.02ms=… moved_over_0.1ms=…`, then the path; a `.meta` beside it. Needs step 15's
`outputs/real/finder_all/shots.csv`. The script stops if the output exists, if a card peak in
the dataset differs from today's finder by more than 0.001 ms, or if a new P0 is outside the
trace or not before A1. Only `P[:, 0]` changes, on the labelled shots; the old value is kept
as `P0_rule`, the finder's `turnover` / `shoulder` as `P0_kind`. `run`, `n_found` and `good`
are untouched, so the run split is the same. Finder-failure shots stay unlabelled.

### 18. Synthetic merged peaks
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.make_merges
```
**Expect:** one line `data/real_shots_merge.npz merge_depth_A=0.3 max_depth_A=0.7 seed=0
real=… synthetic=… skipped=0 merged=… from_train=… from_val=… merged_by_pair={…}`, then the
path; a `.meta` beside it. About half the synthetic shots come out merged. Real rows come
first and are identical to step 17's file; one synthetic shot is appended per labelled
training or validation shot, in the same run; hard-test runs get none. New arrays:
`synthetic`, `src_idx`, `merge_pair` (1 = A1+B1, 2 = B1+A2, 3 = A2+B2), `merge_depth_A`;
`finder_t_ms` is NaN on synthetic rows. A merged shot has exactly one NaN landmark.

### 19. Train on the relabelled and on the merged dataset
**Known-good command:**
```bash
micromamba run -n $ENV python -m src.train_real --arch unet --base 32 --levels 4 --data data/real_shots_p0finder.npz --out outputs/real/p0finder/u32l4 > outputs/real/p0finder_u32l4.log 2>&1
micromamba run -n $ENV python -m src.train_real --arch unet --base 32 --levels 4 --data data/real_shots_merge.npz --out outputs/real/merge/u32l4 > outputs/real/merge_u32l4.log 2>&1
```
**Expect:** as step 5; about 8 min for the first and 15 min for the second on the laptop GPU
(the second has twice the training shots). The first changes only the P0 label, the second
adds the merges, so a change on real shots can be assigned to one of them. With a dataset
that has a `synthetic` array, `metrics.json` gains `n_train_synthetic` and a `val_synthetic`
block (position error on the landmarks that are there, share of present landmarks with
confidence ≥ 0.5, share of absent ones below it), and `predictions.npz` gains `synval_pos`,
`synval_conf`, `synval_idx`. `val` and `val_idx` cover real validation shots only, so they
stay comparable with earlier models.

### 20. Today's finder vs a chosen model
**Known-good command:**
```bash
mkdir -p outputs/real/finder_vs_ml_merge && micromamba run -n $ENV python -W ignore -m diagnostics.finder_vs_ml data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --data data/real_shots_p0finder.npz --pred outputs/real/merge/u32l4/predictions.npz --out outputs/real/finder_vs_ml_merge --min-conf 0.5 > outputs/real/finder_vs_ml_merge/full_run.log 2>&1
```
**Expect:** as step 11 (about 10 min), written to the `--out` folder; step 11's own folder is
untouched. `--min-conf` blanks a model landmark whose confidence is below it (the model says
it is not there); a blank on one side only counts as a disagreement. `--limit 8` is a quick
check. Without the options the script behaves as in step 11.

### 21. All held-out shots of one test type in the GUI
**Known-good command:**
```bash
micromamba run -n $ENV python -W ignore -m diagnostics.make_replay_run data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --table outputs/real/finder_vs_ml_merge/shots.csv --test-type incoming_A --out ~/Documents/vfe_peak_ml/replay/all_heldout_incoming_A_merge
~/Documents/vfe_peak_ml/run_lvr_tester.sh --selftest --replay ~/Documents/vfe_peak_ml/replay/all_heldout_incoming_A_merge; tail -2 ~/Documents/vfe_peak_ml/lvr_tester_gui.log; ~/Documents/vfe_peak_ml/run_lvr_tester.sh --replay ~/Documents/vfe_peak_ml/replay/all_heldout_incoming_A_merge
```
**Expect:** first `<n> runs of incoming_A; left out: [...]` (runs of another test type), one
line per shot, then the folder path; about 10 min and about 3 MB per shot. Shots are ordered
by voltage, run, shot; `campaign_replay/replay_index.csv` maps each shot number to run,
voltage, shot and set. The self-test prints `replay OK — 4 traces …` and
`replay table OK — 6 columns, … rows`; the last command opens the window as in step 14.
`--table` picks the comparison folder of step 20; without `--test-type` the script builds one
run as in step 13.

## Conventions

- **Exactly five landmarks.** The model always returns five positions; a doubtful peak is
  signalled by low heat-map confidence, never by a shorter output. For the merge model a
  landmark with confidence below 0.5 counts as not there (`--min-conf 0.5` in step 20).
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
  Where network and rule disagree on P0, the rule is right: on the two held-out 12 V shots
  more than 1 ms apart, the rule's P0 is a small bump at the start of the rise into A1 and
  the network's P0 is a miss, flagged by its low confidence (decided 2026-11-11).
- **Real landmark map:** P0 = peak before A1, P1 = A1, P2 = B1, P3 = A2, P4 = B2 (4-peak test).
- **No simulated data, with one exception.** Toy generator, toy models, deck and toy figures
  were removed (commit 6ef0dc9); only real traces are used. Exception (decided 2026-11-13):
  the model that handles merged peaks is trained on synthetic merges, because the archive
  holds no merged shots (today's finder finds four separate peaks on every shot).
- **Merged-peak rule (decided 2026-11-13).** Valley depth = lower apex minus the minimum
  between the two apexes. Under 0.3 A the two pulses are one peak, labelled at the taller
  apex; the lower landmark is NaN and trains an all-zero heat-map. 0.3 A or more: both keep
  their labels. 0.3 A sits between the two figures in the finder's settings note (notch under
  0.1 A, real valley about 0.8 A); it is my choice, not checked on a real merged shot.
- **Synthetic merges are cut, not added (decided 2026-11-13).** `src/make_merges.py` deletes
  the stretch between two neighbouring card pulses at a chosen current level, so every sample
  is real; the join is smoothed over a few bins. The depth is measured again on the finished
  trace and the label follows that measurement. P0 keeps the finder's label on every shot.
  Because the taller pulse survives, almost only "A1 absent" and "A2 absent" are trained.
- **Datasets sit side by side.** `data/real_shots.npz` (P0 by my rule), `…_p0finder.npz`
  (P0 from the finder), `…_merge.npz` (the latter plus synthetic shots). Models go to
  `outputs/real/p0finder/<tag>/` and `outputs/real/merge/<tag>/`; nothing is overwritten.
- **One replay folder holds one test type.** The GUI takes one test type and one set of peak
  names per folder, so pooled folders are built per test type (step 21). Their verdict table
  pools many boards and describes no single board; the per-shot view and the review do.
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
  `diagnostics/make_replay_run.py` is tracked and has no default run: the run or the test
  type is always given on the command line.
- **Write-up numbers are macros.** Every real-data number in
  `writeup/src/vfe_peak_ml_writeup.tex` is a `\newcommand` generated into
  `writeup/plots/real/numbers.tex`, and the real tables are generated `.tex` rows read with
  `\csname @@input\endcsname` (plain `\input` inside a tabular breaks `\bottomrule`). The
  `.tex` source therefore holds no real results; `writeup/plots/real/` and `writeup/outputs/`
  are git-ignored.
- **Tester changes live on a local branch.** `ml-peak-review` in
  `~/Documents/vfe_peak_ml/lvr-tester`, uncommitted, not pushed anywhere. Onur's repository
  stays private; it goes to my private GitHub first and to Onur's only with his agreement.
  The launcher `run_lvr_tester.sh` sits outside his repo.
- **`runner_python` falls back off Windows.** `MainWindow._runner_python` uses the configured
  interpreter on Windows or when it exists, else `sys.executable`; the DB-upload and sync
  calls go through it too.
- **Disagree = more than 0.02 ms** between the network and the finder for a landmark
  (`ML_TOL_MS` in `plot_panel.py`, the same flag as `finder_vs_ml`).
- **Review rules.** The operator chooses only where the two disagree, one choice per
  landmark (finder / ml / neither + clicked time); agreed landmarks are saved as `agree` with
  the finder's value. `review.csv` columns: shot_index, vin_setting, landmark, finder_ms,
  ml_ms, choice, chosen_ms, operator, timestamp; written atomically on the first choice;
  resuming needs the saved finder and network times to match within 1e-3 ms. A reviewed
  `review.csv` is never overwritten by a rebuild.
- **Retraining is manual on the laptop GPU; the bench gets an ONNX export.** Network values
  go into the DB only after the unsigned steps are signed off and Onur agrees.
- **Two P0 definitions exist.** My labels: the peak just before A1 (`p0_before_a1`). The
  finder: `peak0.selection` in `configuration/base/measurement_base.yaml`, "nearest" or
  "tallest" in the window before A1; the P0 limits in `incoming_A.yaml` were derived from the
  finder's P0. Above 10 V the C+D bump splits in two and the more prominent hump changes from
  shot to shot, so the two definitions can differ there. Decided 2026-11-13: the model takes
  its P0 label from the finder's rule (step 15's output, step 17); the earlier dataset and
  models are kept beside it.
- **Tester safeguards are read-only.** Voltage and amplitude values and every safeguard in
  the tester stay untouched: the `hard:` limits and `voltages:` of
  `configuration/test_types/*.yaml`, the supply settings under `configuration/base/`, the
  forced fields in `lvr_tester/core/yaml_build.py`, `lvr_tester/runner/psu_panic.py`, and the
  finder thresholds in `measurement_base.yaml`.
- **Limits stay as they are.** Step 16 only describes the distribution and feeds suggestions;
  the windows in the tester's test types are not changed from here. Its population: the
  production A cards, one record per board (the latest confirmed `incoming_A` run), the
  per-voltage board mean stored in the record, voltage points without all four peaks in
  every shot left out.
- **Algorithm steps are unsigned.** No `docs/algorithm_*.md` exists yet; any deliverable
  names the unsigned steps (dataset build and P0 rule, run split, training, peak readout,
  synthetic merge generation).

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
- **Symptom:** after a reboot, `<members.txt>` and `<yaml_dir>` from step 2 are gone, so
  step 4 cannot be rerun.
  **Cause:** they were put under `/tmp`, which is cleared at shutdown.
  **Fix:** use `data/archive_yaml` from step 10 as `<yaml_dir>`, and keep `<members.txt>`
  under `data/` too.
- **Symptom:** `Command 'python' not found`, or `No module named diagnostics` / `src`.
  **Cause:** the shell has no bare `python`, or the command ran outside the repository root.
  **Fix:** `cd` to the repo root and use `micromamba run -n $ENV python -m ...`.
- **Symptom:** "Check instruments" in the tester crashes with `FileNotFoundError` on a
  `C:\Users\...\python.exe` path.
  **Cause:** `runner_python` in the tester config is the bench's Windows interpreter.
  **Fix:** the `_runner_python` fallback (Conventions); on Linux it uses the running python.
- **Symptom:** `pkill -f "python main.py --replay"` kills the shell that runs it (exit 144).
  **Cause:** the pattern matches the shell's own command line.
  **Fix:** `pkill -f "[m]ain.py --replay"`.
- **Symptom:** the replay window shows one plot instead of two, with no network markers.
  **Cause:** the replay folder was rebuilt (step 13) while the window was open, and it
  redrew while `ml_peaks.csv` was missing.
  **Fix:** click another Trace entry, or restart the window.
- **Symptom:** the tester's `pytest` reports 14 failures, and a file literally named
  `Z:\nonexistent\dir\r.sqlite` appears in its root.
  **Cause:** failures already present upstream; the same 14 fail on `main`.
  **Fix:** none needed. Compare against that baseline (14 failed / 809 passed / 19 skipped)
  and delete the stray file.
- **Symptom:** `knowledge/`, `docs/IDEAS.md` or `diagnostics/make_replay_run.py` are missing
  from the working copy, and a `.meta` sidecar names a commit `git cat-file -t` cannot find.
  **Cause:** the working copy was behind `origin/main`, and untracked scripts and local data
  existed only in another folder.
  **Fix:** `git pull --ff-only`, then `rsync -a -n -i --exclude .git --exclude __pycache__
  <other folder>/ ./` to list the differences, and the same without `-n` to copy them.

- **Symptom:** `collect() gave <N> shots, npz has <M>: record order cannot be trusted`
  from `diagnostics.finder_vs_ml`.
  **Cause:** `--data` pointed at the merged dataset; the script checks the archive's shot
  count against the dataset's rows.
  **Fix:** pass `--data data/real_shots_p0finder.npz`. The real rows and their indices are the
  same in both files, so the merge model's saved predictions still line up.
- **Symptom:** `AssertionError: no shot_*_V<n>V.npz under replay/...` from the launcher's
  `--selftest --replay`.
  **Cause:** a relative replay path; the launcher changes into the tester folder first.
  **Fix:** give the replay folder as an absolute path.

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
- 2026-11-12 — Working copy `~/Documents/vfe_peak_ml/vfe-peak-ml` recorded; step 9 expect
  updated to the 11-page real-only write-up; P0 decision (rule over network on disagreement)
  added to Conventions; gotcha for step-2 files lost from `/tmp` at shutdown.
- 2026-11-12 — Steps 10–14 added: result YAMLs into `data/archive_yaml`, today's finder vs the
  network (`diagnostics/finder_vs_ml.py`), tester GUI on the laptop, replay folder
  (`diagnostics/make_replay_run.py`), replay GUI with side-by-side plots and the per-peak
  review strip. Step 8 takes the YAML root and gains three pages. Conventions: tester branch,
  `runner_python` fallback, 0.02 ms disagree, review rules, retraining and ONNX, two P0
  definitions. Four gotchas added; the `/tmp` gotcha now points at step 10.
- 2026-11-13 — Steps 15–16 added: today's finder on every shot
  (`diagnostics/finder_all_shots.py`) and the limits-versus-distribution study on the
  production cards. Conventions: synthetic merges allowed for the merged-peak model, P0 label
  from the finder's rule for the next model, tester safeguards read-only, limits unchanged.
  One gotcha added (working copy behind, files only in another folder).
- 2026-11-13 — Steps 17–21 added: P0 label from the finder, synthetic merged peaks, training
  on both datasets, finder vs a chosen model, all held-out shots of one test type in the GUI.
  Conventions for the merged-peak rule, the cut-valley generator, dataset and model folders,
  pooled replay folders; two gotchas (dataset row count in the comparison, relative replay path).
- 2026-11-13 — `diagnostics/make_replay_run.py` committed without a default run (`--run` or
  `--test-type` must be given); the convention that kept it untracked is replaced.
