# VFE-board QC peak finding (ML) — Runbook (detail)

**Last updated:** 2026-11-15
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

Steps 22–24 prepare my own review as the label source: today's finder against an
out-of-sample network answer on every shot, the disputed shots in the GUI ordered by the size
of the difference, and the figure behind the candidate-peak rule the review uses.

## Steps

The commands and the expected output of each step are in three files:

- Steps 1–9, archive, dataset, training, report: `RUNBOOK_vfe_peak_ml-DETAIL-steps-01-09.md`
- Steps 10–16, tester, finder comparison, replay, limits: `RUNBOOK_vfe_peak_ml-DETAIL-steps-10-16.md`
- Steps 17–24, relabelling, merged peaks, review: `RUNBOOK_vfe_peak_ml-DETAIL-steps-17-24.md`

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
  Exception: the review-queue folder (step 23) pools every test type and takes the first
  row's; only its positions are reviewed, its table is ignored.
- **My picks are the reference (decided 2026-11-15).** The labels are the finder's output, so
  a network trained on them cannot be shown to be better than the finder, and the current
  trace holds no physical truth for a peak position (the Enable signals are not recorded in
  production). On the peaks where finder and network differ, the label becomes my choice.
  The review covers every shot, with the network side from the cross-fit answers (step 22):
  picks on training-run shots become training labels; picks on validation and failure runs
  only correct the reference the models are scored against. Shots where the two agree get
  their own review later.
- **Review queue.** Disputed = more than 0.02 ms apart, or a landmark the finder did not give;
  shots ordered one-sided first, then by their largest disputed difference, descending. No
  confidence cut on the network in step 22, unlike step 20.
- **Candidate peaks (decided 2026-11-15).** A candidate is a raw sample that is the highest
  within ±0.05 ms of itself; nothing is smoothed. Every stored answer is a candidate's time:
  Finder / ML move to the nearest one within 0.1 ms, Neither is a click that takes the nearest
  one in view, and the click's own x is never stored. Plain raw local maxima do not work as
  candidates: they sit under a microsecond apart on slopes and in valleys alike, so the
  nearest one is where the click was. 0.05 ms was chosen by eye from step 24's figure. The
  rule lives in `candidate_peaks` and `nearest_candidate` in the tester's
  `lvr_tester/ui/review_strip.py`.
- **A disputed peak left unanswered drops its shot (decided 2026-11-15).** Neither followed by
  Esc writes no row, so an unanswered peak cannot be told from one not reached. Once I declare
  the review done, an unanswered disputed peak means no marked peak was right, and the whole
  shot leaves training. `src/relabel_review.py` applies this; it is written and not yet run.
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
  The launcher `run_lvr_tester.sh` sits outside his repo. Git holds no copy of the
  uncommitted files, so an edit to them is preceded by a dated `.orig` copy in the tester's
  backup folder.
- **`runner_python` falls back off Windows.** `MainWindow._runner_python` uses the configured
  interpreter on Windows or when it exists, else `sys.executable`; the DB-upload and sync
  calls go through it too.
- **Disagree = more than 0.02 ms** between the network and the finder for a landmark
  (`ML_TOL_MS` in `plot_panel.py`, the same flag as `finder_vs_ml`).
- **Review rules.** The operator chooses only where the two disagree, one choice per
  landmark (finder / ml / neither, stored as a candidate peak's time); agreed landmarks are saved as `agree` with
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
- **Symptom:** an `awk -F,` lookup of the last column of one of the CSVs by its header name
  prints the whole line instead of the value.
  **Cause:** Python's `csv` writer ends lines with `\r\n`, so the last header carries a `\r`
  and the name does not match.
  **Fix:** pipe through `tr -d '\r'` first.

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
- 2026-11-15 — Steps 22–24 added: finder vs the cross-fit network on every shot with the review
  queue (`diagnostics/crossfit_vs_finder.py`), the disputed shots in the GUI in queue order
  (`make_replay_run.py --all-runs --keep-order`, `idx` in `replay_index.csv`), the
  candidate-peak figure. Steps 14 and 21 updated for the candidate rule and `idx`.
  Conventions: my picks as the reference, review queue, candidate peaks, unanswered peak drops
  the shot, pooled queue folder, `.orig` before editing the tester's uncommitted files. One
  gotcha added (`\r` in the last CSV column). The detail passed 40 KB, so the Steps section
  moved verbatim into three per-stage files (`-DETAIL-steps-01-09`, `-10-16`, `-17-24`).
