# VFE-board QC peak finding (ML) — Runbook (detail), steps 17–24: relabelling, merged peaks, review

**Main detail:** `RUNBOOK_vfe_peak_ml-DETAIL.md` — overview, conventions, gotchas, changelog.
**Card:** `RUNBOOK_vfe_peak_ml.md` — commands only.

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
by voltage, run, shot; `campaign_replay/replay_index.csv` maps each shot number to the
dataset row (`idx`), run, voltage, shot and set. The self-test prints `replay OK — 4 traces …` and
`replay table OK — 6 columns, … rows`; the last command opens the window as in step 14.
`--table` picks the comparison folder of step 20; without `--test-type` the script builds one
run as in step 13.

### 22. Today's finder vs the cross-fit network on every shot
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.crossfit_vs_finder
```
Needs step 15's `outputs/real/finder_all/shots.csv`, step 17's dataset and
`outputs/real/crossfit/predictions_all.npz` (one network answer per shot from a model that
never saw its run, written by `src/assemble_crossfit.py`; the cross-fit training and assembly
are not in this runbook yet).
**Expect:** a few seconds, nothing re-run. The summary is printed (shots, disputed shots,
one-sided shots, disputed landmarks per name and per voltage; for all shots and for the
train / val / hard runs), then three paths under `outputs/real/crossfit_vs_finder/`, each with
a `.meta`: `shots.csv` (one row per shot: the columns step 23's builder reads, plus `fold`,
`P*_diff_ms`, `one_sided`, `max_diff_ms`), `review_queue.csv` (the disputed shots only,
one-sided first, then `max_diff_ms` descending), `summary.json`. The script stops if an output
exists, if the finder table and the dataset do not list the same shots, or if the finder table
differs from the dataset labels by more than 0.001 ms. Check: rows of `shots.csv` = shots of
the dataset; per-set counts = `params` in `predictions_all.npz.meta`; rows of
`review_queue.csv` = `shots_any_disagree`; the three sets' disputed shots add up to the total.

### 23. Disputed shots in the GUI, largest difference first
**Known-good command:**
```bash
micromamba run -n $ENV python -W ignore -m diagnostics.make_replay_run data/archive_yaml ~/Documents/vfe_peak_ml/lvr-tester --table outputs/real/crossfit_vs_finder/review_queue.csv --all-runs --keep-order --out ~/Documents/vfe_peak_ml/replay/crossfit_disputed > outputs/real/crossfit_vs_finder/replay_build.log 2>&1
~/Documents/vfe_peak_ml/run_lvr_tester.sh --selftest --replay ~/Documents/vfe_peak_ml/replay/crossfit_disputed; tail -2 ~/Documents/vfe_peak_ml/lvr_tester_gui.log; ~/Documents/vfe_peak_ml/run_lvr_tester.sh --replay ~/Documents/vfe_peak_ml/replay/crossfit_disputed
```
`--all-runs` takes every row of `--table`, whatever its run or test type; `--keep-order` keeps
the table's row order instead of sorting by voltage, run, shot. Smoke test: the header and
first three rows of the queue as `--table`, a scratch folder as `--out`.
**Expect:** the build log starts `all <n> rows of <table>, <m> runs`, has one line per shot
and ends with the folder path; the number of `shots/*.npz` equals the rows of the queue, and
`replay_index.csv` starts at the queue's first `idx`. `replay.yaml` takes the test type of the
first row. The self-test prints `replay OK — 4 traces …` and `replay table OK — 6 columns, …
rows`. In the window, select the Trace row of shot 1's voltage; Next-flagged then follows the
shot number, which is the queue's order, across voltages. Review as in step 14.

### 24. Candidate-peak figure
**Known-good command:**
```bash
micromamba run -n $ENV python -m diagnostics.candidate_maxima ~/Documents/vfe_peak_ml/replay/crossfit_disputed diagnostics/<date>_real_candidate_maxima.png
```
Needs step 22's queue and step 23's folder.
**Expect:** six lines (the first queue shot with a disputed P0 and the first with a disputed
P1, each at w = 0.02, 0.05, 0.10 ms) giving the candidates in view, then the PNG path. The
count falls as w grows. Each panel shows the raw trace around the disputed peak, the finder's
and the network's positions, and the candidates as circles.

