# VFE-board QC peak finding (ML) — Runbook (detail), steps 10–16: tester, finder comparison, replay, limits

**Main detail:** `RUNBOOK_vfe_peak_ml-DETAIL.md` — overview, conventions, gotchas, changelog.
**Card:** `RUNBOOK_vfe_peak_ml.md` — commands only.

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
with Finder / ML / Neither and Prev / Next; **F / M / N** work after a click into the plot.
Candidate peaks are orange circles: Finder / ML store the nearest one within 0.1 ms (refused
with a hint if there is none), Neither waits for a click and stores the nearest one in view
(Esc cancels). Each choice is written to `campaign_replay/review.csv` at once. A window that
was open before a code change still runs the old code; close and reopen it.

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

