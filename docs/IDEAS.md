# Ideas — where the peak finder and the review can get better

My working list, ordered roughly by what I expect to pay off first. No results here: the
numbers behind each item are in the local outputs named next to it. Procedures are in
`RUNBOOK_vfe_peak_ml.md`, the committed plan in `../TODO.md`.

## 1. One P0 definition
My labels take the peak just before A1; the tester's finder now takes the tallest bump in the
window before A1. Most disagreements in the replay are P0. Options, to settle with Onur:
- relabel P0 with the finder's rule ("tallest") and retrain, so both sides answer the same question;
- or train the network to output both candidates (two P0 channels) and let the config pick.
Check: P0 disagreement rate in `outputs/real/finder_vs_ml/summary.json` before and after.

## 2. Operator choices as training labels
`review.csv` (GUI) and `judgments.csv` (`diagnostics/judge_disagreements.py`) hold a per-peak
verdict. Turn them into a label-override file read by `src/real_data.py`: where the operator
picked ML or clicked a position, that becomes the label; where they picked the finder, the
label is confirmed. Retrain and score on runs nobody reviewed, so the check stays held out.

## 3. Better review triage
- Flag a peak for review also when the two agree but the network's confidence is low.
- Calibrate the confidence (reliability curve on held-out runs) so a threshold means something.
- Sort the review queue by |ML − finder| or by confidence, so the clearest cases come last.

## 4. Sub-bin positions on the raw trace
The network sees 10 µs bins; the tolerance is 0.02 ms, two bins. Snap each network apex to
the highest raw sample within ±0.02 ms, as the finder reports the raw maximum. This also
gives the measurements in item 5 the same footing as the finder's.

## 5. Measurements from the network's peaks
Feed the network's apexes through the tester's own width/edge code and verdict table, and
show a Finder-vs-ML measurement table in replay (in progress). Closure test first: the
finder's own apexes through the same path must reproduce the finder's numbers exactly.

## 6. Live use on the bench
- Export the report model to ONNX and run it with onnxruntime inside the tester (no torch on the bench).
- Record the model file's hash and the export date in each `ml_peaks.csv.meta`.
- Review before Confirm, with Confirm locked until every flagged peak has a choice.
- Measure inference time on the bench PC before deciding whether it runs per shot or per point.

## 7. Model
- Seed ensemble (3 seeds) and use the spread as a second uncertainty.
- Per-voltage scores; the hardest region is above 10 V, where the C+D bump splits.
- A whole-trace "unusual shape" score from the network's features, to flag failure shots the finder still passes.

## 8. Tests and hygiene
- Headless Qt tests (`QT_QPA_PLATFORM=offscreen`) for the two-plot view and the review strip
  (`split_landmarks`, atomic write, resume rule).
- A small test for `diagnostics/make_replay_run.py` on one synthetic shot.
- One `smoke.sh` running the tester selftests (`--selftest`, `--selftest --replay`).

## 9. Sign-off
Write `docs/algorithm_vfe_peak_ml.md` and go through its four unsigned steps (dataset build
and P0 rule, run split, training, peak readout). Network values go into the DB only after
that and with Onur's agreement.
