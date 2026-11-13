# VFE-board QC peak finding (ML) — TODO

**Last updated:** 2026-11-13

## Now
- Merge model (P0 from the finder + synthetic merged peaks, `outputs/real/merge/u32l4`) is open in the replay GUI on all held-out `incoming_A` shots: offscreen self-test passes, shots not yet looked through. Open points: P0 is lost on more real validation shots than with the old model (cause not known); merged-peak handling is checked on synthetic merges only, no real merged shot exists; almost only "A1 absent" and "A2 absent" were trained.
- ML measurements from the network's peaks: closure test is exact on 220 shots (`outputs/real/finder_vs_ml/ml_measure_closure.json`); `lvr_tester/ml/` read through, open points not fixed (peak count never tested, P0 cap and gap guard not applied to a network P0); still uncommitted.

## Next
1. Go through the pooled replay folder in the GUI (`~/Documents/vfe_peak_ml/replay/all_heldout_incoming_A_merge`, runbook step 21) and review the flagged landmarks.
2. Decide on `diagnostics/make_replay_run.py`: remove its real run ID default and commit it, or keep it untracked (runbook steps 13 and 21 need it).
3. Find why the relabelled and merge models lose P0 on more real validation shots than the old model.
4. Decide which findings of the 2026-11-13 code review to fix (failed `review.csv` save hidden, hand-set positions dropped on a model re-run, unsigned review rows, `--replay` taking the bench lock, among others).
5. Run `python -m diagnostics.judge_disagreements data/archive_yaml` and go through the disagreements.
6. Settle one P0 definition with Onur ("nearest" vs "tallest"; `docs/IDEAS.md` item 1).
7. Commit `ml-peak-review` and push it to the private GitHub mirror.
8. Live use: ONNX export, network run per shot in the tester, review before Confirm with Confirm locked.
9. Draft `docs/algorithm_vfe_peak_ml.md` (chain flowchart, step list, blocks for the four real-chain steps) and go through sign-off one step at a time.

## Blocked
- Any further deliverable from the real-data chain — **waiting on:** sign-off of its four steps (dataset build and P0 rule, run split, training, peak readout), or a per-deliverable override as given for the 2026-11-11 PDF and the full write-up.
- Network values in the tester's DB — **waiting on:** the sign-off above and Onur's agreement.
- Push of lvr-tester to the private GitHub mirror — **waiting on:** a working token.
- Peak 0 limits against the board distribution — **waiting on:** `scripts/peak0_reanalysis.csv` from the bench.

## Done
- 2026-11-13 — Runbook steps 17–21 (P0 from the finder, synthetic merges, training, finder vs a chosen model, pooled replay), conventions for the merged-peak rule and the cut-valley generator, two gotchas — `docs/RUNBOOK_vfe_peak_ml.md`, `docs/RUNBOOK_vfe_peak_ml-DETAIL.md`, commit 4d3b144.
- 2026-11-13 — Merge model against today's finder on the held-out shots, and one replay folder with all held-out `incoming_A` shots (`replay_index.csv` maps shot number to run); GUI self-test `replay OK`, window opened — `outputs/real/finder_vs_ml_merge/` (local), `diagnostics/finder_vs_ml.py` options (commit 86b15d4), `diagnostics/make_replay_run.py` options (untracked).
- 2026-11-13 — Two models trained, U-Net 32/4 seed 0: P0 relabel only and with synthetic merges; real validation stays real-only, synthetic validation scored separately — `outputs/real/p0finder/u32l4`, `outputs/real/merge/u32l4` (local), `src/train_real.py`, commit 86b15d4.
- 2026-11-13 — Datasets built beside the old one: P0 from the finder's peak 0, then synthetic merged shots by cutting the valley between two card pulses (one peak under 0.3 A valley depth) — `src/relabel_p0_finder.py`, `src/make_merges.py`, commit 86b15d4; `data/real_shots_p0finder.npz`, `data/real_shots_merge.npz` (local).
- 2026-11-13 — Today's finder run on all 2984 shots: 4 peaks on every shot, P0 kind turnover 2902 / shoulder 82, 341 s with 12 workers — `diagnostics/finder_all_shots.py` (commit 86b15d4), `outputs/real/finder_all/shots.csv`, `full_run.log` (local).
- 2026-11-13 — Runbook steps 15–16, four conventions (P0 label from the finder's rule, synthetic merges allowed, tester safeguards read-only, limits unchanged) and one gotcha — `docs/RUNBOOK_vfe_peak_ml.md`, `docs/RUNBOOK_vfe_peak_ml-DETAIL.md`.
- 2026-11-13 — Limits study on the production A cards, one record per board: table and plots of board means against the current windows; limits unchanged — `~/Documents/vfe_peak_ml/limits_study/diagnostics/2026-11-13_limits_distribution.{py,csv,pdf}`.
- 2026-11-13 — Working copy brought to f05695f and completed from the other folder; a second rsync dry run lists 0 differences.
- 2026-11-12 — Runbook steps 10–14 (YAMLs into `data/archive_yaml`, today's finder vs the network, tester GUI on the laptop, replay folder, replay GUI with review), new conventions and four gotchas; `knowledge/` created (`lvr_tester.md` git-ignored); `docs/IDEAS.md` — `docs/RUNBOOK_vfe_peak_ml.md`, `docs/RUNBOOK_vfe_peak_ml-DETAIL.md`.
- 2026-11-12 — Per-peak review strip in replay: Finder / ML / Neither per flagged landmark, Prev / Next, F / M / N keys, choices written to `review.csv` (first real choices saved) — `lvr_tester/ui/review_strip.py`, `plot_panel.py`, branch `ml-peak-review` (uncommitted).
- 2026-11-12 — Replay mode `--replay DIR`: finder (left) and network (right) side by side, zoomed to the landmarks, Zoom out button, disagreements over 0.02 ms in red; `selftest --replay` prints `replay OK` — `main.py`, `lvr_tester/ui/plot_panel.py`; replay folder built by `diagnostics/make_replay_run.py` (untracked) into `~/Documents/vfe_peak_ml/replay/`.
- 2026-11-12 — Tester GUI runs on this laptop: launcher `~/Documents/vfe_peak_ml/run_lvr_tester.sh`, desktop icons, `_runner_python` falls back off Windows so Check instruments no longer crashes; tester pytest matches `main` (14 failed / 809 passed / 19 skipped).
- 2026-11-12 — lvr-tester cloned to `~/Documents/vfe_peak_ml/lvr-tester` at `main` 3bbad75, remotes `origin` (GitLab) and `github` (private mirror, not yet pushed).
- 2026-11-12 — Today's finder vs the network on held-out shots, judging tool, three new PDF pages — `diagnostics/finder_vs_ml.py`, `diagnostics/judge_disagreements.py`, `src/report_real.py`, commit 156ce20.
- 2026-11-12 — Runbook updated: working copy recorded, step 9 expect = 11-page real-only write-up, P0 decision in Conventions, gotcha for step-2 files lost from `/tmp` — `docs/RUNBOOK_vfe_peak_ml.md`, `docs/RUNBOOK_vfe_peak_ml-DETAIL.md`, commit 7541afb.
- 2026-11-12 — Working folder `~/Documents/vfe_peak_ml/` made: git clone of the repo at 0a9b4ff plus copies of the local `data/`, `outputs/`, `writeup/plots/real/`, `writeup/outputs/` and real diagnostics PNGs (2.0 GB, `git status` clean). Step-2 listing and YAML folder were lost from `/tmp` at shutdown and are not in it.
- 2026-11-11 — Decided: on the two held-out 12 V shots where network and rule disagree on P0, the rule is right (small bump at the start of the rise into A1); the network's P0 there is a miss, flagged by its low confidence — `diagnostics/2026-11-11_real_p0_misses.png` (local).
- 2026-11-11 — Both PDFs rebuilt on the new P0: ML-vs-finder (7 pages) and write-up (11 pages, real data only); report model stays U-Net 32/4 seed 0 — `outputs/real/2026-11-11_ml_vs_finder.pdf`, `writeup/outputs/vfe_peak_ml_writeup.pdf` (local).
- 2026-11-11 — All 11 models retrained on the new P0 labels (report model, transformer, 7-config sweep, seeds 1–2 for 16/3 and 32/4); old models kept in `outputs/real/old_p0inrush/`, old dataset in `data/real_shots_p0inrush.npz` (local).
- 2026-11-11 — P0 redefined as the peak just before A1 (walking rule, `src/real_data.py`), checked on 16 plotted shots across all voltages and the failing cases — `diagnostics/real_p0_rule.py`, commit 6ef0dc9.
- 2026-11-11 — Simulated data removed from the repo (toy code, outputs, deck, toy figures, toy runbook steps); `outputs/toy_gallery.png` deleted with it — commit 6ef0dc9.
- 2026-11-11 — Runbook step 19 (full write-up build), write-up conventions — `docs/RUNBOOK_vfe_peak_ml.md`, `docs/RUNBOOK_vfe_peak_ml-DETAIL.md`.
- 2026-11-11 — Full write-up, 16-page LaTeX note from toy study to real traces, unsigned steps in every footer (override given); real numbers are generated macros, PDF and real figures local; deck hybrid panel y-range fixed — `writeup/outputs/vfe_peak_ml_writeup.pdf` (local), `writeup/make_figures.py`, `writeup/src/vfe_peak_ml_writeup.tex`, commit d2179a8.
- 2026-11-11 — ML-vs-finder PDF rebuilt, 7 pages: pages 1–6 from U-Net 32/4 seed 0 (all four within 0.02 ms 91.1 → 92.4 %, within 0.1 ms 100.0 %, A1 placed on all 26 failure shots), page 7 = model-size table over all configs with seed spread — `outputs/real/2026-11-11_ml_vs_finder.pdf` (local), `src/report_real.py`, commits 8314d07, d837354.
- 2026-11-11 — Seeds 1 and 2 for U-Net 16/3 and 32/4: 16/3 spreads 87.0–93.2 % (all five within 0.02 ms, sub-bin), 32/4 spreads 92.4–93.0 % — `outputs/real/sweep_seed/` (local).
- 2026-11-11 — Runbook step 16 carries the sweep loop and seed re-runs; sweep folder naming in Conventions — commit 509116e.
- 2026-11-11 — Model-size sweep, 7 configs at seed 0: no larger model beats the base-16 U-Net (0.46M; 93.2 % of shots with all five within 0.02 ms on sub-bin readout); U-Nets within 1 % of each other, transformers 2–3 % lower — `diagnostics.real_sweep_summary`, `outputs/real/sweep/` (local).
- 2026-11-11 — Pushed the five code files behind runbook steps 15, 16 and 18 (`src/model.py`, `src/real_data.py`, `src/train_real.py`, `src/report_real.py`, `diagnostics/real_sweep_summary.py`), no data; `.claude-backups/` added to `.gitignore` — commit 7bafced.
- 2026-11-11 — Runbook split into card + detail, real-trace steps 12–18 added, superseded gotchas pruned — `docs/RUNBOOK_vfe_peak_ml.md`, `docs/RUNBOOK_vfe_peak_ml-DETAIL.md`, commit 85eabef.
- 2026-11-11 — ML-vs-existing-finder comparison PDF, 6 pages, unsigned steps named on every page — `outputs/real/2026-11-11_ml_vs_finder.pdf` (local, git-ignored), `src/report_real.py`.

## Done (archive)
- 2026-11-11 — Failure-shot check: both models place all five landmarks on every shot where the existing finder found three — `diagnostics/real_hard_cases.py`, plot local and git-ignored.
- 2026-11-11 — Real-trace U-Net and transformer trained and scored on held-out runs; both met the done criteria (median ≤ 0.02 ms, ≥ 99 % within 0.1 ms), U-Net ahead — `outputs/real/{unet,transformer}/metrics.json` (local).
- 2026-11-11 — Real dataset built from the tester archive, finder output kept for every shot — `micromamba run -n $ENV python -m src.real_data ...`, `data/real_shots.npz` (local).
- 2026-11-11 — LeCroy `.trc` reader validated: current at the finder's reported time equals its reported amplitude; established that the finder reports the raw sample maximum — `src/trc.py`, commit 4ac652c.
- 2026-11-11 — Real-trace pipeline code pushed, no data — commit 4ac652c.
- 2026-11-10 — ±2-sample study on apex labels (target width / epochs sweep, sub-sample decoding, oracle readers); none beat the current model; new deck slide — commit 76d7df1.
- 2026-11-10 — Apex-ambiguity and decoding checks, training sweep overrides — commit e7782bd.
- 2026-11-10 — Public repo created and first push — https://github.com/santanumahata/vfe-peak-ml, commit 73bb33d.
- 2026-11-10 — Toy labels moved to the visible apex; P0 placed 10–60 samples ahead of P1; retrained, toy P0 gross miss 2.3 % → 0.8 % — `src/toy_waveforms.py`, commit 73bb33d.
- 2026-11-10 — Apex reporting with guarded snap — `src/refine.py`, commit 73bb33d.
- 2026-11-10 — Lab deck, dark-beamer style, with backup section — `talk/outputs/vfe_peak_ml.pdf`, commit 73bb33d.
- 2026-11-09 — Runbook created — `docs/RUNBOOK_vfe_peak_ml.md`.
- 2026-11-09 — Toy generator, heat-map U-Net, hybrid stage and ablation, toy results PDF — `outputs/vfe_peak_ml_report.pdf`.
