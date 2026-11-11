# VFE-board QC peak finding (ML) — TODO

**Last updated:** 2026-11-11

## Now
- Seeds 1 and 2 for U-Net base 16 / 3 levels and base 32 / 4 levels (`outputs/real/sweep_seed/<tag>_s<seed>/`) — training in the background, 4 runs.
- Model-size page (page 7) added to `src/report_real.py` — rendered alone on the seed-0 runs and checked; the full 7-page PDF not yet built.

## Next
1. When the seed re-runs finish, rebuild the ML-vs-finder PDF with the model-size page (runbook step 18) and check all 7 pages.
2. Draft `docs/algorithm_vfe_peak_ml.md` (chain flowchart, step list, blocks for the four real-chain steps) and go through sign-off one step at a time.

## Blocked
- Any further deliverable from the real-data chain — **waiting on:** sign-off of its four steps (dataset build and P0 rule, run split, training, peak readout), or a per-deliverable override as given for the 2026-11-11 PDF.
- Keep or delete `outputs/toy_gallery.png` (drawn by the pre-widening generator, superseded) — **waiting on:** decision.

## Done
- 2026-11-11 — Runbook step 16 carries the sweep loop and seed re-runs; sweep folder naming in Conventions — commit 509116e.
- 2026-11-11 — Model-size sweep, 7 configs at seed 0: no larger model beats the base-16 U-Net (0.46M; 93.2 % of shots with all five within 0.02 ms on sub-bin readout); U-Nets within 1 % of each other, transformers 2–3 % lower — `diagnostics.real_sweep_summary`, `outputs/real/sweep/` (local).
- 2026-11-11 — Pushed the five code files behind runbook steps 15, 16 and 18 (`src/model.py`, `src/real_data.py`, `src/train_real.py`, `src/report_real.py`, `diagnostics/real_sweep_summary.py`), no data; `.claude-backups/` added to `.gitignore` — commit 7bafced.
- 2026-11-11 — Runbook split into card + detail, real-trace steps 12–18 added, superseded gotchas pruned — `docs/RUNBOOK_vfe_peak_ml.md`, `docs/RUNBOOK_vfe_peak_ml-DETAIL.md`, commit 85eabef.
- 2026-11-11 — ML-vs-existing-finder comparison PDF, 6 pages, unsigned steps named on every page — `outputs/real/2026-11-11_ml_vs_finder.pdf` (local, git-ignored), `src/report_real.py`.
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

## Done (archive)
