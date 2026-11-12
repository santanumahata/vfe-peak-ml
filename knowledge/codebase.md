# Codebase — this repository

- `src/real_data.py` builds `data/real_shots.npz` from the archive (binning, P0 rule, labels). Source: file header.
- `src/model.py` holds the networks and `normalise` (median subtraction, divide by max |value|);
  every inference path reuses it. Source: `src/model.py:8`; runbook Conventions.
- `src/train_real.py` trains the U-Net / transformer; the report model is the U-Net base 32,
  4 levels, seed 0 in `outputs/real/sweep/u32l4`. Source: runbook steps 5–6, 8.
- `src/report_real.py` writes the ML-vs-finder PDF (7 pages, 10 with step 11's output). Source: runbook step 8.
- `diagnostics/finder_vs_ml.py` re-runs today's tester finder on the held-out shots and
  writes `outputs/real/finder_vs_ml/shots.csv`; disagree = more than 0.02 ms. Source: runbook step 11.
- `diagnostics/make_replay_run.py` turns one archived run into a folder the tester GUI can
  replay (`shots/*.npz`, finder CSVs, `ml_peaks.csv`), keeping an existing `review.csv`. Source: runbook step 13.
- `diagnostics/judge_disagreements.py` is a matplotlib click-through for every disagreement
  (`m/f/b/n`, resumable), writing `outputs/real/finder_vs_ml/judgments.csv`; not yet run.
  Source: its file header.
- Every output file gets a `.meta` sidecar via `write_meta` in `diagnostics/finder_vs_ml.py`. Source: that function.
- `data/`, `outputs/real/`, `diagnostics/*_real_*`, `writeup/plots/real/`, `writeup/outputs/`
  are git-ignored; the public repository carries code and docs only. Source: `.gitignore`.
