# Purpose: show the candidate peaks (raw samples highest within +-w) for three widths w, on two disputed shots
# Created: 2026-11-15
# Class:   diagnostic
# Inputs:  <replay dir>/campaign_replay/shots/shot_*.npz, outputs/real/crossfit_vs_finder/review_queue.csv
# Outputs: <out.png>
"""Candidate peaks on the raw current, no smoothing.

  python -m diagnostics.candidate_maxima <replay_dir> <out.png>

A candidate is a raw sample that is the highest one within +-w of itself. Nothing is averaged.
Rows: the first shot of the review queue with a disputed P0, and the first with a disputed P1.
Columns: w = 0.02, 0.05, 0.10 ms. Each panel shows the raw trace around the disputed peak, the finder's and the
network's positions, and the candidates; the count in the title is the number of candidates in view.
"""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter1d

QUEUE = "outputs/real/crossfit_vs_finder/review_queue.csv"
WIDTHS_MS = (0.02, 0.05, 0.10)      # half-widths w tried
MARGIN_MS = 0.6                     # shown on each side of the two disputed positions


def candidate_maxima(y, half_n):
    """Indices of the raw samples that are the highest within +-half_n samples (first sample of a flat top)."""
    top = maximum_filter1d(y, size=2 * half_n + 1, mode="nearest")   # running maximum over the +-half_n window
    idx = np.flatnonzero(y == top)                                   # samples equal to their window's maximum
    keep = np.r_[True, np.diff(idx) > half_n]                        # equal samples on one top: keep the first
    return idx[keep]
    # Total: time O(samples), space O(samples); maximum_filter1d cost UNVERIFIED


def first_disputed(rows, lab):
    """(shot_index, row) of the first queue row whose landmark `lab` is disputed; shot_index = row number from 1."""
    for k, r in enumerate(rows, start=1):
        if r[f"{lab}_disagree"] == "1":
            return k, r
    raise SystemExit(f"no disputed {lab} in {QUEUE}")
    # Total: time O(rows), space O(1)


def main():
    camp = pathlib.Path(sys.argv[1]) / "campaign_replay"
    with open(QUEUE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    picks = [("P0",) + first_disputed(rows, "P0"), ("P1",) + first_disputed(rows, "P1")]
    fig, axes = plt.subplots(len(picks), len(WIDTHS_MS), figsize=(16, 8), squeeze=False)
    for row_axes, (lab, k, r) in zip(axes, picks):
        path = next((camp / "shots").glob(f"shot_{k:05d}_V*V.npz"))
        z = np.load(path)
        t = z["Input_current__t"] * 1e3                               # s -> ms
        y = z["Input_current__y"].astype(float)                       # A
        dt_ms = float(np.median(np.diff(t)))
        f_ms, m_ms = float(r[f"{lab}_finder_ms"]), float(r[f"{lab}_ml_ms"])
        lo, hi = min(f_ms, m_ms) - MARGIN_MS, max(f_ms, m_ms) + MARGIN_MS
        view = (t >= lo) & (t <= hi)
        for ax, w in zip(row_axes, WIDTHS_MS):
            cand = candidate_maxima(y, max(1, int(round(w / dt_ms))))
            cand = cand[(t[cand] >= lo) & (t[cand] <= hi)]
            ax.plot(t[view], y[view], color="#222222", lw=0.5)
            ax.axvline(f_ms, color="#C0392B", ls="--", lw=1.2)
            ax.axvline(m_ms, color="#1F77B4", ls="--", lw=1.2)
            ax.plot(t[cand], y[cand], "o", mfc="none", mec="#E67E22", ms=7, mew=1.5)
            ax.set_title(f"{lab}  shot {k} ({r['vin']})  w = {w:.2f} ms  {len(cand)} candidates in view", fontsize=9)
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("input current (A)")
            print(f"{lab} shot {k} {r['vin']} finder {f_ms} ml {m_ms}  w {w:.2f} ms: {len(cand)} candidates "
                  f"in {hi - lo:.2f} ms")
    fig.suptitle("raw trace; red dashed = finder, blue dashed = network, orange o = candidate "
                 "(raw sample highest within +-w)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(sys.argv[2], dpi=110)
    print(pathlib.Path(sys.argv[2]).resolve())
    # Total: time O(shots x widths x samples), space O(samples)


if __name__ == "__main__":
    main()
