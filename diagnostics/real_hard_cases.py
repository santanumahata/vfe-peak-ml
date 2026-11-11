"""Where does a trained model put the five landmarks on the shots the finder got wrong?

Usage: python -m diagnostics.real_hard_cases <model_dir> [<model_dir> ...] <out_png>

For each failure shot (finder found 3 of 4): draw the trace zoomed on 15-27 ms, the
finder's three marks (red, its labels P1..P3) and each model's five (P0 is off-panel).
Prints, per model and per run, the spread of the model's A1 time across those shots,
and the A1 time it gives on the same run's clean shots at the same voltage, if any.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["P0", "P1", "P2", "P3", "P4"]
COLS = ["#888888", "#00A6D6", "#E69F00", "#56B4E9", "#D55E00"]


def main():
    *dirs, out = sys.argv[1:]
    d = np.load("data/real_shots.npz")
    X, P, t0, dt = d["X"], d["P"], d["t0_ms"], d["dt_ms"]
    run, vin, nf, good = d["run"], d["vin"], d["n_found"], d["good"]
    tm = t0[0] + dt[0] * np.arange(X.shape[-1])

    preds = {}
    for m in dirs:
        z = np.load(f"{m}/predictions.npz")
        full = np.full((len(X), 5), np.nan)
        full[z["hard_idx"]] = z["hard_pos"]
        full[z["val_idx"]] = z["val_pos"]
        preds[m] = full

    bad = np.flatnonzero(nf == 3)
    for m, full in preds.items():
        print(f"--- {m}")
        for r in sorted(set(run[bad])):
            sel = bad[run[bad] == r]
            a1 = full[sel, 1] * dt[0] + t0[0]
            same_v = np.flatnonzero((run == r) & good & (vin == vin[sel[0]]))
            ref = (full[same_v, 1] * dt[0] + t0[0]) if same_v.size else np.array([])
            other = np.flatnonzero((run == r) & good)
            lab = P[other, 1] * dt[0] + t0[0]
            print(f"  {r}  Vin {vin[sel[0]]}  failure shots {len(sel)}: model A1 "
                  f"{np.nanmean(a1):.3f} ms, spread (std) {np.nanstd(a1)*1e3:.1f} us | "
                  f"finder A1 on this run's clean shots (other Vin): "
                  f"{np.nanmean(lab):.3f} +- {np.nanstd(lab)*1e3:.0f} us (n={len(other)})"
                  + (f" | model A1 there {np.nanmean(ref):.3f}" if ref.size else ""))

    show = bad[:8]
    fig, axes = plt.subplots(len(show), 1, figsize=(11, 1.9 * len(show)), sharex=True)
    for ax, i in zip(np.atleast_1d(axes), show):
        w = (tm > 15) & (tm < 27)
        ax.plot(tm[w], X[i, 0, w], lw=0.7, color="#1f3b73")
        for j, (m, full) in enumerate(preds.items()):
            for k in range(1, 5):
                b = int(round(full[i, k]))
                ax.plot(tm[b], X[i, 0, b], ["v", "^"][j], color=COLS[k], ms=8,
                        mfc="none" if j else COLS[k])
        ax.set_title(f"{run[i]}  {vin[i]}  shot {d['shot'][i]}  (finder found 3)",
                     fontsize=8, loc="left")
        ax.set_ylabel("I [A]", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("time from trigger [ms]   "
                                       "(filled = first model, hollow = second; "
                                       "blue A1, orange B1, light blue A2, red B2)")
    fig.tight_layout(); fig.savefig(out, dpi=110); print(out)


if __name__ == "__main__":
    main()
