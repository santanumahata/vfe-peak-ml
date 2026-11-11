"""Check a candidate P0 rule on real shots: P0 = last ripple peak before the dip that precedes A1.

  python -m diagnostics.real_p0_rule <out.png>

Walk back from the rise into A1 down to the dip, then up to the first local maximum = P0.
Plots 16 shots (4 per input voltage) from 14 ms to A1 + 0.5 ms and prints the
P0 -> A1 gap and the dip depth per voltage.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RISE_FRAC = 0.4      # below this fraction of A1's height the rise into A1 has not started
TOL_A = 0.005        # current wiggle tolerated while walking, amps
MAX_MS = 3.0         # a walk longer than this is a rule failure


def p0_rule(x, a1, dt_ms):
    """Return (dip, p0, ok). Walk back from the rise into A1: down to the dip, then up to the peak."""
    below = np.flatnonzero(x[:a1] < RISE_FRAC * x[a1])
    if below.size == 0:
        return a1, a1, False
    j = int(below[-1])                                # on the rise, just above the dip
    while j > 0 and x[j - 1] <= x[j] + TOL_A:         # still falling going back -> not at the dip yet
        j -= 1
    dip = j + int(np.argmin(x[j:int(below[-1]) + 1]))  # lowest point of the walked stretch
    j = dip
    while j > 0 and x[j - 1] >= x[j] - TOL_A:         # still rising going back -> not at the peak yet
        j -= 1
    p0 = j + int(np.argmax(x[j:dip + 1]))             # highest point of the walked stretch
    ok = (a1 - p0) * dt_ms <= MAX_MS
    return dip, p0, ok


def main():
    d = np.load("data/real_shots.npz")
    X, P, good, vin = d["X"][:, 0], d["P"], d["good"], d["vin"]
    t0, dt = float(d["t0_ms"][0]), float(d["dt_ms"][0])
    tm = t0 + dt * np.arange(X.shape[1])
    idx = np.flatnonzero(good)
    gap, depth, at_edge = {}, {}, {}
    res = {}
    for i in idx:
        a1 = int(round(P[i, 1]))
        dip, p0, ok = p0_rule(X[i], a1, dt)
        res[i] = (dip, p0)
        v = vin[i]
        gap.setdefault(v, []).append((a1 - p0) * dt)
        depth.setdefault(v, []).append((X[i, p0] - X[i, dip]) / X[i, a1])
        at_edge.setdefault(v, []).append((not ok) or p0 == dip)
    print(f"shots with finder labels: {len(idx)}")
    for v in sorted(gap, key=lambda s: int(s[:-1])):
        g = np.array(gap[v]); r = np.array(depth[v]); e = np.array(at_edge[v])
        print(f"{v:>4}: n={len(g):4d}  P0->A1 gap ms median {np.median(g):.3f} "
              f"[5%,95%] {np.percentile(g, 5):.3f} {np.percentile(g, 95):.3f}  "
              f"P0-dip prominence / A1 median {np.median(r):.2f} max {r.max():.2f}  "
              f"rule hit a window edge: {e.sum()}")
    rng = np.random.default_rng(1)
    pick = []
    for v in sorted(set(vin[idx]), key=lambda s: int(s[:-1])):
        pick += list(rng.choice(idx[vin[idx] == v], 4, replace=False))
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    for ax, i in zip(axes.flat, pick):
        a1 = int(round(P[i, 1])); dip, p0 = res[i]
        w = (tm >= 14) & (tm <= tm[a1] + 0.5)
        ax.plot(tm[w], X[i, w], color="#222222", lw=0.8)
        ax.plot(tm[a1], X[i, a1], "x", color="#C0392B", ms=8, mew=2)
        ax.plot(tm[dip], X[i, dip], "^", color="#7F7F7F", ms=6)
        ax.plot(tm[p0], X[i, p0], "o", mfc="none", mec="#1F77B4", ms=9, mew=1.6)
        ax.set_title(f"{vin[i]}  shot {d['shot'][i]}  gap {(a1 - p0) * dt:.2f} ms", fontsize=8)
    fig.suptitle("red x = finder A1, grey ^ = dip, blue o = candidate P0", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(sys.argv[1], dpi=110)
    print(sys.argv[1])
    # Total: time O(N n), space O(N n)


if __name__ == "__main__":
    main()
