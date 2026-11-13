# Purpose: add synthetic merged-peak shots to the dataset by cutting the valley between two card pulses out of real traces
# Created: 2026-11-13
# Class:   pipeline
# Inputs:  data/real_shots_p0finder.npz
# Outputs: data/real_shots_merge.npz (+ .meta)
"""Synthetic merges (the one allowed use of non-real traces, decided 2026-11-13).

  python -m src.make_merges [--data data/real_shots_p0finder.npz] [--out data/real_shots_merge.npz] [--seed 0]

Every real shot is kept as it is. One synthetic shot is appended per labelled training or validation shot, in the
same run (so it lands in the same split). Hard-test runs get none. A synthetic shot is a real trace with the
stretch between two neighbouring card pulses removed at a chosen current level, which sets how deep the valley
between them still is. Rule: valley depth under MERGE_DEPTH_A = one peak, labelled at the taller apex; the lower
landmark gets NaN (= not there). Otherwise both keep their labels.
"""
import argparse
import datetime
import json
import pathlib
import subprocess

import numpy as np

from src.train_real import split_runs                            # - | the run-level split used for training

MERGE_DEPTH_A = 0.3                                              # - | under this valley depth two pulses are one peak
MAX_DEPTH_A = 0.7                                                # - | deepest synthetic valley; real ones start at 0.77 A
APEX_SEARCH = 10                                                 # - | bins around the label to find the binned apex
SMOOTH_HALF = 5                                                  # - | bins each side of the join that get smoothed
SMOOTH_WIDTH = 5                                                 # - | moving-average width at the join, bins


def binned_apex(x, p):                                           # - | label (fractional bin) -> apex bin
    lo = max(int(round(p)) - APEX_SEARCH, 0)                     # - | window start
    return lo + int(np.argmax(x[lo:int(round(p)) + APEX_SEARCH + 1]))  # O(1) | highest bin near the label
    # Total: time O(1), space O(1)


def smooth_join(y, j):                                           # - | soften the kink where the two edges meet
    lo, hi = max(j - SMOOTH_HALF, SMOOTH_WIDTH), min(j + SMOOTH_HALF, len(y) - SMOOTH_WIDTH)
    kernel = np.ones(SMOOTH_WIDTH, np.float32) / SMOOTH_WIDTH    # - | moving average
    pad = SMOOTH_WIDTH // 2                                      # - | bins needed each side
    y[lo:hi] = np.convolve(y[lo - pad:hi + pad], kernel, mode="valid")  # O(1) | only the join region changes
    # Total: time O(1), space O(1)


def cut_valley(x, p, pair, depth_target):                        # - | one synthetic shot
    """Remove the valley between landmarks pair and pair+1 down to the requested depth.

    Returns (trace, labels, measured depth in A) or None when the real valley is not deep enough to cut.
    """
    ia, ib = binned_apex(x, p[pair]), binned_apex(x, p[pair + 1])   # - | apex bins of the two pulses
    level = min(x[ia], x[ib]) - depth_target                     # A | current level at which the edges are joined
    between = x[ia:ib + 1]                                       # (ib-ia+1,) | apex to apex
    below = np.flatnonzero(between <= level)                     # - | bins of the valley under that level
    if len(below) == 0:                                          # - | valley never reaches the level
        return None
    i_fall, i_rise = ia + below[0], ia + below[-1]               # - | first and last bin under the level
    n_cut = i_rise - i_fall + 1                                  # - | bins removed

    # Join the falling edge of the first pulse to the rising edge of the second; later samples move earlier and
    # the freed tail holds the last value, as the training augmentation does for a shift.
    y = np.concatenate([x[:i_fall], x[i_rise + 1:], np.full(n_cut, x[-1], x.dtype)])  # (n,) | same length
    smooth_join(y, i_fall)
    q = p.copy()                                                 # (5,) | labels, fractional bins
    q[pair + 1:] -= n_cut                                        # - | everything after the cut moves earlier

    # Depth is measured on the finished trace, so the label follows what the network will see.
    ja, jb = binned_apex(y, q[pair]), binned_apex(y, q[pair + 1])
    depth = float(min(y[ja], y[jb]) - y[ja:jb + 1].min())        # A | lower apex minus valley minimum
    if depth < MERGE_DEPTH_A:                                    # - | one peak: the lower landmark is not there
        q[pair if y[ja] < y[jb] else pair + 1] = np.nan
    return y, q, depth
    # Total: time O(n), space O(n)


def main():                                                      # - | write the dataset with merges appended
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_shots_p0finder.npz")
    ap.add_argument("--out", default="data/real_shots_merge.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert not pathlib.Path(a.out).exists(), f"{a.out} exists; not overwriting"

    d = dict(np.load(a.data))                                    # dict of arrays | real shots
    n_real = len(d["good"])
    train, val, hard = split_runs(d["run"], d["n_found"])        # (N,) bool | same split as training
    sources = np.flatnonzero((train | val) & d["good"])          # - | labelled, not in a hard-test run
    rng = np.random.default_rng(a.seed)

    traces, labels, src, pairs, depths = [], [], [], [], []
    for i in sources:                                            # O(N) | one synthetic shot per source
        pair = int(rng.integers(1, 4))                           # - | 1: A1+B1, 2: B1+A2, 3: A2+B2
        made = cut_valley(d["X"][i, 0], d["P"][i], pair, float(rng.uniform(0, MAX_DEPTH_A)))
        if made is None:
            continue
        traces.append(made[0]); labels.append(made[1]); src.append(i); pairs.append(pair); depths.append(made[2])
    src = np.array(src); n_syn = len(src)
    assert not hard[src].any(), "a synthetic shot came from a hard-test run"

    # Per-shot arrays: synthetic rows copy their source, except what the cut changed or made meaningless.
    out = {k: np.concatenate([v, v[src]]) for k, v in d.items()}
    out["X"][n_real:, 0] = np.array(traces)
    out["P"][n_real:] = np.array(labels)
    out["finder_t_ms"][n_real:] = np.nan                         # - | finder times belong to the uncut trace
    out["synthetic"] = np.arange(n_real + n_syn) >= n_real
    out["src_idx"] = np.concatenate([np.full(n_real, -1), src])
    out["merge_pair"] = np.concatenate([np.zeros(n_real, int), pairs])
    out["merge_depth_A"] = np.concatenate([np.full(n_real, np.nan), depths])
    np.savez_compressed(a.out, **out)                            # O(N N_BINS) | write

    merged = np.isnan(np.array(labels)).any(axis=1)              # (n_syn,) | one landmark absent
    params = {"merge_depth_A": MERGE_DEPTH_A, "max_depth_A": MAX_DEPTH_A, "seed": a.seed, "real": n_real,
              "synthetic": n_syn, "skipped": int(len(sources) - n_syn), "merged": int(merged.sum()),
              "from_train": int(train[src].sum()), "from_val": int(val[src].sum()),
              "merged_by_pair": {p: int((merged & (np.array(pairs) == p)).sum()) for p in (1, 2, 3)}}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"script": "src/make_merges.py", "commit": commit, "inputs": [a.data],
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "params": params}
    pathlib.Path(a.out + ".meta").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(a.out, "  ".join(f"{k}={v}" for k, v in params.items()))
    print(pathlib.Path(a.out).resolve())
    # Total: time O(N N_BINS), space O(N N_BINS)


if __name__ == "__main__":                                       # - | script entry
    main()                                                       # - | run it
