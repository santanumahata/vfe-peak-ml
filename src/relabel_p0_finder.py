# Purpose: copy data/real_shots.npz with the P0 label replaced by the tester finder's peak 0 (P1..P4 untouched)
# Created: 2026-11-13
# Class:   pipeline
# Inputs:  data/real_shots.npz, outputs/real/finder_all/shots.csv (runbook step 15)
# Outputs: data/real_shots_p0finder.npz (+ .meta)
"""P0 label from the finder.

  python -m src.relabel_p0_finder [--data data/real_shots.npz] [--finder outputs/real/finder_all/shots.csv]
                                  [--out data/real_shots_p0finder.npz]

The old file is not changed. Every array is copied as it is except P[:, 0]; the old P0 bins are kept as P0_rule.
"""
import argparse
import csv
import datetime
import json
import pathlib
import subprocess

import numpy as np

# The four card peaks in the dataset are the finder's own; if today's finder moved one, the two files do not
# describe the same peaks and the P0 swap must not happen silently.
MAX_CARD_PEAK_SHIFT_MS = 1e-3                                    # - | allowed |finder now - label|, ms


def read_finder(path, n_shots):                                  # - | finder table -> arrays
    """P0 time (ms), P0 kind and the four card-peak times (ms), one row per shot in npz order."""
    with open(path, newline="") as fh:                           # O(N) | whole table
        rows = list(csv.DictReader(fh))                          # list of dict | one per shot
    assert len(rows) == n_shots, f"{len(rows)} finder rows for {n_shots} shots"
    assert all(int(r["idx"]) == i for i, r in enumerate(rows)), "finder rows are not in npz order"
    p0_ms = np.array([float(r["P0_time_ms"] or "nan") for r in rows])               # (N,) | ms, nan = no P0
    kind = np.array([r["P0_kind"] for r in rows])                                   # (N,) | turnover / shoulder
    card_ms = np.array([[float(r[f"P{k}_time_ms"] or "nan") for k in range(1, 5)] for r in rows])  # (N, 4) | ms
    return p0_ms, kind, card_ms
    # Total: time O(N), space O(N)


def main():                                                      # - | write the relabelled npz
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_shots.npz")
    ap.add_argument("--finder", default="outputs/real/finder_all/shots.csv")
    ap.add_argument("--out", default="data/real_shots_p0finder.npz")
    a = ap.parse_args()
    assert not pathlib.Path(a.out).exists(), f"{a.out} exists; not overwriting"

    d = dict(np.load(a.data))                                    # dict of arrays | the current dataset
    t0, dt, good = d["t0_ms"], d["dt_ms"], d["good"]             # (N,) | bin 0 time, bin width, labels exist
    p0_ms, kind, card_ms = read_finder(a.finder, len(good))

    # Same-peaks check on the shots that carry labels (good): label bins back to ms, against today's finder.
    label_ms = t0[:, None] + d["P"][:, 1:] * dt[:, None]         # (N, 4) | ms
    shift = np.abs(card_ms - label_ms)[good]                     # (n_good, 4) | ms
    assert np.nanmax(shift) <= MAX_CARD_PEAK_SHIFT_MS, f"card peaks moved by up to {np.nanmax(shift):.4f} ms"

    # Swap P0 only where the shot is labelled, so the set of labelled shots and the run split stay what they were.
    p_new = d["P"].copy()                                        # (N, 5) | fractional bins
    p0_bin = (p0_ms - t0) / dt                                   # (N,) | ms -> bin, same formula as P1..P4
    p_new[good, 0] = p0_bin[good]                                # - | finder P0 on labelled shots
    in_trace = (p_new[good, 0] >= 0) & (p_new[good, 0] < d["X"].shape[-1])
    assert in_trace.all(), "a finder P0 falls outside the binned trace"
    assert (p_new[good, 0] < p_new[good, 1]).all(), "a finder P0 is not before A1"

    moved_ms = np.abs(p_new[:, 0] - d["P"][:, 0]) * dt           # (N,) | ms, nan where the rule gave no P0
    had_rule = good & np.isfinite(d["P"][:, 0])                  # (N,) | labelled by the rule before
    d["P0_rule"] = d["P"][:, 0].copy()                           # - | old label kept for comparison
    d["P0_kind"] = kind                                          # - | finder's turnover / shoulder
    d["P"] = p_new
    np.savez_compressed(a.out, **d)                              # O(N N_BINS) | write

    params = {"p0": "finder peak 0 (outputs/real/finder_all/shots.csv), labelled shots only",
              "labelled": int(good.sum()), "newly_labelled_p0": int((good & ~had_rule).sum()),
              "moved_over_0.02ms": int((moved_ms[had_rule] > 0.02).sum()),
              "moved_over_0.1ms": int((moved_ms[had_rule] > 0.1).sum())}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"script": "src/relabel_p0_finder.py", "commit": commit, "inputs": [a.data, a.finder],
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "params": params}
    pathlib.Path(a.out + ".meta").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{a.out}  shots={len(good)}  " + "  ".join(f"{k}={v}" for k, v in params.items() if k != "p0"))
    print(pathlib.Path(a.out).resolve())
    # Total: time O(N N_BINS), space O(N N_BINS)


if __name__ == "__main__":                                       # - | script entry
    main()                                                       # - | run it
