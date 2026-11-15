# Purpose: today's finder against the cross-fit network on every shot, and the review queue (largest dispute first)
# Created: 2026-11-15
# Class:   diagnostic
# Inputs:  data/real_shots_p0finder.npz, outputs/real/finder_all/shots.csv, outputs/real/crossfit/predictions_all.npz
# Outputs: outputs/real/crossfit_vs_finder/{shots.csv, review_queue.csv, summary.json} (+ .meta sidecars)
"""Finder vs cross-fit network, all shots.

  python -m diagnostics.crossfit_vs_finder [--data NPZ] [--finder CSV] [--pred NPZ] [--out DIR]

Nothing is re-run: the finder times are read from the all-shots finder table and the network positions from the
merged cross-fit predictions (every shot answered by a model that never saw its run).
A landmark is disputed when |network - finder| > 0.02 ms, or when the finder has no answer for it (one-sided).
review_queue.csv holds the shots with at least one disputed landmark: one-sided shots first, then by the largest
disputed difference, descending. Its columns are the ones diagnostics/make_replay_run.py reads.
"""
import argparse
import csv
import json
import pathlib

import numpy as np

from diagnostics.finder_vs_ml import LANDMARKS, SELF_CHECK_TOL_MS, TOL_MS, write_meta

SET_OF = {"crossfit": "train", "val_model": "val", "hard_model": "hard"}   # - | which run group a shot belongs to


def read_finder_times(path, n_shots):                            # - | finder table -> rows, (N, 5) ms
    """The finder table's rows and its P0..P4 apex times in ms (NaN = no answer), one row per shot in npz order."""
    with open(path, newline="") as fh:                           # O(N) | whole table
        rows = list(csv.DictReader(fh))                          # list of dict | one per shot
    assert len(rows) == n_shots, f"{len(rows)} finder rows for {n_shots} shots"
    assert all(int(r["idx"]) == i for i, r in enumerate(rows)), "finder rows are not in npz order"
    times = np.array([[float(r[f"{lab}_time_ms"] or "nan") for lab in LANDMARKS] for r in rows])  # (N, 5) | ms
    return rows, times
    # Total: time O(N), space O(N)


def count_block(rows):                                           # - | counts for one group of shots
    """Shots, disputed shots, disputed landmarks per name, and disputed shots per input voltage."""
    vins = sorted({r["vin"] for r in rows}, key=lambda v: int(v.rstrip("Vv")))   # O(n log n) | 9V, 10V, ...
    return {"n": len(rows),
            "shots_any_disagree": sum(r["any_disagree"] for r in rows),
            "shots_one_sided": sum(r["one_sided"] for r in rows),
            "disagree_per_landmark": {lab: sum(r[f"{lab}_disagree"] for r in rows) for lab in LANDMARKS},
            "shots_any_disagree_by_vin": {v: [sum(r["any_disagree"] for r in rows if r["vin"] == v),
                                              sum(r["vin"] == v for r in rows)] for v in vins}}
    # Total: time O(n), space O(1)


def write_csv(path, rows):                                       # - | list of dict -> CSV
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))         # - | columns in first-row order
        w.writeheader()
        w.writerows(rows)
    # Total: time O(rows), space O(1)


def main():                                                      # - | write the table, the queue, the summary
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_shots_p0finder.npz")
    ap.add_argument("--finder", default="outputs/real/finder_all/shots.csv")
    ap.add_argument("--pred", default="outputs/real/crossfit/predictions_all.npz")
    ap.add_argument("--out", default="outputs/real/crossfit_vs_finder")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    paths = {name: out / name for name in ("shots.csv", "review_queue.csv", "summary.json")}
    for p in paths.values():
        assert not p.exists(), f"{p} exists; not overwriting"

    d = np.load(a.data)                                          # - | dataset: traces, labels, run, shot
    pr = np.load(a.pred)                                         # - | cross-fit answers, one per shot
    t0, dt, good = d["t0_ms"], d["dt_ms"], d["good"]             # (N,) | bin 0 time ms, bin width ms, labelled
    n = len(t0)
    frows, f_ms = read_finder_times(a.finder, n)                 # (N, 5) | finder apex times, ms
    same_shot = all(r["run"] == str(d["run"][i]) and int(r["shot"]) == int(d["shot"][i]) for i, r in enumerate(frows))
    assert same_shot, "finder table and dataset do not list the same shots in the same order"

    # The dataset labels are the finder's own times. If the table and the labels differ, the table is not the
    # finder run the network was taught from, and a dispute would not mean what it is taken to mean.
    label_ms = t0[:, None] + d["P"] * dt[:, None]                # (N, 5) | label bins -> ms
    shift = np.abs(f_ms - label_ms)[good]                        # (n_good, 5) | ms, NaN where either is missing
    assert np.nanmax(shift) <= SELF_CHECK_TOL_MS, f"finder table and labels differ by up to {np.nanmax(shift):.4f} ms"

    # Network time from its bin position, same formula as diagnostics/finder_vs_ml.py (no confidence cut).
    ml_ms = t0[:, None] + pr["pos"] * dt[:, None]                # (N, 5) | ms
    conf = pr["conf"]                                            # (N, 5) | sigmoid peak value
    assert np.isfinite(ml_ms).all(), "a shot has no network answer"

    has_finder = np.isfinite(f_ms)                               # (N, 5) | finder gave this landmark
    diff = np.abs(ml_ms - f_ms)                                  # (N, 5) | ms, NaN where the finder has none
    disagree = ~has_finder | (diff > TOL_MS)                     # (N, 5) | one-sided, or further apart than 0.02 ms
    one_sided = (~has_finder).any(axis=1)                        # (N,) | a landmark only the network gave
    max_diff = np.where(has_finder & disagree, diff, 0.0).max(axis=1)   # (N,) | ms, 0 if no two-sided dispute

    rows = []
    for i, fr in enumerate(frows):                               # O(N) | one output row per shot
        row = {"set": SET_OF[str(pr["source"][i])], "idx": i, "run": fr["run"], "vin": fr["vin"],
               "shot": int(fr["shot"]), "fold": int(pr["fold"][i]), "finder_n_peaks": int(fr["n_peaks"])}
        for k, lab in enumerate(LANDMARKS):
            row[f"{lab}_ml_ms"] = round(float(ml_ms[i, k]), 4)
            row[f"{lab}_ml_conf"] = round(float(conf[i, k]), 3)
            row[f"{lab}_finder_ms"] = round(float(f_ms[i, k]), 4) if has_finder[i, k] else ""
            row[f"{lab}_diff_ms"] = round(float(diff[i, k]), 4) if has_finder[i, k] else ""
            row[f"{lab}_disagree"] = int(disagree[i, k])
        row["any_disagree"] = int(disagree[i].any())
        row["one_sided"] = int(one_sided[i])
        row["max_diff_ms"] = round(float(max_diff[i]), 4)
        rows.append(row)

    # Review order: shots with a landmark only one side gave, then the largest disputed difference first.
    disputed = [r for r in rows if r["any_disagree"]]            # O(N) | shots to review
    queue = sorted(disputed, key=lambda r: (not r["one_sided"], -r["max_diff_ms"], r["idx"]))   # O(n log n)

    summary = {"n_shots": n, "tol_ms": TOL_MS, "all": count_block(rows)}
    for name in ("train", "val", "hard"):
        summary[name] = count_block([r for r in rows if r["set"] == name])

    out.mkdir(parents=True, exist_ok=True)
    write_csv(paths["shots.csv"], rows)
    write_csv(paths["review_queue.csv"], queue)
    paths["summary.json"].write_text(json.dumps(summary, indent=1) + "\n")
    params = {"tol_ms": TOL_MS, "ml_readout": "argmax, cross-fit", "shots": n, "queue": len(queue),
              "queue_order": "one-sided first, then max_diff_ms descending"}
    for p in paths.values():
        write_meta(p, [a.data, a.finder, a.pred], params, script="diagnostics/crossfit_vs_finder.py")
    print(json.dumps(summary, indent=1))
    for p in paths.values():
        print(p.resolve())
    # Total: time O(N log N), space O(N)


if __name__ == "__main__":                                       # - | script entry
    main()                                                       # - | run it
