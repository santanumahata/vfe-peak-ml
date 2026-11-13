# Purpose: run today's lvr-tester finder on EVERY shot of data/real_shots.npz; keep P0 (time, kind, amplitude) and the four peaks' apex and pulse edges
# Created: 2026-11-13
# Class:   diagnostic
# Inputs:  data/real_shots.npz, data/raw/ (C1+C4 .trc), <yaml_root>, <lvr_tester> checkout, ~/Downloads/results.tar
# Outputs: outputs/real/finder_all/shots.csv (+ .meta), outputs/real/finder_all/c1_to_extract.txt
"""Finder on all shots.

  python -m diagnostics.finder_all_shots <yaml_root> <lvr_tester_dir> [--workers N] [--limit N]

One row per shot, in npz order (column idx = row of the npz). Times in ms from the trigger; blank = no answer.
"""
import argparse
import csv
import multiprocessing
import pathlib
import subprocess
import time

import numpy as np

from diagnostics.finder_vs_ml import RAW, TAR, load_finder, shot_members, write_meta

OUT = pathlib.Path("outputs/real/finder_all")
PEAK_FIELDS = (("time_ms", "time_s"), ("amp_A", "amplitude"), ("start_ms", "pulse_start_t"),
               ("end_ms", "pulse_end_t"), ("level_A", "pulse_level"))
_finder = {}                                          # per worker process: analyzer class, settings, trc reader


def extract_vin(c1_members):
    """Pull the C1 (Vin) traces not on disk yet, in one tar pass, without archive owners/permissions."""
    todo = [m for m in c1_members if not (RAW / m).exists()]
    if todo:
        lst = OUT / "c1_to_extract.txt"
        lst.write_text("\n".join(todo) + "\n")
        subprocess.run(["tar", "-xf", str(TAR), "-C", str(RAW), "--no-same-owner",
                        "--no-same-permissions", "-T", str(lst)], check=True)
    return len(todo)
    # Total: time O(archive), space O(1)


def init_worker(lvr_dir):
    _finder["tools"] = load_finder(lvr_dir)
    # Total: time O(1), space O(1)


def ms(seconds):
    """Seconds to ms, blank when missing."""
    return "" if seconds is None else round(float(seconds) * 1e3, 5)
    # Total: time O(1), space O(1)


def one_shot(job):
    """Finder output for one shot as a flat row."""
    idx, c4 = job
    Analyzer, analysis, read_trc = _finder["tools"]
    row = {"idx": idx}
    try:
        wfs = {"Vin": read_trc(RAW / c4.replace("/C4--", "/C1--"), "Vin", "C1", "V"),
               "Input_current": read_trc(RAW / c4, "Input_current", "C4", "A")}
        res = Analyzer(analysis, wfs).analyze()
    except Exception as exc:                          # one bad shot must not stop the run; it is counted
        row.update({"n_peaks": -1, "warnings": f"finder raised: {exc!r}"})
        return row
    values = res.values
    peaks = (values.get("current_peak.Input_current.peaks") or {}).get("value") or []
    peaks = sorted(peaks, key=lambda p: p["time_s"])
    p0 = values.get("peak0.Input_current.time") or {}
    row.update({"n_peaks": len(peaks), "warnings": " | ".join(res.warnings),
                "vin_cross_ms": ms((values.get("t_vin_valid_threshold.Vin") or {}).get("value")),
                "P0_time_ms": ms(p0.get("value")), "P0_kind": p0.get("kind", ""),
                "P0_amp_A": (values.get("peak0.Input_current.amplitude") or {}).get("value", "")})
    for k, peak in enumerate(peaks[:4], start=1):     # more than four is recorded in n_peaks, not in columns
        for column, key in PEAK_FIELDS:
            value = peak.get(key)
            row[f"P{k}_{column}"] = ms(value) if key.endswith(("_s", "_t")) else ("" if value is None else value)
    return row
    # Total: time O(samples), UNVERIFIED inside lvr_scope; space O(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_root")
    ap.add_argument("lvr_tester")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d = np.load("data/real_shots.npz")
    members = shot_members(a.yaml_root, d)            # checked against the npz, shot by shot
    jobs = list(enumerate(members))
    if a.limit:
        jobs = jobs[:a.limit]
    n_new = extract_vin([c4.replace("/C4--", "/C1--") for _, c4 in jobs])
    print(f"{len(jobs)} shots, {n_new} Vin traces extracted", flush=True)

    columns = ["idx", "run", "vin", "shot", "archived_n_found", "n_peaks", "vin_cross_ms",
               "P0_time_ms", "P0_kind", "P0_amp_A"]
    columns += [f"P{k}_{column}" for k in range(1, 5) for column, _ in PEAK_FIELDS] + ["warnings"]
    t_start, rows = time.time(), []
    with multiprocessing.Pool(a.workers, initializer=init_worker, initargs=(a.lvr_tester,)) as pool:
        for n, row in enumerate(pool.imap(one_shot, jobs, chunksize=8)):   # imap keeps npz order
            i = row["idx"]
            row.update({"run": d["run"][i], "vin": d["vin"][i], "shot": int(d["shot"][i]),
                        "archived_n_found": int(d["n_found"][i])})
            rows.append(row)
            if n % 250 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t_start:.0f}s", flush=True)

    csv_path = OUT / "shots.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["n_peaks"]] = counts.get(row["n_peaks"], 0) + 1
    kinds = {}
    for row in rows:
        kinds[row.get("P0_kind", "") or "none"] = kinds.get(row.get("P0_kind", "") or "none", 0) + 1
    assert len(rows) == len(jobs), "row count differs from shot count"
    assert [r["idx"] for r in rows] == [i for i, _ in jobs], "rows are not in npz order"
    print(f"shots {len(rows)}   peaks found per shot {dict(sorted(counts.items()))}   P0 kind {kinds}   "
          f"{time.time() - t_start:.0f}s")
    write_meta(csv_path, ["data/real_shots.npz", str(RAW), a.yaml_root,
                          str(pathlib.Path(a.lvr_tester) / "configuration/base/measurement_base.yaml")],
               {"workers": a.workers, "limit": a.limit, "finder": "today's settings, measurement_base.yaml"},
               script="diagnostics/finder_all_shots.py")
    print(csv_path.resolve())
    # Total: time O(shots x samples / workers), space O(shots)


if __name__ == "__main__":
    main()
