# Purpose: run the lvr-tester peak finder (today's settings) and the U-Net on the same held-out shots; flag disagreements
# Created: 2026-11-12
# Class:   diagnostic
# Inputs:  data/real_shots.npz, outputs/real/sweep/u32l4/predictions.npz, data/raw/ (C1+C4 .trc),
#          <yaml_root> (result YAMLs from the archive), <lvr_tester> checkout, ~/Downloads/results.tar
# Outputs: outputs/real/finder_vs_ml/shots.csv, summary.json (+ .meta sidecars)
"""ML vs finder on the held-out shots.

  python -m diagnostics.finder_vs_ml <yaml_root> <lvr_tester_dir> [--limit N]

The finder is the vendored lvr_scope WaveformAnalyzer fed with configuration/base/measurement_base.yaml.
The model side is read from the saved held-out predictions (argmax readout), not re-run.
A landmark disagrees when |ML - finder| > 0.02 ms or exactly one side has no answer.
"""
import argparse
import csv
import datetime
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import yaml

from src.real_data import collect

LANDMARKS = ["P0", "P1", "P2", "P3", "P4"]          # P0 = bump before A1; P1..P4 = A1, B1, A2, B2 in time order
TOL_MS = 0.02
SELF_CHECK_TOL_MS = 0.001
DATA = "data/real_shots.npz"
PRED = "outputs/real/sweep/u32l4/predictions.npz"
RAW = pathlib.Path("data/raw")
TAR = pathlib.Path.home() / "Downloads" / "results.tar"
OUT = pathlib.Path("outputs/real/finder_vs_ml")


def load_finder(lvr_dir):
    """Import the vendored lvr_scope and build today's analysis settings."""
    sys.path.insert(0, str(pathlib.Path(lvr_dir) / "vendor"))
    from lvr_scope.analysis import WaveformAnalyzer
    from lvr_scope.config_loader import from_dict
    from lvr_scope.waveform_io import read_trc
    base = pathlib.Path(lvr_dir) / "configuration" / "base" / "measurement_base.yaml"
    analysis = from_dict(yaml.safe_load(base.read_text())).analysis
    return WaveformAnalyzer, analysis, read_trc
    # Total: time O(1), space O(1)


def shot_members(yaml_root, d):
    """C4 archive member per shot, rebuilt with the same collect() that built the npz, then checked against it."""
    have = {str(p.relative_to(RAW)) for p in RAW.rglob("*.trc")}
    recs = collect(yaml_root, have)
    if len(recs) != len(d["run"]):
        raise SystemExit(f"collect() gave {len(recs)} shots, npz has {len(d['run'])}: record order cannot be trusted")
    for i, r in enumerate(recs):
        if r["run"] != d["run"][i] or str(r["shot"]) != str(d["shot"][i]):
            raise SystemExit(f"shot {i}: collect() {r['run']}/{r['shot']} != npz {d['run'][i]}/{d['shot'][i]}")
    return [r["member"] for r in recs]
    # Total: time O(total YAML size + files), space O(shots)


def extract_vin(c1_members):
    """Pull the C1 (Vin) traces that are not on disk yet, in one tar pass, without archive owners/permissions."""
    todo = [m for m in c1_members if not (RAW / m).exists()]
    if not todo:
        return 0
    lst = OUT / "c1_to_extract.txt"
    lst.write_text("\n".join(todo) + "\n")
    subprocess.run(["tar", "-xf", str(TAR), "-C", str(RAW), "--no-same-owner",
                    "--no-same-permissions", "-T", str(lst)], check=True)
    return len(todo)
    # Total: time O(archive), space O(1)


def run_finder(Analyzer, analysis, read_trc, c1, c4):
    """Finder times in ms for P0..P4 (NaN where it has no answer), number of card peaks, warnings."""
    wfs = {"Vin": read_trc(RAW / c1, "Vin", "C1", "V"),
           "Input_current": read_trc(RAW / c4, "Input_current", "C4", "A")}
    res = Analyzer(analysis, wfs).analyze()
    peaks = (res.values.get("current_peak.Input_current.peaks") or {}).get("value") or []
    p0 = (res.values.get("peak0.Input_current.time") or {}).get("value")
    t = np.full(5, np.nan)
    if p0 is not None:
        t[0] = float(p0) * 1e3
    if len(peaks) == 4:
        t[1:] = sorted(float(p["time_s"]) * 1e3 for p in peaks)
    return t, len(peaks), list(res.warnings)
    # Total: time O(samples) per shot, UNVERIFIED inside lvr_scope, space O(samples)


def write_meta(path, inputs, params, script="diagnostics/finder_vs_ml.py"):
    """Provenance sidecar <file>.meta; `script` names the producer."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = "no-repo"
    meta = {"script": script, "commit": sha or "no-repo", "inputs": inputs,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "params": params}
    pathlib.Path(str(path) + ".meta").write_text(json.dumps(meta, indent=1) + "\n")
    # Total: time O(1), space O(1)


def main():
    global OUT                                                  # the helpers above write under OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_root")
    ap.add_argument("lvr_tester")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--pred", default=PRED)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-conf", type=float, default=0.0)     # below this the model's landmark counts as absent
    a = ap.parse_args()
    OUT = pathlib.Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)

    d = np.load(a.data)
    pr = np.load(a.pred)
    t0, dt = d["t0_ms"], d["dt_ms"]
    members = shot_members(a.yaml_root, d)

    # held-out clean runs and failure runs, with the model's saved argmax positions (bins) and confidences
    sets = [("val", pr["val_idx"], pr["val_pos"], pr["val_conf"]),
            ("hard", pr["hard_idx"], pr["hard_pos"], pr["hard_conf"])]
    rows_in = [(name, int(i), pos[k], conf[k]) for name, idx, pos, conf in sets for k, i in enumerate(idx)]
    if a.limit:
        rows_in = rows_in[:a.limit]

    c1 = {i: members[i].replace("/C4--", "/C1--") for _, i, _, _ in rows_in}
    n_new = extract_vin(list(c1.values()))
    print(f"{len(rows_in)} shots, {n_new} Vin traces extracted", flush=True)

    Analyzer, analysis, read_trc = load_finder(a.lvr_tester)
    rows, fails, t_start = [], 0, time.time()
    for n, (name, i, pos, conf) in enumerate(rows_in):
        ml_ms = np.where(conf >= a.min_conf, t0[i] + pos * dt[i], np.nan)   # NaN = model says not there
        try:
            f_ms, n_peaks, warns = run_finder(Analyzer, analysis, read_trc, c1[i], members[i])
        except Exception as exc:                 # one bad shot must not stop the comparison; it is counted
            f_ms, n_peaks, warns = np.full(5, np.nan), -1, [f"finder raised: {exc!r}"]
            fails += 1
        archived = t0[i] + d["P"][i, 1:] * dt[i] if d["good"][i] else np.full(4, np.nan)
        row = {"set": name, "idx": i, "run": d["run"][i], "vin": d["vin"][i], "shot": int(d["shot"][i]),
               "finder_n_peaks": n_peaks, "archived_n_found": int(d["n_found"][i]), "warnings": " | ".join(warns)}
        for k, lab in enumerate(LANDMARKS):
            both = np.isfinite(ml_ms[k]) and np.isfinite(f_ms[k])
            one = np.isfinite(ml_ms[k]) != np.isfinite(f_ms[k])
            row[f"{lab}_ml_ms"] = round(float(ml_ms[k]), 4) if np.isfinite(ml_ms[k]) else ""
            row[f"{lab}_ml_conf"] = round(float(conf[k]), 3)
            row[f"{lab}_finder_ms"] = round(float(f_ms[k]), 4) if np.isfinite(f_ms[k]) else ""
            row[f"{lab}_disagree"] = int(one or (both and abs(ml_ms[k] - f_ms[k]) > TOL_MS))
            if k:
                same = np.isfinite(archived[k - 1]) and np.isfinite(f_ms[k]) and \
                    abs(archived[k - 1] - f_ms[k]) <= SELF_CHECK_TOL_MS
                row[f"{lab}_archived_ms"] = round(float(archived[k - 1]), 4) if np.isfinite(archived[k - 1]) else ""
                row[f"{lab}_same_as_archived"] = int(same)
        row["any_disagree"] = int(any(row[f"{lab}_disagree"] for lab in LANDMARKS))
        rows.append(row)
        if n % 100 == 0:
            print(f"{n}/{len(rows_in)}  {time.time() - t_start:.0f}s", flush=True)

    csv_path = OUT / "shots.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"n_shots": len(rows), "finder_exceptions": fails, "tol_ms": TOL_MS,
               "seconds": round(time.time() - t_start, 1)}
    for name in ("val", "hard"):
        sub = [r for r in rows if r["set"] == name]
        if not sub:
            continue
        s = {"n": len(sub), "finder_not_4_peaks": sum(r["finder_n_peaks"] != 4 for r in sub),
             "shots_any_disagree": sum(r["any_disagree"] for r in sub),
             "disagree_per_landmark": {lab: sum(r[f"{lab}_disagree"] for r in sub) for lab in LANDMARKS}}
        clean = [r for r in sub if r["archived_n_found"] == 4 and r["finder_n_peaks"] == 4]
        s["self_check_P1_P4_same_as_archived"] = {
            "n_shots": len(clean),
            "all_four_same": sum(all(r[f"{lab}_same_as_archived"] for lab in LANDMARKS[1:]) for r in clean)}
        s["shots_any_disagree_by_vin"] = {
            v: [sum(r["any_disagree"] for r in sub if r["vin"] == v), sum(r["vin"] == v for r in sub)]
            for v in sorted({r["vin"] for r in sub})}
        summary[name] = s
    sum_path = OUT / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=1) + "\n")

    inputs = [a.data, a.pred, str(RAW), a.yaml_root, str(pathlib.Path(a.lvr_tester) / "configuration/base/measurement_base.yaml")]
    params = {"tol_ms": TOL_MS, "ml_readout": "argmax", "model": a.pred, "min_conf": a.min_conf, "limit": a.limit}
    write_meta(csv_path, inputs, params)
    write_meta(sum_path, inputs, params)
    print(json.dumps(summary, indent=1))
    print(csv_path.resolve())
    print(sum_path.resolve())
    # Total: time O(shots x samples), space O(shots)


if __name__ == "__main__":
    main()
