# Purpose: build an archived-run folder (traces, finder peaks, ML peaks) that lvr-tester --replay shows, every shot of the run, ordered by Vin then shot
# Created: 2026-11-12
# Class:   diagnostic
# Inputs:  outputs/real/finder_vs_ml/shots.csv, data/real_shots.npz, data/raw/ (C1+C4 .trc), <yaml_root>, <lvr_tester> checkout
# Outputs: <out>/campaign_replay/{shots/shot_*_V*V.npz, campaign_peaks.csv, campaign_shot_metrics.csv, ml_peaks.csv, replay.yaml,
#          ml/campaign_peaks.csv, ml/campaign_shot_metrics.csv} (+ .meta sidecars)
"""Replay folder for the lvr-tester plot panel.

  python -m diagnostics.make_replay_run <yaml_root> <lvr_tester_dir> --run RUN [--out DIR] [--table CSV]
  python -m diagnostics.make_replay_run <yaml_root> <lvr_tester_dir> --test-type TYPE --out DIR [--table CSV]

Exactly one of --run (one run) and --test-type (every run of that test type in the table, pooled) is given.
Every shot of the chosen run(s) in shots.csv is kept; shot_index runs 1..N by numeric Vin, then run, then shot.
Old shots/*.npz and the generated files are removed first; a review.csv is left alone.
The finder is run LIVE with today's settings; the ML times come from shots.csv.
The layout mirrors campaign.py (npz keys <signal>__t/__y/__unit, peaks CSV columns).
The finder CSVs are campaign.py's own rows (all columns summarise_point reads); campaign_replay/ml/ holds the same
two CSVs measured at the ML apex times by lvr_tester.ml.measure.measure_at. replay.yaml names the test type
(from the run's result YAML) so the GUI can pick the limits.
"""
import argparse
import csv
import pathlib
import re
import sys

import numpy as np
import yaml

from diagnostics.finder_vs_ml import LANDMARKS, OUT, RAW, TOL_MS, load_finder, shot_members, write_meta

GENERATED = ("campaign_peaks.csv", "campaign_shot_metrics.csv", "ml_peaks.csv", "replay.yaml", "replay_index.csv",
             "ml/campaign_peaks.csv", "ml/campaign_shot_metrics.csv")


def vin_value(label):
    """'12V' -> 12."""
    return int(str(label).strip().rstrip("Vv"))
    # Total: time O(1), space O(1)


def num(row, key):
    """Float from a shots.csv cell, NaN when empty."""
    return float(row[key]) if row.get(key, "") != "" else float("nan")
    # Total: time O(1), space O(1)


def pick_shots(table, run, yaml_root, test_type=None):
    """Shots of one run, or (test_type given) of every run of that test type; by numeric Vin, run, shot."""
    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if test_type:
        kinds = {r: run_identity(yaml_root, r)["test_type"] for r in sorted({x["run"] for x in rows})}
        left_out = sorted(r for r, k in kinds.items() if k != test_type)
        print(f"{len(kinds) - len(left_out)} runs of {test_type}; left out: {left_out}")
        rows = [r for r in rows if kinds[r["run"]] == test_type]
    else:
        rows = [r for r in rows if r["run"] == run]
    if not rows:
        raise SystemExit(f"no rows for run {run} / test type {test_type} in {table}")
    return sorted(rows, key=lambda r: (vin_value(r["vin"]), r["run"], int(r["shot"])))
    # Total: time O(rows log rows), space O(rows)


def save_npz(path, wfs):
    """Per-shot npz in the campaign.py layout."""
    arrs = {}
    for name, w in wfs.items():
        arrs[f"{name}__t"] = np.asarray(w.t, dtype=float)
        arrs[f"{name}__y"] = np.asarray(w.y)
        arrs[f"{name}__unit"] = np.asarray(str(w.unit))
    np.savez_compressed(path, **arrs)
    # Total: time O(samples), space O(samples)


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # Total: time O(rows), space O(1)


def run_identity(yaml_root, run):
    """test_type, expected_peaks, card_type of the run, from its result YAML (safe_load only)."""
    doc = yaml.safe_load((pathlib.Path(yaml_root) / "results" / run / f"{run}.yaml").read_text())
    ident, cond = doc.get("identity") or {}, doc.get("conditions") or {}
    m = re.match(r"^LVR5([AB])-", str(ident.get("board_under_test", "")))
    return {"test_type": ident["test_type"], "card_type": m.group(1) if m else None,
            "expected_peaks": int(cond.get("expected_peaks") or 4), "run": run}
    # Total: time O(YAML size), space O(YAML size)


def columns(rows):
    """Union of the row keys, in first-seen order."""
    seen = {}
    for r in rows:
        seen.update(dict.fromkeys(r))
    return list(seen)
    # Total: time O(rows x keys), space O(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_root")
    ap.add_argument("lvr_tester")
    ap.add_argument("--run", default=None)                      # one run, by its folder name in the archive
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", default=str(OUT / "shots.csv"))  # finder-vs-ML table to take the ML times from
    ap.add_argument("--test-type", default=None)                # all held-out runs of this test type in one folder
    a = ap.parse_args()
    if bool(a.run) == bool(a.test_type):                        # no default run: the choice is always stated
        raise SystemExit("give exactly one of --run and --test-type")
    if a.test_type and not a.out:
        raise SystemExit("--test-type needs --out")
    out = pathlib.Path(a.out) if a.out else pathlib.Path.home() / "Documents" / "vfe_peak_ml" / "replay" / a.run
    camp = out / "campaign_replay"
    (camp / "shots").mkdir(parents=True, exist_ok=True)
    (camp / "ml").mkdir(exist_ok=True)
    # Stale files from an earlier build would mix with the new shot numbering.
    for old in list((camp / "shots").glob("*.npz")) + [camp / n for n in GENERATED]:
        old.unlink(missing_ok=True)
        pathlib.Path(str(old) + ".meta").unlink(missing_ok=True)
    if (camp / "review.csv").exists():
        print(f"WARNING: {camp / 'review.csv'} kept, but its shot_index values may no longer match")

    sys.path.insert(0, str(pathlib.Path(a.lvr_tester).resolve()))
    from lvr_tester.ml.measure import measure_at, rows_from_result
    members = shot_members(a.yaml_root, np.load("data/real_shots.npz"))
    Analyzer, analysis, read_trc = load_finder(a.lvr_tester)

    peak_rows, metric_rows, ml_rows, mlp_rows, mlm_rows, index_rows = [], [], [], [], [], []
    picked = pick_shots(a.table, a.run, a.yaml_root, a.test_type)
    for k, r in enumerate(picked, start=1):
        index_rows.append({"shot_index": k, "run": r["run"], "vin_setting": r["vin"], "shot": r["shot"], "set": r["set"]})
        vin = vin_value(r["vin"])
        c4 = members[int(r["idx"])]
        c1 = c4.replace("/C4--", "/C1--")
        wfs = {"Vin": read_trc(RAW / c1, "Vin", "C1", "V"),
               "Input_current": read_trc(RAW / c4, "Input_current", "C4", "A")}
        res = Analyzer(analysis, wfs).analyze()
        save_npz(camp / "shots" / f"shot_{k:05d}_V{vin}V.npz", wfs)
        f_rows, f_met = rows_from_result(res, meta={"vin_setting": r["vin"]}, shot_index=k)
        peak_rows += f_rows
        metric_rows.append(f_met)
        # ML side: the model's apex times (ms -> s; blank = no answer) through the same measurements
        ml_apex = {lab: float(r[f"{lab}_ml_ms"]) * 1e-3 for lab in LANDMARKS if r[f"{lab}_ml_ms"] != ""}
        m_rows, m_met = measure_at(analysis, wfs, ml_apex, finder_result=res, shot_index=k, vin_setting=r["vin"])
        mlp_rows += m_rows
        mlm_rows.append(m_met)
        for lab in LANDMARKS:
            ml_rows.append({"shot_index": k, "vin_setting": r["vin"], "landmark": lab,
                            "time_ms": r[f"{lab}_ml_ms"], "conf": r[f"{lab}_ml_conf"]})
        print(f"{r['vin']:>4} shot {r['shot']:>3} -> index {k}  " + "  ".join(
            f"{lab} ml {r[f'{lab}_ml_ms']} / fnd {r[f'{lab}_finder_ms'] or '-'}" for lab in LANDMARKS))

    ident = run_identity(a.yaml_root, picked[0]["run"])
    if a.test_type:                                              # many boards in one folder: say so in the identity
        cards = [run_identity(a.yaml_root, r)["card_type"] for r in sorted({x["run"] for x in picked})]
        cards = [c for c in cards if c]                          # a run without a board serial names no card type
        ident["card_type"] = max(set(cards), key=cards.count) if cards else None
        ident["run"] = f"all held-out {a.test_type} runs ({len({r['run'] for r in picked})}), pooled"
    (camp / "replay.yaml").write_text(yaml.safe_dump(ident, sort_keys=False))
    files = {"campaign_peaks.csv": (columns(peak_rows), peak_rows),
             "campaign_shot_metrics.csv": (columns(metric_rows), metric_rows),
             "ml_peaks.csv": (["shot_index", "vin_setting", "landmark", "time_ms", "conf"], ml_rows),
             "ml/campaign_peaks.csv": (columns(peak_rows), mlp_rows),
             "ml/campaign_shot_metrics.csv": (columns(metric_rows), mlm_rows),
             "replay_index.csv": (["shot_index", "run", "vin_setting", "shot", "set"], index_rows)}
    inputs = [a.table, "data/real_shots.npz", str(RAW), a.yaml_root]
    for name, (cols, rows) in files.items():
        write_csv(camp / name, cols, rows)
        write_meta(camp / name, inputs, {"tol_ms": TOL_MS, **ident}, script="diagnostics/make_replay_run.py")
    write_meta(camp / "replay.yaml", inputs, {"run": ident["run"]}, script="diagnostics/make_replay_run.py")
    print(out.resolve())
    # Total: time O(shots x samples), space O(samples)


if __name__ == "__main__":
    main()
