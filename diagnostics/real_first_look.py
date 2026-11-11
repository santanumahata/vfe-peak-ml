"""First look at real VFE/LVR current traces, with the existing finder's marks.

Usage: python -m diagnostics.real_first_look <yaml_root> <results.tar> <extract_dir> <out_png> <members.txt>
(<members.txt> = output of `tar -tf results.tar`; only shots whose trace is in the archive are used.)

Picks, from the YAMLs: a few shots where the finder found all 4 peaks, and the shots
where it found only 3. Extracts just those C4 (current) traces from the archive into
<extract_dir> (an isolated folder), reads them with src.trc, and draws each one with
the finder's peaks: apex (time_ms) as a marker, [start_ms, end_ms] shaded.
"""
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.trc import read_trc

COL = {"A1": "#00BCEB", "B1": "#FFB900", "A2": "#7FD6F2", "B2": "#FFD976",
       "P1": "#FF5A5A", "P2": "#FF5A5A", "P3": "#FF5A5A"}


def pick_shots(yaml_root, have):
    good, bad = [], []
    for f in sorted(pathlib.Path(yaml_root).rglob("*.yaml")):
        d = yaml.safe_load(f.read_text())
        if not isinstance(d, dict) or (d.get("conditions") or {}).get("expected_peaks") != 4:
            continue
        for s in (d.get("results") or {}).get("per_shot") or []:
            c4 = [t for t in s.get("trc") or [] if t.startswith("C4")]
            if not c4 or f"results/{f.parent.name}/trc/{c4[0]}" not in have:
                continue
            rec = (f.parent.name, s, c4[0])
            (bad if s.get("n_peaks") == 3 else good).append(rec)
    # good: same runs as the failures where possible, so the comparison is like for like
    bad_runs = {r for r, _, _ in bad}
    same = [g for g in good if g[0] in bad_runs]
    other = [g for g in good if g[0] not in bad_runs]
    return same[:2] + other[:2], bad[:4]


def main():
    yaml_root, tar, ext, out, mem = sys.argv[1:6]
    have = set(pathlib.Path(mem).read_text().split("\n"))
    good, bad = pick_shots(yaml_root, have)
    members = [f"results/{r}/trc/{c}" for r, _, c in good + bad]
    pathlib.Path(ext).mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", tar, "-C", ext, "--no-same-owner",
                    "--no-same-permissions", *members], check=True)

    rows = good + bad
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.1 * len(rows)), sharex=True)
    for ax, (run, s, c) in zip(axes, rows):
        t, v, _ = read_trc(f"{ext}/results/{run}/trc/{c}")
        tm = t * 1e3
        ax.plot(tm, v, lw=0.5, color="#1f3b73")
        for p in s.get("peaks") or []:
            col = COL.get(p["label"], "k")
            ax.axvspan(p["start_ms"], p["end_ms"], color=col, alpha=0.25)
            ax.plot(p["time_ms"], p["amplitude_A"], "v", color=col, ms=7)
            ax.text(p["time_ms"], p["amplitude_A"], " " + p["label"], color=col,
                    fontsize=8, va="bottom")
        tag = "FOUND 3" if s.get("n_peaks") == 3 else "found 4"
        ax.set_title(f"{tag}  |  {run}  {s['vin']} shot {s['shot_in_point']}",
                     fontsize=8, loc="left")
        ax.set_ylabel("I [A]", fontsize=8)
    axes[-1].set_xlabel("time from trigger [ms]")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
