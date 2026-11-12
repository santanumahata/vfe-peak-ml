# Purpose: hand-judge every ML-vs-finder disagreement on the full-resolution trace; one verdict per landmark
# Created: 2026-11-12
# Class:   diagnostic
# Inputs:  outputs/real/finder_vs_ml/shots.csv, data/real_shots.npz, data/raw/ (C4 .trc), <yaml_root>
# Outputs: outputs/real/finder_vs_ml/judgments.csv (+ .meta)
"""Click through the disagreements, one landmark at a time.

  python -m diagnostics.judge_disagreements <yaml_root>

Keys:  m = ML right   f = finder right   b = both fine   n = neither (then click the true position)
       u = undo last   s = skip   q = quit (progress is saved after every verdict; rerun to resume)
"""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("QtAgg")
for _km in ("keymap.fullscreen", "keymap.save", "keymap.quit", "keymap.back"):    # free f, s, q, c for verdicts
    matplotlib.rcParams[_km] = []
import matplotlib.pyplot as plt
import numpy as np

from diagnostics.finder_vs_ml import OUT, RAW, shot_members, write_meta
from src.trc import read_trc

SHOTS = OUT / "shots.csv"
JUDG = OUT / "judgments.csv"
FIELDS = ["set", "idx", "run", "vin", "shot", "landmark", "ml_ms", "ml_conf", "finder_ms", "verdict", "true_ms"]
KEYS = {"m": "ml", "f": "finder", "b": "both", "n": "neither", "s": "skip"}


def items_to_judge():
    """One item per (shot, landmark) flagged as disagreeing."""
    out = []
    for r in csv.DictReader(open(SHOTS)):
        for lab in ("P0", "P1", "P2", "P3", "P4"):
            if r[f"{lab}_disagree"] == "1":
                out.append({"set": r["set"], "idx": int(r["idx"]), "run": r["run"], "vin": r["vin"],
                            "shot": r["shot"], "landmark": lab, "ml_ms": float(r[f"{lab}_ml_ms"]),
                            "ml_conf": r[f"{lab}_ml_conf"], "finder_ms": r[f"{lab}_finder_ms"],
                            "a1_ms": r["P1_finder_ms"]})
    return out
    # Total: time O(shots), space O(items)


def load_done():
    if not JUDG.exists():
        return []
    return list(csv.DictReader(open(JUDG)))
    # Total: time O(items), space O(items)


def save(done):
    with open(JUDG, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(done)
    # Total: time O(items), space O(1)


def zoom_window(it):
    """P0: from 0.8 ms before the earlier answer to 0.2 ms past A1. A1..B2: centred, wide enough to see both."""
    ml = it["ml_ms"]
    fd = float(it["finder_ms"]) if it["finder_ms"] else ml
    if it["landmark"] == "P0" and it["a1_ms"]:
        return min(ml, fd) - 0.8, max(ml, fd, float(it["a1_ms"])) + 0.2
    half = max(3 * abs(ml - fd), 0.1)
    mid = 0.5 * (ml + fd)
    return mid - half, mid + half
    # Total: time O(1), space O(1)


def draw(fig, it, t_ms, y, n_left):
    fig.clf()
    ax_all, ax = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 3]})
    lo, hi = zoom_window(it)
    ax_all.plot(t_ms, y, lw=0.5, color="0.3")
    ax_all.axvspan(lo, hi, color="gold", alpha=0.3)
    sel = (t_ms >= lo) & (t_ms <= hi)
    ax.plot(t_ms[sel], y[sel], lw=0.8, color="0.2", marker=".", ms=1.5)
    ax.axvline(it["ml_ms"], color="tab:blue", lw=1.5, label=f"ML  {it['ml_ms']:.4f} ms  conf {it['ml_conf']}")
    if it["finder_ms"]:
        fd = float(it["finder_ms"])
        ax.axvline(fd, color="tab:orange", lw=1.5, ls="--", label=f"finder  {fd:.4f} ms")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("input current [A]")
    ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"{it['landmark']}  |  {it['run']}  {it['vin']}  shot {it['shot']}  [{it['set']}]   "
                 f"{n_left} left   —   m ML · f finder · b both · n neither+click · u undo · s skip · q quit",
                 fontsize=9)
    fig.canvas.draw_idle()
    # Total: time O(samples), space O(samples)


def main():
    yaml_root = sys.argv[1]
    d = np.load("data/real_shots.npz")
    members = shot_members(yaml_root, d)
    todo = items_to_judge()
    done = load_done()
    key_of = lambda r: (str(r["idx"]), r["landmark"])
    seen = {key_of(r) for r in done if r["verdict"] != "skip"}
    queue = [it for it in todo if key_of(it) not in seen]
    print(f"{len(todo)} items, {len(todo) - len(queue)} already judged, {len(queue)} to go", flush=True)
    if not queue:
        return

    fig = plt.figure(figsize=(13, 7))
    state = {"i": 0, "await_click": False, "trace": {}}

    def trace(it):
        if it["idx"] not in state["trace"]:
            t, v, _ = read_trc(RAW / members[it["idx"]])
            state["trace"] = {it["idx"]: (t * 1e3, v)}          # keep one trace in memory
        return state["trace"][it["idx"]]

    def show():
        if state["i"] >= len(queue):
            plt.close(fig)
            return
        it = queue[state["i"]]
        draw(fig, it, *trace(it), len(queue) - state["i"])

    def record(verdict, true_ms=""):
        it = queue[state["i"]]
        row = {k: it[k] for k in FIELDS if k in it}
        row.update({"verdict": verdict, "true_ms": true_ms})
        done[:] = [r for r in done if key_of(r) != key_of(it)] + [row]
        save(done)
        state["i"] += 1
        show()

    def on_key(ev):
        if state["await_click"]:
            return
        if ev.key == "q":
            plt.close(fig)
        elif ev.key == "u" and state["i"] > 0:
            state["i"] -= 1
            prev = queue[state["i"]]
            done[:] = [r for r in done if key_of(r) != key_of(prev)]
            save(done)
            show()
        elif ev.key == "n":
            state["await_click"] = True
            fig.suptitle("click the true position in the lower panel", color="red")
            fig.canvas.draw_idle()
        elif ev.key in KEYS:
            record(KEYS[ev.key])

    def on_click(ev):
        if state["await_click"] and ev.inaxes is not None and ev.xdata is not None:
            state["await_click"] = False
            record("neither", f"{ev.xdata:.4f}")

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    show()
    plt.show()

    write_meta(JUDG, [str(SHOTS), "data/real_shots.npz", str(RAW), yaml_root], {"tool": "hand judgement"})
    counts = {}
    for r in done:
        counts.setdefault(r["landmark"], {}).setdefault(r["verdict"], 0)
        counts[r["landmark"]][r["verdict"]] += 1
    print(f"judged {len(done)} / {len(todo)}")
    for lab in sorted(counts):
        print(lab, counts[lab])
    print(JUDG.resolve())
    # Total: time O(items x samples), space O(samples)


if __name__ == "__main__":
    main()
