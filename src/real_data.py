"""Build the real-board training set from the LVR tester archive.

Each 4-peak shot gives one input-current trace (scope channel C4) and the existing
finder's apex times for A1, B1, A2, B2. Five landmarks are learned:
  P0 = inrush spike (not labelled by the finder: highest point before P0_WINDOW_MS[1])
  P1..P4 = A1, B1, A2, B2  (finder's time_ms = sample maximum of the raw trace)

Traces are averaged into BIN-sample bins (10 us at the stored 200 ns spacing) and
padded to N_BINS, so a landmark position is a bin index. Usage:

  python -m src.real_data <yaml_root> <results.tar> <members.txt> <data_dir>

writes <data_dir>/real_shots.npz with X (N, 1, N_BINS) raw amps, P (N, 5) bin
positions (NaN where unknown), per-shot run / vin / shot / n_found / good fields, and
the finder's own output on every shot (finder_t_ms, finder_lab), including failures.
"""
import pathlib                                                   # - | paths
import subprocess                                                # - | tar extraction
import sys                                                       # - | argv

import numpy as np                                               # - | arrays
import yaml                                                      # - | result files (safe_load only)

from src.trc import read_trc                                     # - | LeCroy reader

BIN = 50                                                         # - | stored samples per bin: 50 x 200 ns = 10 us
N_BINS = 5120                                                    # - | 5000 bins of trace, padded for the U-Net
P0_WINDOW_MS = (-5.0, 3.0)                                       # - | inrush spike lives before Vin settles
FINDER_LABELS = ("A1", "B1", "A2", "B2")                         # - | P1..P4 in time order


def collect(yaml_root, have):                                    # - | every shot with a trace on disk
    """Return a list of shot records from 4-peak runs whose C4 trace is in the archive."""
    out = []                                                     # - | records
    for f in sorted(pathlib.Path(yaml_root).rglob("*.yaml")):    # O(F) | one result file per run
        d = yaml.safe_load(f.read_text())                        # O(size) | data only, no tags
        if not isinstance(d, dict):                              # - | not a result file
            continue                                             # - | skip
        if (d.get("conditions") or {}).get("expected_peaks") != 4:  # - | only the 4-peak test
            continue                                             # - | skip 2-peak runs
        run = f.parent.name                                      # - | run id
        for s in (d.get("results") or {}).get("per_shot") or []:  # O(S) | shots in the run
            c4 = [t for t in s.get("trc") or [] if t.startswith("C4")]  # - | current channel
            member = f"results/{run}/trc/{c4[0]}" if c4 else None  # - | path inside the tar
            if member is None or member not in have:             # - | trace not kept
                continue                                         # - | skip
            peaks = {p["label"]: p for p in s.get("peaks") or []}  # - | by label
            good = all(k in peaks for k in FINDER_LABELS)        # - | finder found all four
            out.append({"run": run, "vin": s.get("vin"), "shot": s.get("shot"),
                        "member": member, "n_found": s.get("n_peaks"), "good": good,
                        "t_ms": [peaks[k]["time_ms"] if good else np.nan
                                 for k in FINDER_LABELS],          # - | finder apex times
                        "found_t_ms": [p["time_ms"] for p in s.get("peaks") or []],  # - | whatever it reported
                        "found_lab": [p["label"] for p in s.get("peaks") or []]})  # - | and what it called them
    return out
    # Total: time O(total YAML size), space O(shots)


def to_bins(t_s, v):                                             # - | 200 ns samples -> 10 us bins
    """Average into BIN-sample bins and pad to N_BINS; return (t0_ms, dt_ms, binned)."""
    n = (v.size // BIN) * BIN                                    # - | drop the ragged tail
    b = v[:n].reshape(-1, BIN).mean(axis=1)                      # O(n) | anti-aliased downsample
    pad = N_BINS - b.size                                        # - | room left
    if pad < 0:                                                  # - | longer record than expected
        raise ValueError(f"{b.size} bins > N_BINS={N_BINS}")     # - | refuse to truncate silently
    b = np.pad(b, (0, pad), mode="edge")                         # O(N_BINS) | hold the last value
    dt_ms = (t_s[1] - t_s[0]) * BIN * 1e3                        # - | bin width
    t0_ms = (t_s[0] + 0.5 * (t_s[BIN - 1] - t_s[0])) * 1e3       # - | centre of bin 0
    return t0_ms, dt_ms, b
    # Total: time O(n), space O(n)


def main():                                                      # - | build the npz
    yaml_root, tar, members, data_dir = sys.argv[1:5]            # - | inputs
    have = set(pathlib.Path(members).read_text().split("\n"))    # O(M) | archive listing
    recs = collect(yaml_root, have)                              # O(F) | shot records
    raw_dir = pathlib.Path(data_dir) / "raw"                     # - | isolated extraction folder
    raw_dir.mkdir(parents=True, exist_ok=True)                   # - | make it
    todo = [r["member"] for r in recs if not (raw_dir / r["member"]).exists()]  # - | not yet extracted
    if todo:                                                     # - | something to fetch
        lst = pathlib.Path(data_dir) / "members_to_extract.txt"  # - | tar -T list
        lst.write_text("\n".join(todo) + "\n")                   # - | one per line
        subprocess.run(["tar", "-xf", tar, "-C", str(raw_dir), "--no-same-owner",
                        "--no-same-permissions", "-T", str(lst)], check=True)  # O(archive) | one pass

    N = len(recs)                                                # - | shot count
    X = np.zeros((N, 1, N_BINS), np.float32)                     # - | binned traces in amps
    P = np.full((N, 5), np.nan, np.float32)                      # - | landmark bins
    T0 = np.zeros(N); DT = np.zeros(N)                           # - | time axis per shot
    for i, r in enumerate(recs):                                 # O(N) | one trace at a time
        t, v, _ = read_trc(raw_dir / r["member"])                # O(n) | 250k samples
        t0, dt, b = to_bins(t, v)                                # O(n) | 5120 bins
        X[i, 0] = b; T0[i] = t0; DT[i] = dt                      # - | store
        tm = t0 + dt * np.arange(N_BINS)                         # - | bin centres in ms
        w = (tm >= P0_WINDOW_MS[0]) & (tm <= P0_WINDOW_MS[1])    # - | inrush window
        P[i, 0] = np.flatnonzero(w)[np.argmax(b[w])]             # O(N_BINS) | P0 by rule
        if r["good"]:                                            # - | finder labels exist
            P[i, 1:] = (np.array(r["t_ms"]) - t0) / dt           # - | ms -> bin index (fractional)
        if i % 500 == 0:                                         # - | progress
            print(f"{i}/{N}", flush=True)

    out = pathlib.Path(data_dir) / "real_shots.npz"              # - | the dataset
    np.savez_compressed(out, X=X, P=P, t0_ms=T0, dt_ms=DT,
                        run=np.array([r["run"] for r in recs]),
                        vin=np.array([str(r["vin"]) for r in recs]),
                        shot=np.array([r["shot"] for r in recs]),
                        n_found=np.array([r["n_found"] for r in recs]),
                        good=np.array([r["good"] for r in recs]),
                        finder_t_ms=np.array([(r["found_t_ms"] + [np.nan] * 4)[:4] for r in recs]),
                        finder_lab=np.array([",".join(r["found_lab"]) for r in recs]))  # O(N N_BINS) | write
    print(f"{out}  shots={N}  good={sum(r['good'] for r in recs)}  "
          f"runs={len({r['run'] for r in recs})}")
    # Total: time O(N n), space O(N N_BINS)


if __name__ == "__main__":                                       # - | script entry
    main()                                                       # - | run it
