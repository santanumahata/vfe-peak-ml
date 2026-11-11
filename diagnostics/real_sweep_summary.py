"""Table of every real-data model in outputs/real/sweep*/ (metrics.json only).

  python -m diagnostics.real_sweep_summary [glob]       default: outputs/real/sweep*/*/

Columns: parameters, training time, then on the held-out runs -- argmax readout
and sub-bin (parabola) readout -- the share of shots with all five landmarks
within 0.02 ms and 0.05 ms, and the median A1 error in us.
"""
import glob
import json
import sys


def row(path):
    m = json.load(open(path + "metrics.json"))
    out = [path.rstrip("/").split("/")[-1], f"{m['params']/1e6:.2f}M", f"{m['wall_s']:.0f}s",
           str(m.get("seed", 0))]
    for k in ("val", "val_subbin"):
        s = m.get(k)
        if s is None:
            out += ["-", "-", "-"]
            continue
        out += [f"{100*s['all_five_within_0.02ms']:.1f}", f"{100*s['all_five_within_0.05ms']:.1f}",
                f"{1e3*s['per_peak']['P1']['median_ms']:.1f}"]
    return out


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "outputs/real/sweep*/*/"
    hdr = ["model", "params", "train", "seed",
           "argmax<=.02", "argmax<=.05", "argmax A1us",
           "subbin<=.02", "subbin<=.05", "subbin A1us"]
    rows = [row(p) for p in sorted(glob.glob(pat)) if glob.glob(p + "metrics.json")]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(hdr)] if rows else []
    print("  ".join(h.rjust(x) for h, x in zip(hdr, w)))
    for r in rows:
        print("  ".join(c.rjust(x) for c, x in zip(r, w)))


if __name__ == "__main__":
    main()
