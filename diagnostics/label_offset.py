"""How far is each training label from the apex of the noiseless waveform?

The label is the centre t0 of the pulse component. Each main peak also adds a
permanent floor step centred on the same t0, so the summed curve can keep rising
after t0. This measures that displacement, per landmark, on fresh draws.
"""
import numpy as np

from src.toy_waveforms import ToyConfig, make_waveform

LABELS = ["P0", "P1", "P2", "P3", "P4"]
N = 4000
LEFT, RIGHT = 6, 40          # search window around the label, in samples


def clean_apex(clean, pos, k):
    p = pos[k]
    lo = int(max(p - LEFT, 0))
    hi = int(min(p + RIGHT, clean.size - 1))
    if k < 4:                                    # never reach into the next landmark
        hi = int(min(hi, (pos[k] + pos[k + 1]) / 2))
    j = lo + int(np.argmax(clean[lo:hi + 1]))
    return j, j >= hi                           # apex, and "no apex inside the window"


def main():
    cfg = ToyConfig()
    rng = np.random.default_rng(2024)
    off = np.zeros((N, 5)); edge = np.zeros((N, 5), bool)
    for i in range(N):
        _, pos, clean = make_waveform(rng, cfg, return_clean=True)
        for k in range(5):
            j, e = clean_apex(clean, pos, k)
            off[i, k] = j - pos[k]; edge[i, k] = e
    print(f"{N} draws; offset = clean apex - label (samples; + means apex is later)")
    for k in range(5):
        o = off[:, k]
        print(f"{LABELS[k]}: median {np.median(o):+5.1f}  p90 {np.percentile(o,90):+5.1f}  "
              f"max {o.max():+5.1f}  |off|>2 on {(np.abs(o)>2).mean():.3f}  "
              f"no apex in window {edge[:,k].mean():.3f}")


if __name__ == "__main__":
    main()
