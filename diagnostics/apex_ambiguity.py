"""Is +-2 samples on the apex reachable at this noise level?

Two model-free checks on the 1000 held-out traces, both told roughly where each
peak is (so they never confuse one peak for another -- an advantage the network
does not get):

1. Oracle readers: take the highest sample of the noisy trace within +-6 of the
   true apex, raw or after a short running mean. How often are they within 2?
2. Flatness: how often does the noiseless waveform stay within one noise sigma
   of the apex height somewhere 3 or more samples away? There, a single-sample
   reading cannot tell the apex from its neighbour.

Neither is a strict bound -- a reader that knows the pulse shape can average over
many samples -- so they say whether the ceiling is plausibly noise, not prove it.
"""
import json

import numpy as np

from src.toy_waveforms import ToyConfig, make_waveform
from src.train import N_VAL

LABELS = ["P0", "P1", "P2", "P3", "P4"]
HALF = 6


def main():
    cfg = ToyConfig()
    rng = np.random.default_rng(1)                  # same stream as the val split
    X = np.zeros((N_VAL, cfg.n_samples)); C = np.zeros_like(X); P = np.zeros((N_VAL, 5))
    for i in range(N_VAL):
        X[i], P[i], C[i] = make_waveform(rng, cfg, return_clean=True)

    noise = X - C
    sig = 1.4826 * np.median(np.abs(noise - np.median(noise, 1, keepdims=True)), 1)  # robust sigma

    res = {"oracle": {}}
    for w in (1, 3, 5):
        S = X if w == 1 else np.apply_along_axis(
            lambda v: np.convolve(v, np.ones(w) / w, mode="same"), 1, X)
        e = np.zeros((N_VAL, 5))
        for i in range(N_VAL):
            for k in range(5):
                a = int(P[i, k]); lo, hi = max(a - HALF, 0), min(a + HALF + 1, X.shape[1])
                e[i, k] = abs(lo + int(np.argmax(S[i, lo:hi])) - a)
        tag = "raw noisy trace" if w == 1 else f"running mean of {w}"
        res["oracle"][str(w)] = {"all_five_within_2": float((e <= 2).all(1).mean()),
                                 "per_peak_within_2": (e <= 2).mean(0).tolist()}
        print(f"oracle reader, {tag:17s}: all5<=2 {(e <= 2).all(1).mean():.3f}   "
              f"per-peak {' '.join('%.3f' % v for v in (e <= 2).mean(0))}")

    flat = np.zeros((N_VAL, 5), bool)
    for i in range(N_VAL):
        for k in range(5):
            a = int(P[i, k]); h = C[i, a]
            far = np.r_[max(a - HALF, 0):max(a - 2, 0), min(a + 3, C.shape[1]):min(a + HALF + 1, C.shape[1])]
            flat[i, k] = far.size > 0 and (h - C[i, far]).min() < sig[i]
    print(f"\napex flat to within 1 noise sigma at >=3 samples: "
          f"{' '.join('%s %.3f' % (LABELS[k], flat[:, k].mean()) for k in range(5))}")
    print(f"traces where at least one of the five is that flat: {flat.any(1).mean():.3f}")
    print(f"noise sigma (normalised units) median {np.median(sig):.4f}")
    res["flat_per_peak"] = flat.mean(0).tolist()
    res["flat_any"] = float(flat.any(1).mean())
    res["noise_sigma_median"] = float(np.median(sig))
    with open("outputs/apex_ambiguity.json", "w") as f:
        json.dump(res, f, indent=2)
    print("outputs/apex_ambiguity.json")


if __name__ == "__main__":
    main()
