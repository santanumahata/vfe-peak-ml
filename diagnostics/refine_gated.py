"""Confidence-gated snap, and the same numbers under the other definition of
truth (sample maximum rather than pulse centre). Writes outputs/refine_gated.json.
"""
import json

import numpy as np
import torch
from scipy.signal import find_peaks

from src.model import PeakUNet
from src.refine import refine_batch
from src.toy_waveforms import ToyConfig, make_dataset
from src.train import normalise, N_VAL, SIGMA, GROSS_MISS

LABELS = ["P0", "P1", "P2", "P3", "P4"]


def score(dt):
    return {"all_five_within_2": float((np.abs(dt) <= 2).all(1).mean()),
            "gross_miss_any": float((np.abs(dt) > GROSS_MISS).any(1).mean()),
            "per_peak": {LABELS[k]: {
                "median_abs_dt": float(np.median(np.abs(dt[:, k]))),
                "frac_within_2": float((np.abs(dt[:, k]) <= 2).mean()),
                "gross_miss_rate": float((np.abs(dt[:, k]) > GROSS_MISS).mean()),
            } for k in range(5)}}


def apex_truth(Xn, Pv, half=6):
    """Truth redefined as the highest sample within +-half of the pulse centre."""
    out = np.zeros_like(Pv)
    for i in range(len(Xn)):
        s = Xn[i, 0]
        for k in range(5):
            p = int(round(Pv[i, k]))
            lo, hi = max(p - half, 0), min(p + half + 1, s.size)
            out[i, k] = lo + int(np.argmax(s[lo:hi]))
    return out


def main():
    cfg = ToyConfig()
    Xv, _, Pv = make_dataset(N_VAL, seed=1, cfg=cfg, sigma=SIGMA)
    Xn = normalise(Xv)
    net = PeakUNet()
    net.load_state_dict(torch.load("outputs/peak_unet.pt", map_location="cpu"))
    net.eval()
    pos = np.zeros((len(Xv), 5)); conf = np.zeros((len(Xv), 5))
    with torch.no_grad():
        for i in range(0, len(Xv), 64):
            p = torch.sigmoid(net(torch.from_numpy(Xn[i:i + 64])))
            c, a = p.max(dim=-1)
            pos[i:i + 64] = a.numpy(); conf[i:i + 64] = c.numpy()

    res = {"n_val": int(len(Xv)), "centre_truth": {}, "apex_truth": {}}
    res["centre_truth"]["network_only"] = score(pos - Pv)
    print(f"centre truth  network only   all5={res['centre_truth']['network_only']['all_five_within_2']:.3f} "
          f"gross={res['centre_truth']['network_only']['gross_miss_any']:.4f}")

    grid = [(3, 0.05, 0.5), (3, 0.05, 0.8), (3, 0.05, 0.9), (3, 0.10, 0.9),
            (5, 0.05, 0.5), (5, 0.05, 0.8), (5, 0.10, 0.8), (5, 0.10, 0.9),
            (8, 0.10, 0.5), (8, 0.10, 0.8)]
    rows = []
    for w, g, c in grid:
        P, I, A, S = refine_batch(Xn, pos, conf, window=w, min_gain=g,
                                  smooth=1, max_conf=c)
        s = score(P - Pv)
        s.update(window=w, min_gain=g, max_conf=c, snap_rate=float(S.mean()))
        rows.append(s)
        print(f"  gated w={w} gain={g} conf<{c}: all5={s['all_five_within_2']:.3f} "
              f"gross={s['gross_miss_any']:.4f} snapped={S.mean():.4f}")
    res["centre_truth"]["gated"] = rows

    # ---- the other definition of truth -------------------------------------
    Pa = apex_truth(Xn, Pv)
    print(f"\napex truth differs from centre truth by median "
          f"{np.median(np.abs(Pa - Pv)):.2f} samples")
    res["apex_truth"]["network_only"] = score(pos - Pa)
    print(f"apex truth    network only   all5={res['apex_truth']['network_only']['all_five_within_2']:.3f}")
    P, I, A, S = refine_batch(Xn, pos, conf, window=5, min_gain=0.0, smooth=1,
                              max_conf=1.01)
    res["apex_truth"]["always_snap_w5"] = score(P - Pa)
    res["apex_truth"]["always_snap_w5"]["snap_rate"] = float(S.mean())
    print(f"apex truth    always snap w5 all5={res['apex_truth']['always_snap_w5']['all_five_within_2']:.3f}")
    res["centre_truth"]["always_snap_w5"] = score(P - Pv)
    print(f"centre truth  always snap w5 all5={res['centre_truth']['always_snap_w5']['all_five_within_2']:.3f}")

    with open("outputs/refine_gated.json", "w") as f:
        json.dump(res, f, indent=2)
    print("outputs/refine_gated.json")


if __name__ == "__main__":
    main()
