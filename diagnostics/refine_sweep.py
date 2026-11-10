"""Sweep the snap window and the gain guard; report the NET effect, not just the
cases that improve. Writes outputs/refine_sweep.json."""
import json

import numpy as np
import torch

from src.model import PeakUNet
from src.refine import refine_batch
from src.toy_waveforms import ToyConfig, make_dataset
from src.train import normalise, N_VAL, SIGMA, GROSS_MISS

LABELS = ["P0", "P1", "P2", "P3", "P4"]
WINDOWS = [3, 5, 8]
GAINS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
SMOOTHS = [1, 3]


def score(dt):
    return {
        "all_five_within_2": float((np.abs(dt) <= 2).all(1).mean()),
        "gross_miss_any": float((np.abs(dt) > GROSS_MISS).any(1).mean()),
        "per_peak": {
            LABELS[k]: {
                "median_abs_dt": float(np.median(np.abs(dt[:, k]))),
                "frac_within_2": float((np.abs(dt[:, k]) <= 2).mean()),
                "gross_miss_rate": float((np.abs(dt[:, k]) > GROSS_MISS).mean()),
            } for k in range(5)},
    }


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

    base = score(pos - Pv)
    print(f"baseline           all5={base['all_five_within_2']:.3f} "
          f"gross={base['gross_miss_any']:.4f}")

    rows = []
    for sm in SMOOTHS:
        for w in WINDOWS:
            for g in GAINS:
                P, I, A, S = refine_batch(Xn, pos, conf, window=w,
                                          min_gain=g, smooth=sm)
                s = score(P - Pv)
                s.update(window=w, min_gain=g, smooth=sm,
                         snap_rate=float(S.mean()))
                rows.append(s)
                print(f"smooth={sm} w={w} gain={g:<5} all5={s['all_five_within_2']:.3f} "
                      f"gross={s['gross_miss_any']:.4f} snapped={S.mean():.3f} "
                      f"P2_w2={s['per_peak']['P2']['frac_within_2']:.3f}")

    best = max(rows, key=lambda r: (r["all_five_within_2"], -r["gross_miss_any"]))
    print("\nbest:", {k: best[k] for k in ("smooth", "window", "min_gain",
                                           "all_five_within_2", "gross_miss_any",
                                           "snap_rate")})
    out = {"baseline": base, "sweep": rows, "best": best, "n_val": int(len(Xv))}
    with open("outputs/refine_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("outputs/refine_sweep.json")


if __name__ == "__main__":
    main()
