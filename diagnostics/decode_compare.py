"""Compare ways of reading a position off the heatmap, for one or more models.

  argmax   : the sample where the channel is largest (what everything uses now)
  parabola : fit a parabola through the argmax and its two neighbours
  com<k>   : centre of mass of the channel within +-k samples of the argmax

Usage: python -m diagnostics.decode_compare <model_dir> [<model_dir> ...]
Each <model_dir> holds peak_unet.pt (outputs/ or a sweep folder).
"""
import sys

import numpy as np
import torch

from src.model import PeakUNet
from src.toy_waveforms import ToyConfig, make_dataset
from src.train import normalise, N_VAL, GROSS_MISS


def heatmaps(model_dir, Xn):
    net = PeakUNet()
    net.load_state_dict(torch.load(f"{model_dir}/peak_unet.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(net(torch.from_numpy(Xn[i:i + 64]))).numpy()
                               for i in range(0, len(Xn), 64)])


def decode(prob, how):
    N, K, n = prob.shape
    am = prob.argmax(-1)
    if how == "argmax":
        return am.astype(float)
    idx = np.arange(n)
    out = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            p, a = prob[i, k], am[i, k]
            if how == "parabola":
                if 0 < a < n - 1:
                    l, c, r = p[a - 1], p[a], p[a + 1]
                    den = l - 2 * c + r
                    off = 0.5 * (l - r) / den if den != 0 else 0.0
                    out[i, k] = a + np.clip(off, -1, 1)
                else:
                    out[i, k] = a
            else:                                   # com<k>
                w = int(how[3:])
                lo, hi = max(a - w, 0), min(a + w + 1, n)
                ww = p[lo:hi]
                out[i, k] = (ww * idx[lo:hi]).sum() / max(ww.sum(), 1e-9)
    return out


def main():
    dirs = sys.argv[1:] or ["outputs"]
    cfg = ToyConfig()
    Xv, _, Pv = make_dataset(N_VAL, seed=1, cfg=cfg)
    Xn = normalise(Xv)
    for d in dirs:
        prob = heatmaps(d, Xn)
        print(f"--- {d}")
        for how in ("argmax", "parabola", "com2", "com4", "com8"):
            pos = decode(prob, how)
            for rnd in (False, True):
                e = np.abs((np.round(pos) if rnd else pos) - Pv)
                tag = how + (" rounded" if rnd else "")
                print(f"  {tag:17s} all5<=2 {(e <= 2).all(1).mean():.3f}  "
                      f"gross {(e > GROSS_MISS).any(1).mean():.3f}  "
                      f"per-peak<=2 {' '.join('%.3f' % v for v in (e <= 2).mean(0))}")
                if how == "argmax":
                    break


if __name__ == "__main__":
    main()
