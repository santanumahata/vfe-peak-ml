"""Are the new P1-P4 misses real, or the network picking another point on a flat top?

For each >2-sample miss, compare the noiseless waveform at the network's pick with
the noiseless apex height. If the pick sits within a small fraction of the apex,
the two are indistinguishable by eye and by any finder: the label is ambiguous,
not the network wrong. Also reports how wide each apex top is.
"""
import numpy as np
import torch

from src.model import PeakUNet
from src.toy_waveforms import ToyConfig, make_waveform
from src.train import normalise, N_VAL

LABELS = ["P0", "P1", "P2", "P3", "P4"]


def main():
    cfg = ToyConfig()
    rng = np.random.default_rng(1)                  # same stream as make_dataset(seed=1)
    X = np.zeros((N_VAL, 1, cfg.n_samples), np.float32)
    C = np.zeros((N_VAL, cfg.n_samples), np.float32)
    P = np.zeros((N_VAL, 5), np.float32)
    for i in range(N_VAL):
        X[i, 0], P[i], C[i] = make_waveform(rng, cfg, return_clean=True)

    net = PeakUNet()
    net.load_state_dict(torch.load("outputs/peak_unet.pt", map_location="cpu"))
    net.eval()
    Xn = normalise(X)
    with torch.no_grad():
        pos = np.concatenate([torch.sigmoid(net(torch.from_numpy(Xn[i:i + 64])))
                              .argmax(-1).numpy() for i in range(0, N_VAL, 64)])
    dt = np.abs(pos - P)

    for k in range(5):
        top_w, rel = [], []
        for i in range(N_VAL):
            c = C[i]; a = int(P[i, k]); h = c[a]
            base = np.median(c)                       # a scale for "small"
            span = max(h - base, 1e-6)
            # width of the top: samples within 1% of the apex height, contiguous
            lo = a
            while lo > 0 and h - c[lo - 1] < 0.01 * span:
                lo -= 1
            hi = a
            while hi < c.size - 1 and h - c[hi + 1] < 0.01 * span:
                hi += 1
            top_w.append(hi - lo + 1)
            if dt[i, k] > 2:
                rel.append((h - c[int(pos[i, k])]) / span)
        top_w = np.array(top_w); rel = np.array(rel)
        flat = (rel < 0.01).sum() if rel.size else 0
        print(f"{LABELS[k]}: top width (within 1% of apex) median {np.median(top_w):.0f}, "
              f"p90 {np.percentile(top_w, 90):.0f} samples | misses >2: {rel.size:3d}, "
              f"of which on the flat top (<1% below apex): {flat:3d}")


if __name__ == "__main__":
    main()
