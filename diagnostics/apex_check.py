"""Is the network landing beside a peak that is plainly there?

For every held-out trace, compare the network's position with the nearest local
maximum of the trace, and with the true landmark. Reports, per landmark:
  - how often a clear local maximum sits within W samples of the truth,
  - how often the network is further from the truth than that maximum is,
  - the same restricted to the cases the network currently gets wrong.
Prints nothing but numbers; writes no files.
"""
import numpy as np
import torch
from scipy.signal import find_peaks

from src.model import PeakUNet
from src.toy_waveforms import ToyConfig, make_dataset
from src.train import normalise, N_VAL, SIGMA

LABELS = ["P0", "P1", "P2", "P3", "P4"]
WINDOWS = [3, 5, 8, 12]


def main():
    cfg = ToyConfig()
    Xv, _, Pv = make_dataset(N_VAL, seed=1, cfg=cfg, sigma=SIGMA)
    Xn = normalise(Xv)
    net = PeakUNet()
    net.load_state_dict(torch.load("outputs/peak_unet.pt", map_location="cpu"))
    net.eval()

    pos = np.zeros((len(Xv), 5))
    with torch.no_grad():
        for i in range(0, len(Xv), 64):
            p = torch.sigmoid(net(torch.from_numpy(Xn[i:i + 64])))
            pos[i:i + 64] = p.argmax(dim=-1).numpy()

    dt = np.abs(pos - Pv)
    print(f"baseline  all-five-within-2 = {(dt <= 2).all(1).mean():.3f}   "
          f"gross = {(dt > 15).any(1).mean():.3f}")

    # local maxima of the raw trace, and of a 3-sample boxcar version
    sm = np.apply_along_axis(
        lambda v: np.convolve(v, np.ones(3) / 3.0, mode="same"), 1, Xn[:, 0])

    for tag, S in (("raw", Xn[:, 0]), ("smoothed", sm)):
        print(f"\n--- local maxima of the {tag} trace ---")
        for k in range(5):
            near_truth = {w: 0 for w in WINDOWS}
            net_worse = {w: 0 for w in WINDOWS}
            n_bad = 0
            bad_rescued = {w: 0 for w in WINDOWS}
            for i in range(len(Xv)):
                mx, _ = find_peaks(S[i])
                if mx.size == 0:
                    continue
                truth = Pv[i, k]
                d_truth = np.abs(mx - truth)
                j = int(d_truth.argmin())
                d_net = dt[i, k]
                bad = d_net > 2
                n_bad += bad
                for w in WINDOWS:
                    # is the network's own answer already at a local maximum?
                    if np.abs(mx - pos[i, k]).min() <= w:
                        continue_flag = True
                    else:
                        continue_flag = False
                    if d_truth[j] <= w:
                        near_truth[w] += 1
                        # nearest local max to the NETWORK position, within w
                        cand = mx[np.abs(mx - pos[i, k]) <= w]
                        if cand.size:
                            snapped = cand[np.argmax(S[i][cand])]
                            if np.abs(snapped - truth) < d_net - 1e-9:
                                net_worse[w] += 1
                                if bad and np.abs(snapped - truth) <= 2:
                                    bad_rescued[w] += 1
                    _ = continue_flag
            print(f"{LABELS[k]}  net wrong (>2) on {n_bad:4d} traces")
            for w in WINDOWS:
                print(f"    w={w:2d}: local max within w of truth "
                      f"{near_truth[w]/len(Xv):.3f} | snapping would improve "
                      f"{net_worse[w]/len(Xv):.3f} | would rescue "
                      f"{bad_rescued[w]:3d}/{n_bad}")


if __name__ == "__main__":
    main()
