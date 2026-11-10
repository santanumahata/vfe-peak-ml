"""Why does P0 fail on the new generator?

For every trace where the P0 channel's argmax is more than 2 samples from truth,
ask three things:
  1. where did it land -- on P1, in the spike train, or nowhere in particular?
  2. does the P0 channel still have a local maximum at the true position?
  3. would a decode that forces P0 < P1 < P2 < P3 < P4 in time recover it?
Prints numbers only.
"""
import numpy as np
import torch
from scipy.signal import find_peaks

from src.model import PeakUNet
from src.toy_waveforms import ToyConfig, make_dataset
from src.train import normalise, N_VAL, SIGMA

TOL = 2.0


def main():
    cfg = ToyConfig()
    Xv, _, Pv = make_dataset(N_VAL, seed=1, cfg=cfg, sigma=SIGMA)
    Xn = normalise(Xv)
    net = PeakUNet()
    net.load_state_dict(torch.load("outputs/peak_unet.pt", map_location="cpu"))
    net.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(Xn), 64):
            probs.append(torch.sigmoid(net(torch.from_numpy(Xn[i:i + 64]))).numpy())
    prob = np.concatenate(probs)                       # (N, 5, n)
    pos = prob.argmax(-1).astype(float)
    dt = np.abs(pos - Pv)

    bad = np.where(dt[:, 0] > TOL)[0]
    print(f"P0 wrong on {len(bad)} / {len(Xn)} traces "
          f"(gross >15: {(dt[:,0]>15).sum()})")

    on_p1 = sum(abs(pos[i, 0] - Pv[i, 1]) <= 3 for i in bad)
    before_p0 = sum(pos[i, 0] < Pv[i, 0] - 3 for i in bad)
    between = len(bad) - on_p1 - before_p0
    print(f"   landed on P1          : {on_p1}")
    print(f"   landed earlier (train): {before_p0}")
    print(f"   elsewhere             : {between}")

    # does the channel still know the answer?
    rescuable = 0
    rank_hist = []
    for i in bad:
        pk, _ = find_peaks(prob[i, 0], height=0.01)
        if pk.size == 0:
            continue
        d = np.abs(pk - Pv[i, 0])
        if d.min() <= TOL:
            rescuable += 1
            order = np.argsort(-prob[i, 0][pk])
            rank_hist.append(int(np.where(pk[order] == pk[d.argmin()])[0][0]) + 1)
    print(f"   true position is still a local max of the P0 channel: "
          f"{rescuable} / {len(bad)}")
    if rank_hist:
        r = np.array(rank_hist)
        print(f"   its rank among that channel's maxima: median {np.median(r):.0f}, "
              f"rank<=2 on {(r<=2).sum()} of {len(r)}")

    # how often is the raw argmax chain out of time order at all?
    unordered = (np.diff(pos, axis=1) <= 0).any(1).sum()
    print(f"argmax chain not strictly increasing in time: {unordered} / {len(Xn)}")
    p0_after_p1 = (pos[:, 0] >= pos[:, 1]).sum()
    print(f"P0 at or after P1: {p0_after_p1} / {len(Xn)}")


if __name__ == "__main__":
    main()
