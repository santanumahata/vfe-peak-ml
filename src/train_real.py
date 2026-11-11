"""Train a peak network on real LVR input-current traces (data/real_shots.npz).

Split by RUN, never by shot: shots inside a run are near-copies. The runs that
contain finder failures (n_found == 3) are held out entirely as the hard test;
of the rest, VAL_FRAC of runs are validation. Labels: P0 by rule, P1..P4 = finder
apex times (bins of 10 us). Errors are reported in ms against those labels.

  python -m src.train_real --arch unet|transformer --out outputs/real/<tag>
"""
import argparse                                                  # - | options
import json                                                      # - | metrics
import os                                                        # - | output dir
import time                                                      # - | timing

import numpy as np                                               # - | arrays
import torch                                                     # - | training
import torch.nn as nn                                            # - | loss

from src.model import PeakUNet                                   # - | convolutional baseline
from src.train import normalise                                  # - | same per-trace scaling as the toys

LABELS = ["P0", "P1", "P2", "P3", "P4"]                          # - | inrush, A1, B1, A2, B2
VAL_FRAC = 0.2                                                   # - | fraction of clean runs held out
SHIFT = 300                                                      # - | max augmentation shift, bins (3 ms)
TOLS_MS = (0.02, 0.05, 0.1)                                      # - | reporting thresholds
GROSS_MS = 1.0                                                   # - | beyond this a landmark is lost


def split_runs(run, n_found, seed=0):                            # - | run-level split
    """Return boolean masks (train, val, hard) over shots."""
    bad_runs = set(run[n_found == 3])                            # - | runs where the finder failed
    clean = sorted(set(run) - bad_runs)                          # - | the rest
    rng = np.random.default_rng(seed)                            # - | reproducible
    val_runs = set(rng.choice(clean, int(round(VAL_FRAC * len(clean))), replace=False))
    hard = np.isin(run, list(bad_runs))                          # - | whole failure runs
    val = np.isin(run, list(val_runs))                           # - | held-out clean runs
    return ~(hard | val), val, hard
    # Total: time O(N), space O(N)


def targets(P, n, sigma):                                        # - | Gaussian heatmaps
    t = np.arange(n, dtype=np.float32)                           # O(n) | bin axis
    return np.exp(-0.5 * ((t[None, None, :] - P[:, :, None]) / sigma) ** 2).astype(np.float32)
    # Total: time O(5 N n), space O(5 N n)


def augment(x, p, rng):                                          # - | one batch, numpy
    """Random shift (labels follow), gain and small noise; the shape is what must be learned."""
    B, _, n = x.shape                                            # - | batch geometry
    out = np.empty_like(x); q = p.copy()                         # - | outputs
    for i in range(B):                                           # O(B) | per trace
        s = int(rng.integers(-SHIFT, SHIFT + 1))                 # - | shift in bins
        out[i, 0] = np.roll(x[i, 0], s)                          # O(n) | move the trace
        if s > 0:                                                # - | fill the wrapped head
            out[i, 0, :s] = x[i, 0, 0]                           # - | hold the first value
        elif s < 0:                                              # - | fill the wrapped tail
            out[i, 0, s:] = x[i, 0, -1]                          # - | hold the last value
        q[i] += s                                                # - | labels move with it
        out[i, 0] *= rng.uniform(0.8, 1.2)                       # - | gain
        out[i, 0] += rng.normal(0, rng.uniform(0, 0.005), n).astype(np.float32)  # O(n) | extra noise
    return out, q
    # Total: time O(B n), space O(B n)


class PeakTransformer(nn.Module):                                # - | conv stem + transformer + conv head
    """Patch the trace into tokens, mix them globally with self-attention, and decode
    back to full resolution. The attention lets every position see the whole trace --
    e.g. the A1 shoulder can be placed relative to B1 and the inrush spike."""

    def __init__(self, n_peaks=5, patch=16, d=128, depth=4, heads=4, n_max=5120):
        super().__init__()
        self.patch = patch                                       # - | bins per token
        self.stem = nn.Sequential(                               # - | local features at full rate
            nn.Conv1d(1, 32, 9, padding=4), nn.GELU(),
            nn.Conv1d(32, 32, 9, padding=4), nn.GELU())
        self.embed = nn.Conv1d(32, d, patch, stride=patch)       # - | tokens: n/patch of them
        self.pos = nn.Parameter(torch.zeros(1, n_max // patch, d))  # - | learned position code
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)       # - | global mixing
        self.up = nn.ConvTranspose1d(d, 32, patch, stride=patch)  # - | back to full rate
        self.head = nn.Sequential(                               # - | fuse with the stem's detail
            nn.Conv1d(64, 32, 9, padding=4), nn.GELU(), nn.Conv1d(32, n_peaks, 1))

    def forward(self, x):                                        # - | (B, 1, n) -> (B, 5, n)
        s = self.stem(x)                                         # O(B n) | local features
        z = self.embed(s).transpose(1, 2)                        # O(B n) | (B, tokens, d)
        z = self.encoder(z + self.pos[:, :z.shape[1]])           # O(B T^2 d) | attention
        u = self.up(z.transpose(1, 2))                           # O(B n) | full resolution
        return self.head(torch.cat([u, s], 1))                   # O(B n) | logits


def scores(pred, P, dt_ms):                                      # - | errors in ms
    e = np.abs(pred - P) * dt_ms                                 # - | (N, 5)
    res = {"all_five_within_%gms" % t: float(np.nanmean(np.all(e <= t, 1))) for t in TOLS_MS}
    res["gross_any"] = float(np.mean(np.any(e > GROSS_MS, 1)))
    res["per_peak"] = {LABELS[k]: {
        "median_ms": float(np.nanmedian(e[:, k])),
        **{"within_%gms" % t: float(np.nanmean(e[:, k] <= t)) for t in TOLS_MS},
        "gross": float(np.nanmean(e[:, k] > GROSS_MS))} for k in range(5)}
    return res


def predict(model, X, device, batch=64):                         # - | argmax positions + confidences
    model.eval(); pos, conf = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            p = torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).to(device)))
            c, a = p.max(-1); pos.append(a.cpu().numpy()); conf.append(c.cpu().numpy())
    return np.concatenate(pos).astype(float), np.concatenate(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet", choices=["unet", "transformer"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data", default="data/real_shots.npz")
    ap.add_argument("--out", default="outputs/real/unet")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(a.data)
    X = normalise(d["X"]); P = d["P"].astype(np.float32); dt_ms = float(d["dt_ms"][0])
    tr, va, hard = split_runs(d["run"], d["n_found"])
    tr &= d["good"]                                              # - | only fully labelled shots train
    print(f"train {tr.sum()} shots / {len(set(d['run'][tr]))} runs | "
          f"val {va.sum()} / {len(set(d['run'][va]))} | hard {hard.sum()} / "
          f"{len(set(d['run'][hard]))} on {device}", flush=True)

    model = (PeakUNet() if a.arch == "unet" else PeakTransformer()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, total_steps=a.epochs * int(np.ceil(tr.sum() / a.batch)))
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr, Ptr = X[tr], P[tr]
    t0 = time.time(); hist = []
    for ep in range(a.epochs):
        model.train(); perm = rng.permutation(len(Xtr)); run = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            xb, pb = augment(Xtr[idx], Ptr[idx], rng)
            yb = torch.from_numpy(targets(pb, xb.shape[-1], a.sigma)).to(device)
            loss = loss_fn(model(torch.from_numpy(xb).to(device)), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item() * len(idx)
        pv, _ = predict(model, X[va], device)
        s = scores(pv, P[va], dt_ms)
        hist.append({"epoch": ep + 1, "loss": run / len(Xtr), **{k: v for k, v in s.items() if k != "per_peak"}})
        print(f"ep {ep+1:02d} loss {run/len(Xtr):.5f} val all5<=0.05ms "
              f"{s['all_five_within_0.05ms']:.3f} gross {s['gross_any']:.3f}", flush=True)

    pv, cv = predict(model, X[va], device)
    ph, ch = predict(model, X[hard], device)
    hard_good = d["good"][hard]
    m = {"arch": a.arch, "params": n_params, "device": device, "epochs": a.epochs,
         "sigma_bins": a.sigma, "dt_ms": dt_ms, "wall_s": round(time.time() - t0, 1),
         "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_hard": int(hard.sum()),
         "val": scores(pv, P[va], dt_ms),
         "hard_runs_clean_shots": scores(ph[hard_good], P[hard][hard_good], dt_ms),
         "history": hist}
    with open(f"{a.out}/metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    torch.save(model.state_dict(), f"{a.out}/model.pt")
    np.savez_compressed(f"{a.out}/predictions.npz", val_pos=pv, val_conf=cv,
                        hard_pos=ph, hard_conf=ch, val_idx=np.flatnonzero(va),
                        hard_idx=np.flatnonzero(hard))
    print(json.dumps({k: v for k, v in m.items() if k not in ("history",)}, indent=1))


if __name__ == "__main__":
    main()
