"""Train the peak U-Net on toy VFE waveforms and score it on held-out toys."""

import json                                                      # - | metrics dump
import time                                                      # - | wall-clock timing

import numpy as np                                               # - | arrays
import torch                                                     # - | training
import torch.nn as nn                                            # - | loss

from src.model import PeakUNet, peaks_from_logits                # - | architecture and decoder
from src.toy_waveforms import ToyConfig, make_dataset            # - | data generator

N_TRAIN = 8000                                                   # - | training toys
N_VAL = 1000                                                     # - | held-out toys
EPOCHS = 25                                                      # - | passes over the training set
BATCH = 32                                                       # - | minibatch size
LR = 1e-3                                                        # - | Adam learning rate
SIGMA = 4.0                                                      # - | heatmap target width in samples
GROSS_MISS = 15                                                  # - | |dt| beyond this counts as a gross miss
CHANNEL_W = (1.0, 1.0, 1.0, 1.0, 1.0)                            # - | P0 is the hard one: weight its channel up
HARD_NEG_W = 1.0                                                 # - | extra weight on the spike train, where P0 is lost
OUT_DIR = "outputs"                                              # - | where artefacts land


def normalise(X):                                                # - | per-trace scaling
    """Zero-median, unit-max scaling so absolute volts never matter."""
    med = np.median(X, axis=-1, keepdims=True)                   # O(Nn) | robust baseline per trace
    out = X - med                                                # O(Nn) | remove the pedestal
    scale = np.max(np.abs(out), axis=-1, keepdims=True) + 1e-6   # O(Nn) | peak-to-peak scale
    return (out / scale).astype(np.float32)                      # O(Nn) | scaled trace
    # Total: time O(N n), space O(N n)


def batch_weights(p0, n, device):                                # - | per-sample loss weights
    """Weight the P0 channel up, and its spike train up again.

    P0 is the only landmark the network loses, and it loses it backwards into the
    train of small spikes that precedes it. Those samples are true negatives for
    the P0 channel, so penalising them harder is the direct lever.
    """
    B = len(p0)                                                  # - | batch size
    w = torch.ones(B, 5, n, device=device)                       # O(5Bn) | start uniform
    for k, cw in enumerate(CHANNEL_W):                           # O(5) | per-channel scale
        w[:, k] *= cw                                            # O(Bn) | apply it
    t = torch.arange(n, device=device).view(1, -1)               # O(n) | sample index
    edge = torch.as_tensor(p0, device=device).view(-1, 1) - 3.0 * SIGMA  # O(B) | just before P0
    w[:, 0] = torch.where(t < edge, w[:, 0] * HARD_NEG_W, w[:, 0])  # O(Bn) | upweight the train
    return w                                                     # - | (B, 5, n)
    # Total: time O(5 B n), space O(5 B n)


def evaluate(model, Xv, Pv, device, batch=64):                   # - | decode and score the val set
    """Return per-peak |dt| arrays plus confidences."""
    model.eval()                                                 # - | no batchnorm updates
    dts, confs = [], []                                          # - | collect per batch
    with torch.no_grad():                                        # - | inference only
        for i in range(0, len(Xv), batch):                       # O(N/b) | chunk the val set
            xb = torch.from_numpy(Xv[i:i + batch]).to(device)    # O(bn) | move to device
            pos, conf = peaks_from_logits(model(xb))             # O(bn) | decode heatmaps
            dts.append(pos.cpu().numpy() - Pv[i:i + batch])      # O(b) | signed error in samples
            confs.append(conf.cpu().numpy())                     # O(b) | per-peak confidence
    return np.concatenate(dts), np.concatenate(confs)            # - | (N,5) each
    # Total: time O(N n), space O(N)


def main():                                                      # - | full training run
    cfg = ToyConfig()                                            # - | generator settings
    device = "cuda" if torch.cuda.is_available() else "cpu"      # - | use the GPU when present
    t0 = time.time()                                             # - | start the clock

    Xt, Yt, Pt = make_dataset(N_TRAIN, seed=0, cfg=cfg, sigma=SIGMA)  # O(Nn) | training toys
    Xv, Yv, Pv = make_dataset(N_VAL, seed=1, cfg=cfg, sigma=SIGMA)    # O(Nn) | validation toys
    Xt, Xv = normalise(Xt), normalise(Xv)                        # O(Nn) | scale both splits
    print(f"data {Xt.shape} {Xv.shape} in {time.time()-t0:.1f}s on {device}", flush=True)  # - | progress

    model = PeakUNet().to(device)                                # - | build the network
    opt = torch.optim.Adam(model.parameters(), lr=LR)            # - | optimiser
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)   # - | decay the learning rate
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")             # - | weighted per sample, reduced by hand
    Xt_t = torch.from_numpy(Xt)                                  # O(Nn) | keep training data on host
    Yt_t = torch.from_numpy(Yt)                                  # O(5Nn) | targets on host
    history = []                                                 # - | per-epoch record

    for ep in range(EPOCHS):                                     # O(E) | epochs
        model.train()                                            # - | training mode
        perm = torch.randperm(len(Xt_t))                         # O(N) | shuffle each epoch
        run = 0.0                                                # - | running loss
        for i in range(0, len(perm), BATCH):                     # O(N/B) | minibatches
            idx = perm[i:i + BATCH]                              # - | this batch's indices
            xb = Xt_t[idx].to(device, non_blocking=True)         # O(Bn) | inputs
            yb = Yt_t[idx].to(device, non_blocking=True)         # O(5Bn) | targets
            wb = batch_weights(Pt[idx.numpy(), 0], xb.shape[-1], device)  # O(5Bn) | loss weights
            raw = loss_fn(model(xb), yb)                         # O(Bn) | per-sample loss
            loss = (raw * wb).sum() / wb.sum()                   # O(5Bn) | weighted mean
            opt.zero_grad(set_to_none=True)                      # - | clear grads
            loss.backward()                                      # O(Bn) | backprop
            opt.step()                                           # - | update weights
            run += loss.item() * len(idx)                        # - | accumulate
        sched.step()                                             # - | anneal the learning rate
        dt, _ = evaluate(model, Xv, Pv, device)                  # O(Nn) | score this epoch
        med = float(np.median(np.abs(dt)))                       # - | headline accuracy
        history.append({"epoch": ep + 1, "loss": run / len(Xt), "median_abs_dt": med})  # - | log it
        print(f"ep {ep+1:02d} loss {run/len(Xt):.5f} median|dt| {med:.2f}", flush=True)  # - | progress

    dt, conf = evaluate(model, Xv, Pv, device)                   # O(Nn) | final scoring
    labels = list(cfg.labels)                                    # - | P0..P4
    per_peak = {}                                                # - | metrics per peak
    for k, name in enumerate(labels):                            # O(5) | one peak at a time
        a = np.abs(dt[:, k])                                     # - | absolute error for this peak
        per_peak[name] = {                                       # - | record the numbers
            "median_abs_dt": float(np.median(a)),                # - | typical error
            "mean_abs_dt": float(a.mean()),                      # - | mean error
            "p95_abs_dt": float(np.percentile(a, 95)),           # - | tail error
            "frac_within_2": float((a <= 2).mean()),             # - | done-criterion fraction
            "gross_miss_rate": float((a > GROSS_MISS).mean()),   # - | decoy-fooled rate
            "mean_conf": float(conf[:, k].mean()),               # - | mean heatmap confidence
        }
    metrics = {                                                  # - | summary blob
        "device": device,
        "n_train": N_TRAIN, "n_val": N_VAL, "epochs": EPOCHS, "sigma": SIGMA,
        "params": int(sum(p.numel() for p in model.parameters())),
        "wall_seconds": round(time.time() - t0, 1),
        "all_five_within_2": float((np.abs(dt) <= 2).all(axis=1).mean()),  # - | whole-trace success
        "gross_miss_any": float((np.abs(dt) > GROSS_MISS).any(axis=1).mean()),  # - | any peak badly wrong
        "per_peak": per_peak, "history": history,
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as fh:             # - | persist metrics
        json.dump(metrics, fh, indent=2)                         # O(1) | write json
    torch.save(model.state_dict(), f"{OUT_DIR}/peak_unet.pt")    # - | persist weights
    np.savez_compressed(f"{OUT_DIR}/val_eval.npz", dt=dt, conf=conf, P=Pv,  # - | arrays for the report
                        X=Xv[:12], logits_P=Pv[:12])             # - | keep a dozen traces for plots
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))  # - | final numbers
    # Total: time O(E * N * n), space O(N * n)


if __name__ == "__main__":                                       # - | script entry
    main()                                                       # - | run it
