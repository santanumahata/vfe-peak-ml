"""Two-stage peak finder: local-maxima candidates + CNN prior, decoded under ordering constraints.

Stage 1  scan the trace for every local maximum and describe each one (height, prominence,
         width, the floor step across it, its rank among candidates).
Stage 2  a gradient-boosted classifier labels each candidate P0..P4 or 'none', using those
         features plus the trained U-Net's heatmap value at that sample as a prior.
Decode   dynamic programming picks one candidate per peak, strictly increasing in time and
         (for P1..P4) non-decreasing in apex height, maximising total log-probability.
"""

import json                                                      # - | metrics dump

import numpy as np                                               # - | arrays
import torch                                                     # - | run the CNN
from scipy.signal import find_peaks, peak_widths                 # - | candidate scan
from sklearn.ensemble import HistGradientBoostingClassifier      # - | stage-2 model

from src.model import PeakUNet, peaks_from_logits                # - | stage-1 CNN
from src.toy_waveforms import ToyConfig, make_dataset            # - | toys

PROMINENCE = 0.010                                               # - | scan threshold on the normalised trace
PRIOR_MIN = 0.05                                                 # - | keep heatmap maxima above this probability
MATCH_TOL = 3.0                                                  # - | candidate counts as peak k within this many samples
FLOOR_HALF = 30                                                  # - | half-window used to measure the local floor
N_FEAT = 22                                                      # - | feature count per candidate
GROSS_MISS = 15                                                  # - | |dt| beyond this is a gross miss
LABELS = ["P0", "P1", "P2", "P3", "P4"]                          # - | channel order


def propose(sig, prob):                                          # - | union of scan maxima and heatmap maxima
    """Scan maxima alone are systematically displaced from the pulse centre, so the CNN's own
    heatmap maxima are added: the network's answer is always inside the candidate set."""
    scan, _ = find_peaks(sig, prominence=PROMINENCE)             # O(n) | waveform local maxima
    from_cnn = []                                                # - | heatmap-proposed samples
    for k in range(prob.shape[0]):                               # O(5) | one channel per peak
        pk, _ = find_peaks(prob[k], height=PRIOR_MIN)            # O(n) | maxima of this channel
        from_cnn.extend(pk.tolist())                             # - | collect them
        from_cnn.append(int(prob[k].argmax()))                   # - | always include the channel argmax
    idx = np.unique(np.concatenate([scan, np.array(from_cnn, dtype=int)]))  # O(C log C) | dedup, time-ordered
    is_cnn = np.isin(idx, np.array(from_cnn, dtype=int)).astype(np.float32)  # O(C) | who proposed it
    is_scan = np.isin(idx, scan).astype(np.float32)              # O(C) | scan also found it
    return idx, is_cnn, is_scan                                  # - | candidates and their provenance
    # Total: time O(n), space O(C)


def candidate_features(sig, prob):                               # - | stage 1 for one trace
    """Return (idx, feats) for every candidate worth considering."""
    idx, is_cnn, is_scan = propose(sig, prob)                    # O(n) | union candidate set
    if len(idx) == 0:                                            # - | degenerate trace
        return idx, np.zeros((0, N_FEAT), dtype=np.float32)      # - | nothing to classify
    prom = np.array([sig[i] - min(sig[max(0, i - FLOOR_HALF):i + 1].min(),
                                  sig[i:i + FLOOR_HALF + 1].min()) for i in idx])  # O(Cw) | prominence by hand
    widths = peak_widths(sig, idx, rel_height=0.5)[0]            # O(n) | FWHM in samples
    n = len(sig)                                                 # - | trace length
    apex = sig[idx]                                              # - | value at the maximum
    left = np.array([np.median(sig[max(0, i - FLOOR_HALF):max(1, i - 4)]) for i in idx])   # O(Cw) | floor before
    right = np.array([np.median(sig[min(n - 1, i + 5):min(n, i + FLOOR_HALF)]) for i in idx])  # O(Cw) | floor after
    rank_apex = np.argsort(np.argsort(-apex)) / len(idx)         # O(C log C) | 0 = tallest candidate
    rank_prom = np.argsort(np.argsort(-prom)) / len(idx)         # O(C log C) | 0 = most prominent
    pri = prob[:, idx].T                                         # O(5C) | CNN heatmap value at the candidate
    pri_win = np.stack([prob[:, max(0, i - 3):i + 4].max(axis=1) for i in idx])  # O(5C) | CNN max nearby
    feats = np.column_stack([                                    # - | assemble the table
        idx / n,                                                 # - | where in the trace
        apex, prom, widths,                                      # - | size and shape
        apex - left, right - left,                               # - | height above floor, floor step across it
        apex / (apex.max() + 1e-6),                              # - | relative height
        prom / (prom.max() + 1e-6),                              # - | relative prominence
        rank_apex, rank_prom,                                     # - | ordinal position among candidates
        pri, pri_win,                                            # - | 10 CNN prior features
        is_cnn, is_scan,                                         # - | provenance of the candidate
    ]).astype(np.float32)                                        # - | (C, 22)
    return idx, feats                                            # - | candidates and their features
    # Total: time O(n + C*w), space O(C)


def label_candidates(idx, pos, tol=MATCH_TOL):                   # - | supervision for stage 2
    """Class k for a candidate within tol of true peak k, else class 5 ('none')."""
    lab = np.full(len(idx), 5, dtype=np.int64)                   # O(C) | default: not a peak
    for k in range(5):                                           # O(5) | one true peak at a time
        d = np.abs(idx - pos[k])                                 # O(C) | distance to this peak
        if d.min() <= tol:                                       # - | only if something is close enough
            lab[int(d.argmin())] = k                             # - | nearest candidate takes the label
    return lab                                                   # - | per-candidate class
    # Total: time O(C), space O(C)


def cnn_heatmaps(model, X, device, batch=64):                    # - | stage-1 priors for a whole split
    """Return (probs, cnn_pos) for every trace."""
    probs, pos_all = [], []                                      # - | collect per batch
    model.eval()                                                 # - | inference mode
    with torch.no_grad():                                        # - | no autograd
        for i in range(0, len(X), batch):                        # O(N/b) | chunked
            xb = torch.from_numpy(X[i:i + batch]).to(device)     # O(bn) | to device
            logits = model(xb)                                   # O(bn) | forward
            probs.append(torch.sigmoid(logits).cpu().numpy())    # O(5bn) | heatmaps
            pos_all.append(peaks_from_logits(logits)[0].cpu().numpy())  # O(5b) | CNN-only answer
    return np.concatenate(probs), np.concatenate(pos_all)        # - | (N,5,n) and (N,5)
    # Total: time O(N n), space O(N n)


def build_table(X, P, probs):                                    # - | flatten all candidates into one table
    """Return (feats, labels, trace_id, cand_idx) over every candidate in the split."""
    F, L, G, I = [], [], [], []                                  # - | accumulators
    for i in range(len(X)):                                      # O(N) | per trace
        idx, feats = candidate_features(X[i, 0], probs[i])       # O(n) | stage 1
        if len(idx) == 0:                                        # - | skip empty scans
            continue                                             # - | nothing to add
        F.append(feats)                                          # - | features
        L.append(label_candidates(idx, P[i]))                    # O(C) | labels
        G.append(np.full(len(idx), i))                           # - | which trace
        I.append(idx)                                            # - | candidate sample positions
    return (np.concatenate(F), np.concatenate(L),                # - | stacked table
            np.concatenate(G), np.concatenate(I))                # - | groups and positions
    # Total: time O(N n), space O(N C)


def decode_trace(idx, logp, apex, monotone=True):                # - | constrained assignment
    """Pick one candidate per peak: strictly increasing time, non-decreasing apex for P1..P4."""
    C = len(idx)                                                 # - | candidate count
    NEG = -1e9                                                   # - | impossible score
    dp = np.full((5, C), NEG)                                    # O(5C) | best score for peak k at candidate c
    bk = np.full((5, C), -1, dtype=np.int64)                     # O(5C) | backpointer
    dp[0] = logp[:, 0]                                           # O(C) | P0 may sit anywhere
    for k in range(1, 5):                                        # O(4) | peaks P1..P4
        for c in range(C):                                       # O(C) | candidate for peak k
            best, arg = NEG, -1                                  # - | search the previous peak
            for p in range(c):                                   # O(C) | must be earlier in time
                if dp[k - 1, p] <= NEG:                          # - | unreachable state
                    continue                                     # - | skip
                if monotone and k >= 2 and apex[c] < apex[p]:    # - | P2..P4 must not be shorter than the one before
                    continue                                     # - | violates monotone amplitude
                if dp[k - 1, p] > best:                          # - | better predecessor
                    best, arg = dp[k - 1, p], p                  # - | remember it
            if arg >= 0:                                         # - | reachable
                dp[k, c] = best + logp[c, k]                     # - | extend the path
                bk[k, c] = arg                                   # - | store the backpointer
    end = int(np.argmax(dp[4]))                                  # - | best final state
    if dp[4, end] <= NEG:                                        # - | no valid chain of five
        return None                                              # - | caller falls back to the CNN
    out = [0] * 5                                                # - | chosen candidate per peak
    out[4] = end                                                 # - | walk the backpointers
    for k in range(4, 0, -1):                                    # O(4) | backtrack
        out[k - 1] = int(bk[k, out[k]])                          # - | previous choice
    return idx[out].astype(np.float32)                           # - | five sample positions
    # Total: time O(5 C^2), space O(5C)


def hybrid_positions(clf, X, probs, cnn_pos):                    # - | stage 2 + decode for a split
    """Return (positions, n_fallback) for every trace."""
    out = np.zeros((len(X), 5), dtype=np.float32)                # O(N) | answers
    fallback = 0                                                 # - | traces where the DP found no chain
    for i in range(len(X)):                                      # O(N) | per trace
        idx, feats = candidate_features(X[i, 0], probs[i])       # O(n) | stage 1
        if len(idx) < 5:                                         # - | too few candidates to assign
            out[i] = cnn_pos[i]                                  # - | fall back to the CNN
            fallback += 1                                        # - | count it
            continue                                             # - | next trace
        p = clf.predict_proba(feats)                             # O(C) | class probabilities
        logp = np.log(np.clip(p[:, :5], 1e-12, None))            # O(C) | log-prob of being peak k
        chosen = decode_trace(idx, logp, X[i, 0][idx])           # O(C^2) | constrained pick
        if chosen is None:                                       # - | no valid chain
            out[i] = cnn_pos[i]                                  # - | fall back
            fallback += 1                                        # - | count it
        else:                                                    # - | normal case
            out[i] = chosen                                      # - | hybrid answer
    return out, fallback                                         # - | positions and fallback count
    # Total: time O(N * C^2), space O(N)


def score(dt, name):                                             # - | metric block for one method
    """Per-peak and whole-trace accuracy from signed errors."""
    per = {}                                                     # - | per-peak numbers
    for k, lab in enumerate(LABELS):                             # O(5) | each peak
        a = np.abs(dt[:, k])                                     # - | absolute error
        per[lab] = {"median_abs_dt": float(np.median(a)), "mean_abs_dt": float(a.mean()),
                    "p95_abs_dt": float(np.percentile(a, 95)), "frac_within_2": float((a <= 2).mean()),
                    "gross_miss_rate": float((a > GROSS_MISS).mean())}  # - | row of metrics
    return {"method": name, "all_five_within_2": float((np.abs(dt) <= 2).all(axis=1).mean()),
            "gross_miss_any": float((np.abs(dt) > GROSS_MISS).any(axis=1).mean()),
            "per_peak": per}                                     # - | summary
    # Total: time O(N), space O(1)


def main():                                                      # - | train stage 2 and compare
    cfg = ToyConfig()                                            # - | same generator as the CNN
    device = "cuda" if torch.cuda.is_available() else "cpu"      # - | GPU when available
    model = PeakUNet().to(device)                                # - | stage-1 network
    model.load_state_dict(torch.load("outputs/peak_unet.pt", map_location=device))  # - | trained weights

    from src.train import normalise                              # - | reuse the exact input scaling
    Xt, _, Pt = make_dataset(3000, seed=10, cfg=cfg)             # O(Nn) | stage-2 training toys (unseen seeds)
    Xv, _, Pv = make_dataset(1000, seed=1, cfg=cfg)              # O(Nn) | same val split as the CNN
    Xt, Xv = normalise(Xt), normalise(Xv)                        # O(Nn) | scale

    prob_t, pos_t = cnn_heatmaps(model, Xt, device)              # O(Nn) | priors for training candidates
    prob_v, pos_v = cnn_heatmaps(model, Xv, device)              # O(Nn) | priors for validation
    Ft, Lt, _, _ = build_table(Xt, Pt, prob_t)                   # O(Nn) | stage-2 training table
    print("stage-2 table", Ft.shape, "class counts", np.bincount(Lt, minlength=6), flush=True)  # - | sanity

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         l2_regularization=1.0, random_state=0)  # - | tabular model
    clf.fit(Ft, Lt)                                              # O(T) | fit stage 2

    hyb, fallback = hybrid_positions(clf, Xv, prob_v, pos_v)     # O(N C^2) | hybrid answers
    res = {"cnn_only": score(pos_v - Pv, "cnn_only"),            # - | baseline metrics
           "hybrid": score(hyb - Pv, "hybrid"),                  # - | hybrid metrics
           "n_val": len(Xv), "n_stage2_train": len(Xt),
           "fallback_traces": int(fallback),                     # - | how often the DP gave up
           "prominence": PROMINENCE, "match_tol": MATCH_TOL,
           "mean_candidates": float(len(Ft) / len(Xt))}          # - | scan load per trace
    with open("outputs/metrics_hybrid.json", "w") as fh:         # - | persist
        json.dump(res, fh, indent=2)                             # - | write json
    np.savez_compressed("outputs/hybrid_eval.npz", hyb=hyb, cnn=pos_v, P=Pv)  # - | arrays for the report
    for m in ("cnn_only", "hybrid"):                             # O(2) | print both
        r = res[m]                                               # - | that method
        print(f"{m:10s} all5<=2 {r['all_five_within_2']:.3f}  grossmiss {r['gross_miss_any']:.4f}  "
              + " ".join(f"{k}:{v['median_abs_dt']:.2f}" for k, v in r["per_peak"].items()), flush=True)  # - | row
    print("fallback traces", fallback, "of", len(Xv))            # - | how often stage 2 abstained
    # Total: time O(N n + N C^2), space O(N n)


if __name__ == "__main__":                                       # - | script entry
    main()                                                       # - | run it
