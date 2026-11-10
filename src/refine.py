"""Report the apex beside every predicted peak, and snap to it when the network
has plainly landed on the shoulder of a peak that is right there.

The network gives a sub-sample-accurate position but nothing about height. For QC
both are wanted: where the peak is, and how high it got. This module adds the
height, and uses it as a guard: if a local maximum inside the search window is
clearly higher than the trace value at the network's own position, the network
missed a peak sitting in front of it and the position moves there. Otherwise the
network's answer stands, because on a clean peak the noisy sample maximum is a
worse estimate of the pulse centre than the heatmap argmax is.
"""
import numpy as np                                               # - | arrays
from scipy.signal import find_peaks                              # - | local maxima

# Defaults chosen by the sweep in diagnostics/refine_sweep.py and
# diagnostics/refine_gated.py on 1000 held-out toys. Snapping harder than this
# makes things worse, badly: with no gain guard it costs more than half the
# traces, because on a clean peak the noisy sample maximum is a worse estimate of
# the pulse centre than the network's own answer.
WINDOW = 3                                                       # - | search half-width in samples
MIN_GAIN = 0.05                                                  # - | snap only if the apex is this much higher
SMOOTH = 1                                                       # - | boxcar width used to find maxima (1 = none)
MAX_CONF = 0.50                                                  # - | only snap where the network is unsure


def _boxcar(v, w):                                               # - | small running mean
    """Smooth once so single-sample glitches do not look like apexes."""
    if w <= 1:                                                   # - | smoothing disabled
        return v                                                 # - | unchanged
    k = np.ones(w, dtype=np.float64) / w                         # O(w) | kernel
    return np.convolve(v, k, mode="same")                        # O(n w) | smoothed trace
    # Total: time O(n w), space O(n)


def refine_trace(sig, pos, conf, window=WINDOW, min_gain=MIN_GAIN,
                 smooth=SMOOTH, max_conf=MAX_CONF):              # - | one trace, five peaks
    """Return refined positions plus the apex index and apex height per peak.

    `sig` is the normalised trace the network saw, so `min_gain` is in the same
    units as the network's input (max |value| = 1 by construction).
    """
    n = sig.size                                                 # - | trace length
    sm = _boxcar(sig.astype(np.float64), smooth)                 # O(n) | glitch-resistant copy
    maxima, _ = find_peaks(sm)                                   # O(n) | every local maximum
    out_pos = np.array(pos, dtype=np.float64).copy()             # O(1) | start from the network
    apex_idx = np.zeros(5, dtype=np.int64)                       # - | where the apex sits
    apex_val = np.zeros(5, dtype=np.float64)                     # - | how high it gets
    snapped = np.zeros(5, dtype=bool)                            # - | did the position move

    for k in range(5):                                           # O(5) | one landmark at a time
        p = int(round(out_pos[k]))                               # - | integer sample
        lo, hi = max(p - window, 0), min(p + window + 1, n)      # - | search window
        here = sm[np.clip(p, 0, n - 1)]                          # - | height at the network's answer
        cand = maxima[(maxima >= lo) & (maxima < hi)]            # O(m) | maxima inside the window
        if cand.size:                                            # - | at least one apex nearby
            j = int(cand[np.argmax(sm[cand])])                   # O(c) | the highest of them
        else:                                                    # - | flat window, no maximum
            j = int(lo + np.argmax(sm[lo:hi]))                   # O(w) | fall back to the plain max
        apex_idx[k] = j                                          # - | record it either way
        apex_val[k] = sig[j]                                     # - | height on the unsmoothed trace
        if sm[j] - here > min_gain and conf[k] < max_conf:       # - | a peak the network stepped off
            out_pos[k] = float(j)                                # - | move onto it
            snapped[k] = True                                    # - | flag the move
    return out_pos, np.asarray(conf), apex_idx, apex_val, snapped
    # Total: time O(n w + 5 m), space O(n)


def refine_batch(X, pos, conf, window=WINDOW, min_gain=MIN_GAIN,
                 smooth=SMOOTH, max_conf=MAX_CONF):              # - | whole val set
    """Apply `refine_trace` to a batch; `X` is (N, 1, n) normalised traces."""
    N = X.shape[0]                                               # - | trace count
    P = np.zeros((N, 5)); A = np.zeros((N, 5))                   # - | positions and heights
    I = np.zeros((N, 5), dtype=np.int64)                         # - | apex indices
    S = np.zeros((N, 5), dtype=bool)                             # - | snap flags
    for i in range(N):                                           # O(N) | per trace
        P[i], _, I[i], A[i], S[i] = refine_trace(
            X[i, 0], pos[i], conf[i], window, min_gain,
            smooth, max_conf)                                    # O(n w) | refine one
    return P, I, A, S
    # Total: time O(N n w), space O(N)


def predict(model, X_raw, batch=64, **kw):                       # - | the full inference path
    """Positions, confidences, apex sample and apex height for raw traces.

    `X_raw` is (N, 1, n) in whatever units the board gives. Returns a dict whose
    `height` is in those original units, and `height_normalised` in the units the
    network sees. This is the only function an analysis should call.
    """
    import torch                                                 # - | local import: generator use needs no torch
    from src.train import normalise                              # - | the exact scaling the model was trained with

    Xn = normalise(np.asarray(X_raw, dtype=np.float32))          # O(Nn) | must match training
    med = np.median(X_raw, axis=-1, keepdims=True)               # O(Nn) | to undo the scaling for heights
    scale = np.max(np.abs(X_raw - med), axis=-1, keepdims=True) + 1e-6  # O(Nn) | per-trace scale
    model.eval()                                                 # - | inference mode
    pos = np.zeros((len(Xn), 5)); conf = np.zeros((len(Xn), 5))  # - | network outputs
    with torch.no_grad():                                        # - | no graph
        for i in range(0, len(Xn), batch):                       # O(N/b) | chunk
            p = torch.sigmoid(model(torch.from_numpy(Xn[i:i + batch])))  # O(bn) | heatmaps
            c, a = p.max(dim=-1)                                 # O(bn) | argmax and height
            pos[i:i + batch] = a.cpu().numpy()                   # - | positions
            conf[i:i + batch] = c.cpu().numpy()                  # - | confidences
    P, I, A, S = refine_batch(Xn, pos, conf, **kw)               # O(N n w) | apex and guarded snap
    return {                                                     # - | one record per trace
        "position": P,                                           # - | refined sample index per peak
        "confidence": conf,                                      # - | network confidence per peak
        "apex_sample": I,                                        # - | where the local maximum sits
        "height_normalised": A,                                  # - | its height in network units
        "height": A * scale[:, 0] + med[:, 0],                   # - | its height in the board's units
        "snapped": S,                                            # - | did the guard move this peak
        "position_network": pos,                                 # - | the unrefined answer, kept for audit
    }
    # Total: time O(N n w), space O(N n)
