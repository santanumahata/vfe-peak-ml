"""Why does the hybrid still lose? Compare decoders and count constraint violations."""

import numpy as np                                               # - | arrays
import torch                                                     # - | CNN inference
from sklearn.ensemble import HistGradientBoostingClassifier      # - | stage-2 model

from src.hybrid import (build_table, candidate_features, cnn_heatmaps,  # - | stage-1/2 pieces
                        decode_trace, score)
from src.model import PeakUNet                                   # - | network
from src.toy_waveforms import make_dataset                       # - | toys
from src.train import normalise                                  # - | input scaling

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = PeakUNet().to(dev)
m.load_state_dict(torch.load("outputs/peak_unet.pt", map_location=dev))
m.eval()

Xt, _, Pt = make_dataset(2000, seed=10); Xt = normalise(Xt)
Xv, _, Pv = make_dataset(1000, seed=1); Xv = normalise(Xv)
pt, _ = cnn_heatmaps(m, Xt, dev)
pv, cnn_pos = cnn_heatmaps(m, Xv, dev)
Ft, Lt, _, _ = build_table(Xt, Pt, pt)
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, l2_regularization=1.0,
                                     random_state=0).fit(Ft, Lt)

mono, plain, greedy, viol, cnn_in_set = [], [], [], 0, 0
for i in range(len(Xv)):
    idx, feats = candidate_features(Xv[i, 0], pv[i])
    p = clf.predict_proba(feats)
    logp = np.log(np.clip(p[:, :5], 1e-12, None))
    apex = Xv[i, 0][idx]
    a = decode_trace(idx, logp, apex, monotone=True)
    b = decode_trace(idx, logp, apex, monotone=False)
    g = idx[logp.argmax(axis=0)].astype(np.float32)              # greedy: best candidate per channel
    mono.append(a if a is not None else cnn_pos[i])
    plain.append(b if b is not None else cnn_pos[i])
    greedy.append(g)
    hits = [np.abs(idx - cnn_pos[i, k]).min() for k in range(5)]  # is the CNN answer a candidate?
    cnn_in_set += int(max(hits) == 0)
    ap = np.array([apex[np.abs(idx - cnn_pos[i, k]).argmin()] for k in range(5)])
    viol += int(np.any(np.diff(ap[1:]) < 0))                      # CNN chain breaks monotone apex?

import json
out = {"n_val": len(Xv), "n_stage2_train": len(Xt),
       "cnn_in_candidate_set": cnn_in_set, "cnn_chain_monotone_violations": viol,
       "variants": {}}
for name, arr in [("cnn_only", cnn_pos), ("dp_monotone", np.array(mono)),
                  ("dp_plain", np.array(plain)), ("greedy", np.array(greedy))]:
    r = score(np.asarray(arr) - Pv, name)
    out["variants"][name] = r
    print(f"{name:12s} all5<=2 {r['all_five_within_2']:.3f}  grossmiss {r['gross_miss_any']:.4f}  "
          + " ".join(f"{k}:{v['median_abs_dt']:.2f}" for k, v in r["per_peak"].items()), flush=True)
print(f"CNN answer present in candidate set: {cnn_in_set}/{len(Xv)}")
print(f"CNN chain violates monotone apex   : {viol}/{len(Xv)}")

from scipy.signal import find_peaks
sweep = []
for prom in [0.004, 0.01, 0.02, 0.04, 0.08]:
    counts, rec = [], []
    for i in range(300):
        idx, _ = find_peaks(Xv[i, 0], prominence=prom)
        counts.append(len(idx))
        for k in range(5):
            rec.append(np.abs(idx - Pv[i, k]).min() <= 3 if len(idx) else False)
    sweep.append({"prominence": prom, "cands_per_trace": float(np.mean(counts)),
                  "recall_within_3": float(np.mean(rec))})
out["scan_only_sweep"] = sweep
with open("outputs/ablation.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote outputs/ablation.json")
