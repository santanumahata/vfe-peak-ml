"""Build the results PDF: method, toy examples, predictions, error distributions, metrics."""

import json                                                      # - | read metrics
from dataclasses import asdict                                   # - | config to dict

import matplotlib                                                # - | plotting backend
matplotlib.use("Agg")                                            # - | headless rendering
import matplotlib.pyplot as plt                                  # - | figures
import numpy as np                                               # - | arrays
import torch                                                     # - | load the trained net
from matplotlib.backends.backend_pdf import PdfPages             # - | multi-page PDF

from src.model import PeakUNet, peaks_from_logits                # - | network and decoder
from src.toy_waveforms import ToyConfig, make_waveform           # - | generator for the gallery

OUT_PDF = "outputs/vfe_peak_ml_report.pdf"                       # - | deliverable path
LABELS = ["P0", "P1", "P2", "P3", "P4"]                          # - | channel order
COLORS = ["#888780", "#2a78d6", "#1baf7a", "#eda100", "#e34948"]  # - | one colour per peak


def page_text(pdf, title, lines):                                # - | a text-only page
    """Render a titled block of monospaced lines as one PDF page."""
    fig = plt.figure(figsize=(8.27, 11.69))                      # - | A4 portrait
    fig.text(0.07, 0.95, title, fontsize=16, weight="bold")      # - | page heading
    fig.text(0.07, 0.91, "\n".join(lines), fontsize=9, va="top", family="monospace")  # - | body
    pdf.savefig(fig)                                             # - | append the page
    plt.close(fig)                                               # - | free the figure
    # Total: time O(L), space O(1)


def page_gallery(pdf, cfg, seed=7):                              # - | six labelled toy traces
    rng = np.random.default_rng(seed)                            # - | reproducible draw
    fig, axes = plt.subplots(6, 1, figsize=(8.27, 11.69), sharex=True)  # - | one trace per row
    for ax in axes:                                              # O(6) | fill each panel
        sig, pos = make_waveform(rng, cfg)                       # O(n) | draw a toy
        ax.plot(sig, lw=0.6, color="#333333")                    # O(n) | the trace
        for k, p in enumerate(pos):                              # O(5) | mark the true peaks
            ax.axvline(p, color=COLORS[k], lw=0.8, ls="--")      # - | truth line
            ax.text(p, sig.max() * 1.03, LABELS[k], color=COLORS[k], fontsize=7, ha="center")  # - | label
        ax.set_ylabel("V (a.u.)", fontsize=7)                    # - | axis label
        ax.tick_params(labelsize=7)                              # - | small ticks
    axes[-1].set_xlabel("sample", fontsize=8)                    # - | time axis
    fig.suptitle("Toy waveforms with ground-truth peak positions", fontsize=13, weight="bold")  # - | title
    fig.tight_layout(rect=[0, 0, 1, 0.97])                       # - | leave room for the title
    pdf.savefig(fig)                                             # - | append
    plt.close(fig)                                               # - | free
    # Total: time O(6n), space O(n)


def page_predictions(pdf, X, P, model, n_show=4):                # - | trace plus heatmaps, truth vs predicted
    with torch.no_grad():                                        # - | inference only
        logits = model(torch.from_numpy(X[:n_show]))             # O(n) | forward pass
        prob = torch.sigmoid(logits).numpy()                     # O(5n) | heatmaps
        pos, conf = peaks_from_logits(logits)                     # O(5n) | decoded peaks
    pos, conf = pos.numpy(), conf.numpy()                        # - | to numpy
    fig, axes = plt.subplots(n_show * 2, 1, figsize=(8.27, 11.69),
                             gridspec_kw={"height_ratios": [2, 1] * n_show})  # - | trace over heatmap
    for i in range(n_show):                                      # O(k) | one validation trace per pair
        ax_t, ax_h = axes[2 * i], axes[2 * i + 1]                # - | trace axis, heatmap axis
        ax_t.plot(X[i, 0], lw=0.6, color="#333333")              # O(n) | normalised trace
        for k in range(5):                                       # O(5) | truth and prediction
            ax_t.axvline(P[i, k], color=COLORS[k], lw=0.8, ls="--")           # - | truth
            ax_t.axvline(pos[i, k], color=COLORS[k], lw=1.2, alpha=0.7)       # - | prediction
            ax_h.plot(prob[i, k], lw=0.8, color=COLORS[k], label=LABELS[k])   # O(n) | heatmap channel
        dt = pos[i] - P[i]                                       # - | signed error per peak
        ax_t.set_title("dt = " + "  ".join(f"{LABELS[k]}:{dt[k]:+.0f}" for k in range(5))
                       + "   conf " + " ".join(f"{c:.2f}" for c in conf[i]), fontsize=7)  # - | per-trace score
        ax_t.tick_params(labelsize=6); ax_h.tick_params(labelsize=6)  # - | small ticks
        ax_h.set_ylim(-0.05, 1.05)                               # - | probability range
        if i == 0:                                               # - | legend once
            ax_h.legend(fontsize=6, ncol=5, loc="upper center")  # - | channel key
    axes[-1].set_xlabel("sample", fontsize=8)                    # - | time axis
    fig.suptitle("Validation traces: dashed = truth, solid = argmax of heatmap",
                 fontsize=13, weight="bold")                     # - | title
    fig.tight_layout(rect=[0, 0, 1, 0.97])                       # - | spacing
    pdf.savefig(fig)                                             # - | append
    plt.close(fig)                                               # - | free
    # Total: time O(k n), space O(k n)


def page_errors(pdf, dt, conf):                                  # - | error and confidence distributions
    fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))        # - | six panels
    for k in range(5):                                           # O(5) | one histogram per peak
        ax = axes.ravel()[k]                                     # - | pick the panel
        ax.hist(dt[:, k], bins=np.arange(-20.5, 21.5, 1), color=COLORS[k])  # O(N) | signed error
        a = np.abs(dt[:, k])                                     # - | absolute error
        ax.set_title(f"{LABELS[k]}  median|dt|={np.median(a):.1f}  "
                     f"within2={100*(a<=2).mean():.1f}%  miss={100*(a>15).mean():.2f}%", fontsize=8)  # - | stats
        ax.set_xlabel("predicted - true (samples)", fontsize=7)  # - | axis label
        ax.tick_params(labelsize=6)                              # - | small ticks
    ax = axes.ravel()[5]                                         # - | last panel: confidences
    ax.boxplot([conf[:, k] for k in range(5)], tick_labels=LABELS)    # O(N) | confidence spread per peak
    ax.set_title("heatmap confidence by peak", fontsize=8)       # - | panel title
    ax.tick_params(labelsize=6)                                  # - | small ticks
    fig.suptitle("Position error on 1000 held-out toys", fontsize=13, weight="bold")  # - | title
    fig.tight_layout(rect=[0, 0, 1, 0.97])                       # - | spacing
    pdf.savefig(fig)                                             # - | append
    plt.close(fig)                                               # - | free
    # Total: time O(N), space O(N)


def page_training(pdf, history):                                 # - | loss and accuracy curves
    ep = [h["epoch"] for h in history]                           # O(E) | epoch axis
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.27, 6.0))      # - | two stacked panels
    a1.plot(ep, [h["loss"] for h in history], color="#2a78d6")   # O(E) | training loss
    a1.set_ylabel("BCE loss", fontsize=8); a1.set_yscale("log")  # - | log scale
    a2.plot(ep, [h["median_abs_dt"] for h in history], color="#e34948")  # O(E) | val accuracy
    a2.set_ylabel("val median |dt| (samples)", fontsize=8)       # - | axis label
    a2.set_xlabel("epoch", fontsize=8)                           # - | axis label
    for a in (a1, a2):                                           # O(1) | tidy both
        a.tick_params(labelsize=7); a.grid(alpha=0.3)            # - | ticks and grid
    fig.suptitle("Training history", fontsize=13, weight="bold")  # - | title
    fig.tight_layout(rect=[0, 0, 1, 0.95])                       # - | spacing
    pdf.savefig(fig)                                             # - | append
    plt.close(fig)                                               # - | free
    # Total: time O(E), space O(E)


def page_compare(pdf, hyb_metrics, abl, dt_cnn, dt_hyb):         # - | CNN-only vs hybrid
    """Bar comparison plus the scan-recall ceiling that explains the result."""
    fig = plt.figure(figsize=(8.27, 11.69))                      # - | A4 portrait
    fig.text(0.07, 0.96, "Stage 2: candidate scan + classifier", fontsize=15, weight="bold")  # - | heading
    names = list(abl["variants"].keys())                         # - | decoder variants
    all5 = [abl["variants"][n]["all_five_within_2"] for n in names]   # - | whole-trace accuracy
    miss = [abl["variants"][n]["gross_miss_any"] for n in names]      # - | gross-miss rate

    ax1 = fig.add_axes([0.10, 0.70, 0.36, 0.20])                 # - | left panel
    ax1.bar(names, all5, color=["#2a78d6", "#1baf7a", "#eda100", "#e34948"])  # O(V) | accuracy bars
    ax1.set_ylim(0.7, 1.0); ax1.set_ylabel("all five within 2 samples", fontsize=8)  # - | axis
    ax1.tick_params(labelsize=7, rotation=30)                    # - | ticks

    ax2 = fig.add_axes([0.58, 0.70, 0.36, 0.20])                 # - | right panel
    ax2.bar(names, [100 * x for x in miss], color=["#2a78d6", "#1baf7a", "#eda100", "#e34948"])  # O(V) | miss bars
    ax2.set_ylabel("traces with a gross miss (%)", fontsize=8)   # - | axis
    ax2.tick_params(labelsize=7, rotation=30)                    # - | ticks

    ax3 = fig.add_axes([0.10, 0.42, 0.84, 0.20])                 # - | error comparison
    bins = np.arange(-8.5, 9.5, 1)                               # - | shared bins
    ax3.hist(dt_cnn.ravel(), bins=bins, alpha=0.6, label="CNN only", color="#2a78d6")  # O(N) | baseline
    ax3.hist(dt_hyb.ravel(), bins=bins, alpha=0.6, label="hybrid", color="#e34948")    # O(N) | hybrid
    ax3.set_yscale("log"); ax3.legend(fontsize=8)                # - | log counts
    ax3.set_xlabel("predicted - true (samples), all peaks pooled", fontsize=8)  # - | axis
    ax3.tick_params(labelsize=7)                                 # - | ticks

    lines = [
        "WHY THE HYBRID DOES NOT WIN ON TOYS",
        "",
        f"  CNN answer inside the candidate set : {abl['cnn_in_candidate_set']}/{abl['n_val']} traces",
        f"  CNN chain violating monotone apex   : {abl['cnn_chain_monotone_violations']}/{abl['n_val']} traces",
        f"  DP fallbacks (no valid chain)       : {hyb_metrics['fallback_traces']}/{hyb_metrics['n_val']}",
        f"  mean candidates per trace           : {hyb_metrics['mean_candidates']:.1f}",
        "",
        "  The scan's local maxima are systematically displaced from the pulse centre:",
        "  the rising floor step and the previous peak's decay pull the apex off. Scan-only",
        "  candidate recall (true peak within 3 samples) therefore caps the method:",
        "",
        "    prominence   candidates/trace   recall within 3 samples",
    ]
    for s in abl["scan_only_sweep"]:                             # O(S) | one row per threshold
        lines.append(f"    {s['prominence']:>10.3f}   {s['cands_per_trace']:>16.1f}   {s['recall_within_3']:>23.3f}")
    lines += [
        "",
        "  Adding the CNN's own heatmap maxima to the candidate set lifts recall to ~0.997,",
        "  and the CNN answer is then always reachable - but the stage-2 classifier still",
        "  ranks worse than the heatmap it is fed, because that prior is one feature among 22",
        "  spread over ~170 candidates. Decoder choice is not the cause: monotone DP, plain DP",
        "  and greedy per-channel all land within 0.005 of each other.",
        "",
        "RECOMMENDATION",
        "  Keep the CNN-only path as the default. Keep the scan + classifier code as a switch to",
        "  re-test on real traces, where it may matter if the true peak is DEFINED as the sample",
        "  maximum, or if the CNN mislocalises on shapes the toys do not cover.",
    ]
    fig.text(0.07, 0.34, "\n".join(lines), fontsize=8, va="top", family="monospace")  # - | text block
    pdf.savefig(fig)                                             # - | append
    plt.close(fig)                                               # - | free
    # Total: time O(N + V + S), space O(N)


def main():                                                      # - | assemble the PDF
    cfg = ToyConfig()                                            # - | generator config
    with open("outputs/metrics.json") as fh:                     # - | load metrics
        m = json.load(fh)                                        # - | parsed json
    d = np.load("outputs/val_eval.npz")                          # - | val arrays
    model = PeakUNet()                                           # - | fresh net
    model.load_state_dict(torch.load("outputs/peak_unet.pt", map_location="cpu"))  # - | trained weights
    model.eval()                                                 # - | inference mode

    pp = m["per_peak"]                                           # - | per-peak metrics
    summary = [
        "WHAT THIS IS",
        "  Peak finder for VFE-board QC voltage traces, trained only on synthetic toys.",
        "  No real board data was used: the generator encodes the verbally described shape.",
        "  Every number below is measured on held-out TOYS, not on real traces.",
        "",
        "WAVEFORM MODEL ENCODED IN THE GENERATOR",
        "  - a train of many small spikes; the last one is P0",
        "  - P1..P4 rise monotonically in amplitude",
        "  - each main peak decays only partway, so the inter-peak floor keeps climbing",
        "  - small irregular structure sits on those floors and must not be reported",
        "  - spiky, asymmetric peaks: Gaussian rise into an exponential decay",
        "  - Gaussian readout noise plus occasional single-sample glitches",
        "",
        "METHOD",
        "  Input   one normalised trace, 1024 samples (median-subtracted, max-abs scaled)",
        "  Model   fully convolutional 1D U-Net, 3 down / 3 up levels, kernel 9",
        f"  Params  {m['params']:,}",
        "  Output  5 per-sample heatmap channels, one per peak; position = argmax, confidence = max sigmoid",
        f"  Loss    BCEWithLogits against Gaussian targets, sigma = {m['sigma']} samples",
        f"  Data    {m['n_train']} training toys, {m['n_val']} held-out toys, {m['epochs']} epochs, device {m['device']}",
        f"  Runtime {m['wall_seconds']} s total",
        "",
        "RESULTS ON HELD-OUT TOYS",
        f"  all five peaks within 2 samples : {100*m['all_five_within_2']:.1f} % of traces",
        f"  any peak off by > 15 samples    : {100*m['gross_miss_any']:.2f} % of traces",
        "",
        "  peak   median|dt|   mean|dt|   p95|dt|   within 2   gross miss   mean conf",
    ]
    for name in LABELS:                                          # O(5) | one row per peak
        s = pp[name]                                             # - | that peak's numbers
        summary.append(f"  {name:<5}  {s['median_abs_dt']:>9.2f}   {s['mean_abs_dt']:>8.2f}   "
                       f"{s['p95_abs_dt']:>7.1f}   {100*s['frac_within_2']:>7.1f}%   "
                       f"{100*s['gross_miss_rate']:>9.2f}%   {s['mean_conf']:>8.3f}")  # - | formatted row
    summary += [
        "",
        "LIMITS",
        "  - Trained and tested on toys only; real-board performance is UNVERIFIED.",
        "  - The generator's amplitudes, noise level and peak widths are placeholders,",
        "    chosen by hand, not fitted to measured traces.",
        "  - The model assumes exactly five peaks per trace and always returns five.",
        "    A board missing a peak will still get five positions, flagged only by low confidence.",
        "",
        "NEXT STEP WITH REAL DATA",
        "  1. Fit the generator config to measured traces (amplitudes, noise, widths, floor steps).",
        "  2. Hand-label a few hundred real traces, especially the ones the current finder gets wrong.",
        "  3. Fine-tune on real labels and report the same table on real held-out traces.",
        "",
        "GENERATOR CONFIG USED",
    ]
    for k, v in asdict(cfg).items():                             # O(C) | dump every knob
        summary.append(f"  {k:<18} {v}")                         # - | one line per setting

    with PdfPages(OUT_PDF) as pdf:                               # - | open the PDF
        page_text(pdf, "ML peak finding for VFE-board QC traces", summary)  # - | method and results
        page_gallery(pdf, cfg)                                   # - | toy examples
        page_predictions(pdf, d["X"], d["P"][:12], model)        # - | predictions
        page_errors(pdf, d["dt"], d["conf"])                     # - | error distributions
        page_training(pdf, m["history"])                         # - | curves
        try:                                                     # - | stage-2 page only if it has been run
            with open("outputs/metrics_hybrid.json") as fh:      # - | hybrid metrics
                hm = json.load(fh)                               # - | parsed
            with open("outputs/ablation.json") as fh:            # - | decoder ablation
                ab = json.load(fh)                               # - | parsed
            he = np.load("outputs/hybrid_eval.npz")              # - | hybrid predictions
            page_compare(pdf, hm, ab, he["cnn"] - he["P"], he["hyb"] - he["P"])  # - | comparison page
        except FileNotFoundError as e:                           # - | stage 2 not run yet
            print("skipping stage-2 page:", e)                   # - | say so
    print("wrote", OUT_PDF)                                      # - | confirm
    # Total: time O(N + k n), space O(N)


if __name__ == "__main__":                                       # - | script entry
    main()                                                       # - | build it
