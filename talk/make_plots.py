"""Figures for the dark-beamer talk. Reads only persisted outputs/ + the toy generator.

Writes PNGs into talk/plots/. All panels use the deck's navy canvas so they sit on
the slide without a white box around them.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PLOTS = ROOT / "talk" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

NAVY = "#0A1B38"
MID = "#123060"
CYAN = "#00BCEB"
GOLD = "#FFB900"
GREY = "#D2DCEB"
WHITE = "#F0F4FA"
LABELS = ["P0", "P1", "P2", "P3", "P4"]

plt.rcParams.update({
    "figure.facecolor": NAVY, "axes.facecolor": NAVY, "savefig.facecolor": NAVY,
    "text.color": WHITE, "axes.labelcolor": WHITE, "axes.edgecolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY, "grid.color": MID,
    "font.size": 11, "axes.titlesize": 12, "legend.frameon": False,
    "figure.dpi": 150,
})


def save(fig, name):
    p = PLOTS / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(p)


# ---------------------------------------------------------------- 1. anatomy
def fig_anatomy():
    from src.toy_waveforms import ToyConfig, make_waveform
    cfg = ToyConfig()
    # pick a draw with well-separated peaks so the annotations do not collide
    best = None
    for seed in range(400):
        rng = np.random.default_rng(seed)
        sig, pos = make_waveform(rng, cfg)
        if (pos[0] > 90 and np.diff(pos[1:]).min() > 140
                and 25 < pos[1] - pos[0] < 55 and best is None):
            best = (sig, pos, seed)
            break
    sig, pos, seed = best
    t = np.arange(sig.size)

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.plot(t, sig, color=CYAN, lw=1.1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel("sample")
    ax.set_ylabel("voltage (arb.)")

    ip = np.clip(np.round(pos).astype(int), 0, sig.size - 1)
    ax.plot(pos, sig[ip], "v", color=GOLD, ms=9, zorder=5)
    for lab, x, y in zip(LABELS, pos, sig[ip]):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", color=GOLD, fontweight="bold")

    # the pre-P0 spike train
    ax.annotate("spike train\n(not peaks)", (pos[0] * 0.45, sig[: ip[0]].max()),
                textcoords="offset points", xytext=(-5, 46), ha="center",
                color=GREY, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))

    # P0 and P1 are a close pair
    ymark = max(sig[ip[0]], sig[ip[1]]) + 0.26
    ax.annotate("", xy=(pos[0], ymark), xytext=(pos[1], ymark),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.0))
    ax.text(pos[1] + 14, ymark, f"P0 and P1 are a pair\n{pos[1]-pos[0]:.0f} samples apart",
            ha="left", va="center", color=GOLD, fontsize=9)

    # the floor between P1 and P2 versus the starting floor
    f0 = np.median(sig[: max(ip[0] - 20, 5)])
    mid12 = slice(int(pos[1]) + 40, int(pos[2]) - 10)
    f12 = np.median(sig[mid12])
    ax.axhline(f0, color=GREY, ls=":", lw=0.9)
    ax.axhline(f12, color=GOLD, ls=":", lw=0.9)
    ax.annotate("", xy=(sig.size * 0.97, f12), xytext=(sig.size * 0.97, f0),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.0))
    ax.text(sig.size * 0.95, (f0 + f12) / 2, "floor never\nreturns to start",
            ha="right", va="center", color=GOLD, fontsize=9)

    fig.tight_layout()
    save(fig, "anatomy.png")
    return seed


# ------------------------------------------------------- 2. heatmap mechanism
def fig_heatmap():
    import torch
    from src.model import PeakUNet
    from src.train import normalise

    d = np.load(ROOT / "outputs" / "val_eval.npz")
    X = d["X"]                                  # (12, 1, n) raw traces
    P = d["P"][: X.shape[0]]
    net = PeakUNet()
    net.load_state_dict(torch.load(ROOT / "outputs" / "peak_unet.pt",
                                   map_location="cpu"))
    net.eval()
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.from_numpy(normalise(X)))).numpy()

    i = int(np.argmax([np.diff(p).min() for p in P]))   # most readable of the 12
    sig, pos, pr = X[i, 0], P[i], prob[i]
    pred = pr.argmax(axis=-1)
    conf = pr.max(axis=-1)
    t = np.arange(sig.size)

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 4.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.0, 1.4], hspace=0.08))
    axes[0].plot(t, sig, color=CYAN, lw=1.0)
    for x in pred:
        axes[0].axvline(x, color=GOLD, ls="-", lw=2.0, alpha=0.9, zorder=2)
    for x in pos:
        axes[0].axvline(x, color=WHITE, ls="--", lw=0.9, zorder=3)
    axes[0].set_ylabel("voltage")
    axes[0].grid(alpha=0.2, lw=0.5)
    axes[0].plot([], [], color=WHITE, ls="--", lw=0.9, label="truth")
    axes[0].plot([], [], color=GOLD, lw=2.0, label="network")
    axes[0].legend(loc="upper left", fontsize=9, labelcolor=WHITE)

    cols = [CYAN, GOLD, "#7FD6F2", "#FFD976", WHITE]
    for k in range(5):
        axes[1].plot(t, pr[k], color=cols[k], lw=1.0,
                     label=f"{LABELS[k]}  conf {conf[k]:.2f}")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("channel output")
    axes[1].set_xlabel("sample")
    axes[1].grid(alpha=0.2, lw=0.5)
    axes[1].legend(ncol=5, fontsize=8, loc="upper center", labelcolor=WHITE)
    save(fig, "heatmap_demo.png")


# ------------------------------------------------------------- 3. per-peak
def fig_perpeak():
    d = np.abs(np.load(ROOT / "outputs" / "val_eval.npz")["dt"])
    w2 = (d <= 2).mean(0)
    w4 = (d <= 4).mean(0)
    x = np.arange(5)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    for ax, v, col, lab in ((axes[0], w2, GOLD, "within 2 samples  [%]"),
                            (axes[1], w4, CYAN, "within 4 samples  [%]")):
        ax.bar(x, 100 * v, color=col, width=0.62)
        ax.set_xticks(x, LABELS)
        ax.set_ylabel(lab)
        ax.set_ylim(85, 100)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        for xi, vv in zip(x, v):
            ax.text(xi, 100 * vv + 0.2, f"{100*vv:.1f}", ha="center",
                    color=WHITE, fontsize=9)
    fig.tight_layout()
    save(fig, "perpeak.png")


# --------------------------------------------------------------- 4. methods
def fig_methods():
    a = json.load(open(ROOT / "outputs" / "ablation.json"))
    order = ["cnn_only", "dp_monotone", "dp_plain", "greedy"]
    nice = ["network alone", "scan + ranker\n(ordered decode)",
            "scan + ranker\n(plain decode)", "scan + ranker\n(greedy)"]
    val = [100 * a["variants"][k]["all_five_within_2"] for k in order]
    cols = [GOLD] + [CYAN] * 3

    sweep = a["scan_only_sweep"]
    prom = [s["prominence"] for s in sweep]
    rec = [100 * s["recall_within_3"] for s in sweep]
    nc = [s["cands_per_trace"] for s in sweep]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    axes[0].bar(np.arange(4), val, color=cols, width=0.6)
    axes[0].set_xticks(np.arange(4), nice, fontsize=8)
    axes[0].set_ylabel(r"all five within 2 samples  [%]")
    axes[0].set_ylim(80, 92)
    axes[0].grid(axis="y", alpha=0.25, lw=0.5)
    for xi, v in zip(np.arange(4), val):
        axes[0].text(xi, v + 0.15, f"{v:.1f}", ha="center", color=WHITE, fontsize=9)

    axes[1].plot(prom, rec, "o-", color=CYAN, lw=1.2)
    axes[1].set_xscale("log")
    axes[1].set_xticks(prom)
    axes[1].set_xticklabels([f"{p:g}" for p in prom])
    axes[1].minorticks_off()
    axes[1].set_xlabel("scan prominence threshold")
    axes[1].set_ylabel(r"true peak in candidates  [%]")
    axes[1].grid(alpha=0.25, lw=0.5)
    for p, r, n in zip(prom, rec, nc):
        axes[1].annotate(f"{n:.0f} cand.", (p, r), textcoords="offset points",
                         xytext=(0, -14), ha="center", color=GREY, fontsize=8)
    axes[1].set_ylim(58, 102)
    fig.tight_layout()
    save(fig, "methods.png")


# ------------------------------------------------- 5. gallery (backup slide)
def fig_gallery():
    from src.toy_waveforms import ToyConfig, make_waveform
    cfg = ToyConfig()
    rng = np.random.default_rng(777)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 4.2))
    for ax in axes.ravel():
        sig, pos = make_waveform(rng, cfg)
        ip = np.clip(np.round(pos).astype(int), 0, sig.size - 1)
        ax.plot(np.arange(sig.size), sig, color=CYAN, lw=0.7)
        ax.plot(pos, sig[ip], "v", color=GOLD, ms=5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MID)
    fig.suptitle("six draws from the generator, same settings", color=GREY,
                 fontsize=10)
    fig.tight_layout()
    save(fig, "gallery.png")


# ------------------------------------------- 6. where the failures are (backup)
def fig_p0():
    d = np.load(ROOT / "outputs" / "val_eval.npz")
    dt = np.abs(d["dt"])
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    bins = np.linspace(0, 20, 61)
    ax.hist(np.clip(dt[:, 0], 0, 20), bins=bins, color=GOLD, alpha=0.85,
            label="P0")
    ax.hist(np.clip(dt[:, 1:].ravel(), 0, 20), bins=bins, color=CYAN,
            histtype="step", lw=1.4, label="P1--P4 pooled")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\Delta t|$  [samples]   (overflow at 20)")
    ax.set_ylabel("entries")
    ax.axvline(2, color=GREY, ls=":", lw=1.0)
    ax.legend(labelcolor=WHITE, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, "p0_errors.png")


# ------------------------------------------- 7. snapping to the apex (backup)
def fig_snap():
    sw = json.load(open(ROOT / "outputs" / "refine_sweep.json"))
    gd = json.load(open(ROOT / "outputs" / "refine_gated.json"))
    base = sw["baseline"]["all_five_within_2"]
    rows = [r for r in sw["sweep"] if r["smooth"] == 1 and r["window"] == 3]
    rows.sort(key=lambda r: r["snap_rate"])
    x = [100 * r["snap_rate"] for r in rows]
    y = [100 * r["all_five_within_2"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    axes[0].plot(x, y, "o-", color=CYAN, lw=1.3)
    axes[0].axhline(100 * base, color=GOLD, ls="--", lw=1.1)
    axes[0].set_ylim(100 * base - 4, 100 * base + 4)
    axes[0].text(0.98, 0.78, "no snapping", color=GOLD, fontsize=9, ha="right",
                 transform=axes[0].transAxes)
    axes[0].text(0.5, 0.06, "every setting is inside the statistical error",
                 color=GREY, fontsize=8, ha="center",
                 transform=axes[0].transAxes)
    axes[0].set_xlabel("peaks moved by the snap  [%]")
    axes[0].set_ylabel("all five within 2 samples  [%]")
    axes[0].grid(alpha=0.25, lw=0.5)

    cen, ape = gd["centre_truth"], gd["apex_truth"]
    groups = ["network\nalone", "always snap\nto the apex"]
    vcen = [100 * cen["network_only"]["all_five_within_2"],
            100 * cen["always_snap_w5"]["all_five_within_2"]]
    vape = [100 * ape["network_only"]["all_five_within_2"],
            100 * ape["always_snap_w5"]["all_five_within_2"]]
    xi = np.arange(2)
    axes[1].bar(xi - 0.19, vcen, width=0.36, color=CYAN, label="scored vs clean apex")
    axes[1].bar(xi + 0.19, vape, width=0.36, color=GOLD, label="scored vs noisy-trace apex")
    axes[1].set_xticks(xi, groups, fontsize=9)
    axes[1].set_ylabel("all five within 2 samples  [%]")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8, labelcolor=WHITE)
    axes[1].grid(axis="y", alpha=0.25, lw=0.5)
    for xx, v in zip(xi - 0.19, vcen):
        axes[1].text(xx, v + 1.5, f"{v:.0f}", ha="center", color=WHITE, fontsize=9)
    for xx, v in zip(xi + 0.19, vape):
        axes[1].text(xx, v + 1.5, f"{v:.0f}", ha="center", color=WHITE, fontsize=9)
    fig.tight_layout()
    save(fig, "snap.png")


# --------------------------------------- 8. how precise can an apex be (main)
def fig_precision():
    amb = json.load(open(ROOT / "outputs" / "apex_ambiguity.json"))
    net = json.load(open(ROOT / "outputs" / "metrics.json"))["all_five_within_2"]
    names = ["highest sample,\nraw trace", "highest sample,\nsmoothed", "network"]
    vals = [100 * amb["oracle"]["1"]["all_five_within_2"],
            100 * amb["oracle"]["3"]["all_five_within_2"], 100 * net]

    sweep_dir = ROOT / "diagnostics" / "2026-11-10_sigma_sweep"
    runs = [("now", "4 / 25", ROOT / "outputs" / "metrics.json")]
    for tag, desc in (("A", "1.5 / 25"), ("B", "2 / 25"), ("C", "3 / 25"),
                      ("D", "4 / 60"), ("E", "2 / 60"), ("F", "2 / 60\n20k")):
        runs.append((tag, desc, sweep_dir / tag / "metrics.json"))
    runs = [r for r in runs if r[2].exists()]
    sv = [100 * json.load(open(r[2]))["all_five_within_2"] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7),
                             gridspec_kw=dict(width_ratios=[1, 1.5]))
    axes[0].bar(np.arange(3), vals, color=[GREY, GREY, GOLD], width=0.6)
    axes[0].set_xticks(np.arange(3), names, fontsize=8)
    axes[0].set_ylabel("all five within 2 samples  [%]")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25, lw=0.5)
    for xi, v in enumerate(vals):
        axes[0].text(xi, v + 1.5, f"{v:.0f}", ha="center", color=WHITE, fontsize=9)
    axes[0].text(0.5, 0.94, "readers told where each peak is", ha="center",
                 transform=axes[0].transAxes, color=GREY, fontsize=8)

    x = np.arange(len(runs))
    axes[1].bar(x, sv, color=[GOLD] + [CYAN] * (len(runs) - 1), width=0.6)
    axes[1].set_xticks(x, [f"{r[0]}\n{r[1]}" for r in runs], fontsize=7.5)
    axes[1].set_xlabel("run:  target width (samples) / epochs", fontsize=9)
    axes[1].set_ylim(60, 85)
    axes[1].grid(axis="y", alpha=0.25, lw=0.5)
    for xi, v in zip(x, sv):
        axes[1].text(xi, v + 0.3, f"{v:.1f}", ha="center", color=WHITE, fontsize=8)
    fig.tight_layout()
    save(fig, "precision.png")


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = False
    seed = fig_anatomy()
    print(f"anatomy trace seed = {seed}")
    fig_heatmap()
    fig_perpeak()
    fig_methods()
    fig_gallery()
    fig_p0()
    fig_snap()
    fig_precision()
