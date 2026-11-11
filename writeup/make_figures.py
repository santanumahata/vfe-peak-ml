"""Figures, tables and number macros for the LaTeX write-up (writeup/src/vfe_peak_ml_writeup.tex).

  python -m writeup.make_figures

Figures -> writeup/plots/real/  (git-ignored): drawn from data/real_shots.npz and outputs/real/.
Every real-data number in the text is a macro written here to writeup/plots/real/numbers.tex.
"""
import json                                                      # - | metrics files
import pathlib                                                   # - | paths

import matplotlib                                                # - | plotting backend
matplotlib.use("Agg")                                            # - | no display
import matplotlib.pyplot as plt                                  # - | figures
import numpy as np                                               # - | arrays

from src.report_real import C_F, C_M, C_T, PEAKS, draw_shot, load, ms, sweep_runs  # - | real-data helpers

ROOT = pathlib.Path(__file__).resolve().parent.parent            # - | repo root
REAL = ROOT / "writeup" / "plots" / "real"                       # - | local, git-ignored
REPORT_MODEL = "outputs/real/sweep/u32l4"                        # - | report model: U-Net 32/4 seed 0
TRANSFORMER = "outputs/real/transformer"                         # - | comparison model
WHITE_PAGE = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
              "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "#333333",
              "xtick.color": "#333333", "ytick.color": "#333333", "grid.color": "#BBBBBB",
              "font.size": 10, "axes.titlesize": 10, "legend.frameon": False, "figure.dpi": 150}


def save(fig, name):                                             # - | one real figure
    path = REAL / name                                           # - | target
    fig.savefig(path, bbox_inches="tight")                       # - | write png
    plt.close(fig)                                               # - | free memory
    print(path)                                                  # - | report path
    # Total: time O(1), space O(1)


def fig_trace(d, m, i):                                          # - | one shot, full and zoomed
    tm = ms(d, np.arange(d["X"].shape[-1]))                      # O(n) | bin centres in ms
    y = d["X"][i, 0]                                             # - | binned current
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), gridspec_kw={"height_ratios": [1, 1.3]})
    ax[0].plot(tm[:5000], y[:5000], color="#222222", lw=0.7)     # O(n) | real bins, padding dropped
    p0 = int(d["P"][i, 0])                                       # - | P0 bin by rule
    ax[0].plot(tm[p0], y[p0], "v", color=C_F, ms=7)              # - | mark P0
    ax[0].annotate("P0", (tm[p0], y[p0]), xytext=(8, 0), textcoords="offset points",
                   color=C_F, fontsize=8, va="center")           # - | label it
    ax[0].axvspan(13.5, 27, color="#1F77B4", alpha=0.08)           # - | zoom region
    ax[0].set_xlabel("time from trigger [ms]"); ax[0].set_ylabel("I [A]")
    draw_shot(ax[1], d, m, i)                                    # - | finder x, ML o, 13.5-27 ms
    ax[1].set_xlabel("time from trigger [ms]")
    fig.tight_layout(); save(fig, "trace.png")
    # Total: time O(n), space O(n)


def fig_training(sweep):                                         # - | validation curves per epoch
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for tag, c in (("u16", "#9EC5E8"), ("u32l4", C_M), ("t256x4", C_T)):  # O(3) | models shown
        h = sweep[tag][0]["history"]                             # - | seed-0 run
        ep = [r["epoch"] for r in h]                             # - | x axis
        ax[0].plot(ep, [r["loss"] for r in h], color=c, label=tag)  # - | training loss
        ax[1].plot(ep, [100 * r["all_five_within_0.02ms"] for r in h], color=c, label=tag)  # - | val metric
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss (BCE)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("held-out: all five within 0.02 ms [%]")
    ax[1].set_ylim(0, 100)
    for a in ax:                                                 # O(2) | both panels
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); save(fig, "training.png")
    # Total: time O(epochs), space O(epochs)


def residuals(d, m, a):                                          # - | ML - finder on held-out shots, us
    v = m[a]["val"]                                              # - | held-out shot indices
    return (ms(d, m[a]["pos"][v, 1:]) - d["finder_t_ms"][v]) * 1e3  # O(4V) | (V, 4) in us
    # Total: time O(V), space O(V)


def fig_residuals(d, m):                                         # - | per-peak histograms + tolerance curve
    r = residuals(d, m, "unet")                                  # - | (V, 4)
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.9), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.3]})
    for k in range(4):                                           # O(4) | one panel per peak
        ax = axes[k]
        ax.hist(np.clip(r[:, k], -60, 60), bins=np.arange(-60, 61, 2.5), color=C_M)  # O(V) | histogram
        ax.set_yscale("log"); ax.set_title(PEAKS[k]); ax.set_xlabel("ML - finder [us]")
        ax.axvline(0, color="k", lw=0.6)
    axes[0].set_ylabel("shots")
    tol = np.linspace(0, 150, 151)                               # - | tolerance scan, us
    for a, c, name in (("unet", C_M, "U-Net 32/4"), ("transformer", C_T, "Transformer")):  # O(2) | models
        e = np.abs(residuals(d, m, a))                           # - | |error|
        axes[4].plot(tol, [100 * np.mean(np.all(e <= t, 1)) for t in tol], color=c, label=name)  # O(151 V) | curve
    axes[4].axvline(10, color="#999999", ls=":", lw=0.8)         # - | one bin
    axes[4].set_xlabel("tolerance [us]"); axes[4].set_ylabel("all four within [%]")
    axes[4].set_ylim(0, 101); axes[4].grid(alpha=0.3); axes[4].legend(fontsize=7)
    fig.tight_layout(); save(fig, "residuals.png")
    # Total: time O(V), space O(V)


def repeatability(d, m):                                         # - | within-group std per peak, us
    v = m["unet"]["val"]                                         # - | held-out shots
    key = np.array([f"{r}|{x}" for r, x in zip(d["run"][v], d["vin"][v])])  # O(V) | run x voltage
    groups = [v[key == g] for g in np.unique(key) if (key == g).sum() >= 5]  # O(V G) | groups >= 5 shots
    sf = np.array([[np.std(d["finder_t_ms"][g, k]) * 1e3 for k in range(4)] for g in groups])  # O(V) | finder
    sm = np.array([[np.std(ms(d, m["unet"]["pos"][g, k + 1])) * 1e3 for k in range(4)] for g in groups])  # O(V) | ML
    return sf, sm
    # Total: time O(V G), space O(G)


def fig_repeatability(sf, sm):                                   # - | finder vs ML spread, per peak
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    for k in range(4):                                           # O(4) | one panel per peak
        ax = axes[k]; hi = max(sf[:, k].max(), sm[:, k].max()) * 1.1
        ax.plot([0, hi], [0, hi], color="#999999", lw=0.8)       # - | equal spread
        ax.scatter(sf[:, k], sm[:, k], s=10, color=C_M)          # - | one point per group
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_title(PEAKS[k]); ax.set_xlabel("finder spread [us]")
    axes[0].set_ylabel("ML spread [us]")
    fig.tight_layout(); save(fig, "repeatability.png")
    # Total: time O(G), space O(1)


def fig_shots(d, m, idx, name):                                  # - | stacked example shots
    fig, axes = plt.subplots(len(idx), 1, figsize=(9, 1.9 * len(idx)), sharex=True)
    for ax, i in zip(axes, idx):                                 # O(len idx) | one shot per row
        draw_shot(ax, d, m, i)                                   # - | finder x, ML o
    axes[-1].set_xlabel("time from trigger [ms]")
    fig.tight_layout(); save(fig, name)
    # Total: time O(len idx * n), space O(n)


def fig_confidence(d, m, bad):                                   # - | confidence distributions
    v = m["unet"]["val"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    bins = np.linspace(0, 1, 41)                                 # - | confidence bins
    for k, c in zip(range(1, 5), ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"]):  # O(4) | per peak
        axes[0].hist(m["unet"]["conf"][v, k], bins=bins, histtype="step", color=c, lw=1.3, label=PEAKS[k - 1])
    axes[0].set_yscale("log"); axes[0].set_xlabel("confidence"); axes[0].set_ylabel("shots")
    axes[0].legend(fontsize=8)
    axes[1].hist(m["unet"]["conf"][v, 1], bins=bins, density=True, alpha=0.6, color=C_M, label="held-out clean")
    axes[1].hist(m["unet"]["conf"][bad, 1], bins=bins, density=True, alpha=0.6, color=C_F, label="finder failures")
    axes[1].set_xlabel("confidence on A1"); axes[1].set_ylabel("density"); axes[1].legend(fontsize=8)
    fig.tight_layout(); save(fig, "confidence.png")
    # Total: time O(V), space O(1)


def fig_sweep(sweep):                                            # - | metric vs parameter count
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for tag, rs in sweep.items():                                # O(configs) | one config at a time
        c = C_M if rs[0]["arch"] == "unet" else C_T              # - | colour by family
        x = [r["params"] / 1e6 for r in rs]                      # - | parameters, M
        y = [100 * r["val_subbin"]["all_five_within_0.02ms"] for r in rs]  # - | one point per seed
        ax.scatter(x, y, color=c, s=20)
        ax.annotate(tag, (x[0], max(y)), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.scatter([], [], color=C_M, label="U-Net"); ax.scatter([], [], color=C_T, label="Transformer")
    ax.set_xscale("log"); ax.set_xlabel("parameters [M]")
    ax.set_ylabel("all five within 0.02 ms [%], sub-bin"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); save(fig, "sweep.png")
    # Total: time O(runs), space O(1)


def pct(x):                                                      # - | fraction -> "92.4"
    return f"{100 * x:.1f}"
    # Total: time O(1), space O(1)


def span(xs, f="%.1f"):                                          # - | "mean (min--max)" over seeds
    x = np.asarray(xs)
    if f % x.min() == f % x.max():                               # - | one seed, or seeds agree when printed
        return f % x.mean()
    return f"{f % x.mean()} ({f % x.min()}--{f % x.max()})"
    # Total: time O(len xs), space O(1)


def perpeak_table(mu, mt):                                       # - | LaTeX rows, per landmark
    rows = []
    variants = (("U-Net 32/4, argmax", mu.get("val")), ("U-Net 32/4, sub-bin", mu.get("val_subbin")),
                ("Transformer, argmax", mt.get("val")), ("Transformer, sub-bin", mt.get("val_subbin")))
    for name, met in [v for v in variants if v[1] is not None]:  # O(4) | variants this run recorded
        for lab in ("P0", "P1", "P2", "P3", "P4"):               # O(5) | landmarks
            p = met["per_peak"][lab]
            rows.append(f"{name} & {lab} & {1e3 * p['median_ms']:.1f} & {pct(p['within_0.02ms'])} & "
                        f"{pct(p['within_0.05ms'])} & {pct(p['within_0.1ms'])} & {pct(p['gross'])} \\\\")
        rows.append("\\midrule")
    return "\n".join(rows[:-1]) + "\n"
    # Total: time O(20), space O(20)


def sweep_table(sweep):                                          # - | LaTeX rows, one per config
    rows = []
    for tag in sorted(sweep, key=lambda t: sweep[t][0]["params"]):  # O(C log C) | by size
        rs = sweep[tag]; r0 = rs[0]
        name = (f"U-Net {r0['base']}/{r0['levels']}" if r0["arch"] == "unet"
                else f"Transformer {r0['d']}$\\times${r0['tdepth']}")
        rows.append(" & ".join([
            name, f"{r0['params'] / 1e6:.2f}", str(len(rs)), span([r["wall_s"] / 60 for r in rs]),
            span([100 * r["val"]["all_five_within_0.02ms"] for r in rs]),
            span([100 * r["val"]["all_five_within_0.05ms"] for r in rs]),
            span([100 * r["val_subbin"]["all_five_within_0.02ms"] for r in rs]),
            span([100 * r["val"]["gross_any"] for r in rs])]) + " \\\\")
    return "\n".join(rows) + "\n"
    # Total: time O(C), space O(C)


def numbers(d, m, mu, mt, sf, sm, bad):                          # - | \newcommand per number
    v = m["unet"]["val"]
    ru, rt = np.abs(residuals(d, m, "unet")), np.abs(residuals(d, m, "transformer"))  # - | |ML - finder|
    tr_runs = len(set(d["run"])) - len(set(d["run"][v])) - len(set(d["run"][bad]))  # - | training runs
    cu, cb = m["unet"]["conf"][v, 1], m["unet"]["conf"][bad, 1]  # - | A1 confidences
    labelled = d["good"] & np.isfinite(d["P"][:, 0])               # - | all five labels known
    n = {"NShots": len(d["X"]), "NRuns": len(set(d["run"])), "NGood": int(d["good"].sum()),
         "NFail": len(bad), "NFailRuns": len(set(d["run"][bad])), "NTrain": mu["n_train"],
         "NTrainRuns": tr_runs, "NVal": mu["n_val"], "NValRuns": len(set(d["run"][v])),
         "NHard": mu["n_hard"], "NHardClean": int(np.sum(labelled[m["unet"]["hard"]])),
         "NNoPzero": int(np.sum(d["good"] & ~labelled)),
         "Vins": ", ".join(sorted(set(d["vin"]), key=lambda s: int(s[:-1]))),
         "FailLabels": ", ".join(sorted(set(d["finder_lab"][bad]))).replace(",", ", "),
         "ParamsUnet": f"{mu['params'] / 1e6:.2f}", "ParamsTrans": f"{mt['params'] / 1e6:.2f}",
         "WallUnet": f"{mu['wall_s'] / 60:.0f}", "Epochs": mu["epochs"], "SigmaBins": f"{mu['sigma_bins']:g}",
         "MedUnet": f"{np.median(ru):.1f}", "MedTrans": f"{np.median(rt):.1f}",
         "TwoUnet": pct(np.mean(np.all(ru <= 20, 1))), "FiveUnet": pct(np.mean(np.all(ru <= 50, 1))),
         "TenUnet": pct(np.mean(np.all(ru <= 100, 1))), "GrossUnet": pct(np.mean(np.any(ru > 1000, 1))),
         "TwoTrans": pct(np.mean(np.all(rt <= 20, 1))), "FiveTrans": pct(np.mean(np.all(rt <= 50, 1))),
         "TenTrans": pct(np.mean(np.all(rt <= 100, 1))), "GrossTrans": pct(np.mean(np.any(rt > 1000, 1))),
         "HardTwo": pct(mu["hard_runs_clean_shots"]["all_five_within_0.02ms"]),
         "HardTen": pct(mu["hard_runs_clean_shots"]["all_five_within_0.1ms"]),
         "RepGroups": len(sf),
         "ConfClean": f"{np.median(cu):.3f}", "ConfFail": f"{np.median(cb):.3f}", "ConfFailMin": f"{cb.min():.3f}"}
    for k, lab in enumerate(["Aone", "Bone", "Atwo", "Btwo"]):   # O(4) | repeatability per peak
        n[f"RepF{lab}"] = f"{np.median(sf[:, k]):.1f}"; n[f"RepM{lab}"] = f"{np.median(sm[:, k]):.1f}"
    lines = [f"\\newcommand{{\\{k}}}{{{x}}}" for k, x in n.items()]  # O(K) | one macro each
    (REAL / "numbers.tex").write_text("\n".join(lines) + "\n")   # - | write
    print(REAL / "numbers.tex")
    # Total: time O(V), space O(K)


def real_figures():                                              # - | everything drawn from board data
    REAL.mkdir(parents=True, exist_ok=True)                      # - | git-ignored folder
    plt.rcParams.update(WHITE_PAGE)                              # - | print style
    d, m = load(REPORT_MODEL)                                    # - | data + U-Net 32/4 + transformer
    mu = json.load(open(f"{REPORT_MODEL}/metrics.json"))         # - | report-model metrics
    mt = json.load(open(f"{TRANSFORMER}/metrics.json"))          # - | transformer metrics
    sweep = sweep_runs()                                         # - | {tag: [metrics per seed]}
    v, bad = m["unet"]["val"], np.flatnonzero(d["n_found"] == 3)  # - | held-out, finder failures
    assert np.all(d["good"][v] & np.isfinite(d["P"][v, 0])), "held-out shot without all five labels"
    assert len(bad) > 0 and np.all(np.isin(bad, m["unet"]["hard"])), "failure shot not in hard set"
    rng = np.random.default_rng(3)                               # - | same draw as report_real
    clean = rng.choice(v, 4, replace=False)                      # - | four example shots
    fig_trace(d, m, clean[0])
    fig_training(sweep)
    fig_residuals(d, m)
    sf, sm = repeatability(d, m)
    fig_repeatability(sf, sm)
    fig_shots(d, m, clean, "clean_shots.png")
    fig_shots(d, m, bad[[0, 6, 11, 14, 16, 21]], "failure_shots.png")
    fig_confidence(d, m, bad)
    fig_sweep(sweep)
    (REAL / "perpeak_table.tex").write_text(perpeak_table(mu, mt)); print(REAL / "perpeak_table.tex")
    (REAL / "sweep_table.tex").write_text(sweep_table(sweep)); print(REAL / "sweep_table.tex")
    numbers(d, m, mu, mt, sf, sm, bad)
    # Total: time O(N n), space O(N n)


if __name__ == "__main__":                                       # - | script entry
    real_figures()                                               # - | local only
