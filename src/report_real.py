"""ML peak finder vs the existing (manual, hard-coded) finder on real LVR current traces.

  python -m src.report_real <out.pdf>

Reads data/real_shots.npz, outputs/real/{unet,transformer}/ and
outputs/real/sweep*/<tag>[_s<seed>]/metrics.json (model-size page). Every number on the
pages is computed here from those files.
"""
import glob
import json
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PEAKS = ["A1", "B1", "A2", "B2"]                   # P1..P4
C_F, C_M, C_T = "#C0392B", "#1F77B4", "#7F7F7F"    # finder, ML (U-Net), transformer
UNSIGNED = ("Unsigned algorithm steps (not yet signed off): dataset build and the "
            "inrush (P0) rule; run-level split; network training; peak read-out "
            "(argmax of each output channel, 10 us bins).")
A4 = (11.69, 8.27)


def load():
    d = dict(np.load("data/real_shots.npz"))
    m = {}
    for a in ("unet", "transformer"):
        z = np.load(f"outputs/real/{a}/predictions.npz")
        pos = np.full((len(d["X"]), 5), np.nan); conf = np.full((len(d["X"]), 5), np.nan)
        pos[z["val_idx"]] = z["val_pos"]; pos[z["hard_idx"]] = z["hard_pos"]
        conf[z["val_idx"]] = z["val_conf"]; conf[z["hard_idx"]] = z["hard_conf"]
        m[a] = {"pos": pos, "conf": conf, "val": z["val_idx"], "hard": z["hard_idx"],
                "metrics": json.load(open(f"outputs/real/{a}/metrics.json"))}
    return d, m


def ms(d, b):                                       # bin index -> ms from trigger
    return d["t0_ms"][0] + d["dt_ms"][0] * b


def footer(fig):
    fig.text(0.02, 0.012, UNSIGNED, fontsize=6.5, color="#555555")


def page_summary(pdf, d, m):
    v = m["unet"]["val"]; g = d["good"]
    e = {a: np.abs(ms(d, m[a]["pos"][v, 1:]) - d["finder_t_ms"][v]) for a in m}
    n_runs_val = len(set(d["run"][v]))
    bad = d["n_found"] == 3
    rows = []
    for a, name in (("unet", "U-Net"), ("transformer", "Transformer")):
        x = e[a]
        rows.append([name, f"{np.median(x)*1e3:.1f}",
                     f"{100*np.mean(np.all(x <= 0.02, 1)):.1f}",
                     f"{100*np.mean(np.all(x <= 0.05, 1)):.1f}",
                     f"{100*np.mean(np.all(x <= 0.1, 1)):.1f}",
                     f"{100*np.mean(np.any(x > 1.0, 1)):.1f}"])
    fig = plt.figure(figsize=A4)
    fig.text(0.05, 0.93, "ML peak finder vs the existing finder on real LVR current traces",
             fontsize=16, weight="bold")
    body = (
        f"Data. {len(d['X'])} shots from {len(set(d['run']))} four-peak runs of the LVR tester, "
        f"input current (scope C4), 250 001 samples per shot at 200 ns, averaged here into "
        f"10 us bins. On {g.sum()} shots the existing finder reported all four peaks "
        f"(A1, B1, A2, B2); on {bad.sum()} it reported three.\n\n"
        f"What the existing finder reports. For each peak, the time and height of the highest "
        f"raw sample inside the peak (checked against the stored traces: the reported "
        f"amplitude equals the trace at the reported time).\n\n"
        f"What I trained. A 1D U-Net that outputs one heat-map per landmark, trained on the "
        f"finder's times from {m['unet']['metrics']['n_train']} shots in "
        f"{len(set(d['run'])) - n_runs_val - len(set(d['run'][bad]))} runs. Five landmarks: "
        f"P0 the inrush spike (by a simple rule, the finder does not report it), and "
        f"P1-P4 = A1, B1, A2, B2. A transformer variant was trained on the same split for "
        f"comparison.\n\n"
        f"How it is tested. Split by run, never by shot. {len(v)} shots in {n_runs_val} "
        f"runs never seen in training are compared with the finder below. The "
        f"{len(set(d['run'][bad]))} runs containing the {bad.sum()} failure shots are held "
        f"out completely.")
    fig.text(0.05, 0.88, "\n".join(textwrap.fill(p, 135) if p else "" for p in body.split("\n")),
             fontsize=10, va="top", linespacing=1.45)
    ax = fig.add_axes([0.05, 0.22, 0.9, 0.22]); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["held-out runs\nA1, B1, A2, B2", "median\n|ML - finder| [us]",
                                           "all four within\n0.02 ms [%]", "all four within\n0.05 ms [%]",
                                           "all four within\n0.1 ms [%]", "any peak\n> 1 ms off [%]"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2.2)
    for (r, _), cell in t.get_celld().items():
        if r == 0:
            cell.set_height(cell.get_height() * 1.6)
    hard = m["unet"]["pos"][bad, 1]
    fig.text(0.05, 0.17, textwrap.fill(
        f"On the {bad.sum()} failure shots the existing finder missed A1 every time and "
        f"shifted the remaining labels. The ML places all five landmarks on every one of them "
        f"(pages 4 and 6); its A1 there has no reference to compare with, so that part is a "
        f"visual and consistency check, not a measured error.", 135), fontsize=10, va="top")
    footer(fig); pdf.savefig(fig); plt.close(fig)
    return rows


def page_agreement(pdf, d, m):
    v = m["unet"]["val"]
    fig, axes = plt.subplots(2, 3, figsize=A4)
    for k in range(4):
        ax = axes.flat[k]
        x = (ms(d, m["unet"]["pos"][v, k + 1]) - d["finder_t_ms"][v, k]) * 1e3
        ax.hist(np.clip(x, -60, 60), bins=np.arange(-60, 61, 2.5), color=C_M)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{PEAKS[k]}:  ML - finder", fontsize=10)
        ax.set_xlabel("time difference [us]   (clipped at +-60)")
        ax.set_yscale("log"); ax.set_ylabel("shots")
        ax.text(0.97, 0.92, f"median {np.median(x):+.1f} us\nspread (IQR) "
                f"{np.subtract(*np.percentile(x, [75, 25])):.1f} us",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax = axes.flat[4]
    tol = np.linspace(0, 0.15, 151)
    for a, c, name in (("unet", C_M, "U-Net"), ("transformer", C_T, "Transformer")):
        e = np.abs(ms(d, m[a]["pos"][v, 1:]) - d["finder_t_ms"][v])
        ax.plot(tol * 1e3, [100 * np.mean(np.all(e <= t, 1)) for t in tol], color=c, label=name)
    ax.set_xlabel("tolerance [us]"); ax.set_ylabel("shots with all four within it [%]")
    ax.axvline(10, color="#999999", ls=":", lw=0.8); ax.text(11, 5, "one 10 us bin", fontsize=7)
    ax.set_ylim(0, 101); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes.flat[5].axis("off")
    axes.flat[5].text(0, 0.9, textwrap.fill(
        f"{len(v)} shots from {len(set(d['run'][v]))} held-out runs. The ML reads positions on a "
        f"10 us grid; the finder uses the raw 200 ns samples, so differences below about 5 us "
        f"are the grid, not disagreement.", 48), fontsize=9, va="top")
    fig.suptitle("Agreement with the existing finder on runs the ML never saw", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96)); footer(fig); pdf.savefig(fig); plt.close(fig)


def draw_shot(ax, d, m, i, finder_marks=True):
    tm = ms(d, np.arange(d["X"].shape[-1]))
    w = (tm > 15) & (tm < 27)
    y = d["X"][i, 0]
    ax.plot(tm[w], y[w], color="#222222", lw=0.8)
    if finder_marks:
        for t, lab in zip(d["finder_t_ms"][i], str(d["finder_lab"][i]).split(",")):
            if np.isfinite(t):
                b = int(np.argmin(np.abs(tm - t)))
                ax.plot(t, y[b], "x", color=C_F, ms=9, mew=2)
                ax.annotate(lab, (t, y[b]), xytext=(0, 9), textcoords="offset points",
                            ha="center", color=C_F, fontsize=7)
    for k in range(1, 5):
        b = int(m["unet"]["pos"][i, k])
        ax.plot(tm[b], y[b], "o", mfc="none", mec=C_M, ms=10, mew=1.6)
        ax.annotate(PEAKS[k - 1], (tm[b], y[b]), xytext=(0, -15), textcoords="offset points",
                    ha="center", color=C_M, fontsize=7)
    ax.set_title(f"{d['run'][i]}   {d['vin'][i]}   shot {d['shot'][i]}", fontsize=8, loc="left")
    ax.set_ylabel("I [A]")


def page_examples(pdf, d, m, idx, title, note):
    fig, axes = plt.subplots(len(idx), 1, figsize=A4, sharex=True)
    for ax, i in zip(np.atleast_1d(axes), idx):
        draw_shot(ax, d, m, i)
    np.atleast_1d(axes)[-1].set_xlabel("time from trigger [ms]")
    fig.suptitle(title, fontsize=13)
    fig.text(0.5, 0.935, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93)); footer(fig); pdf.savefig(fig); plt.close(fig)


def page_repeatability(pdf, d, m):
    v = m["unet"]["val"]
    key = np.array([f"{r}|{x}" for r, x in zip(d["run"][v], d["vin"][v])])
    fig, axes = plt.subplots(1, 4, figsize=A4)
    out = []
    for k in range(4):
        sf, sm = [], []
        for g in np.unique(key):
            sel = v[key == g]
            if len(sel) < 5:
                continue
            sf.append(np.std(d["finder_t_ms"][sel, k]) * 1e3)
            sm.append(np.std(ms(d, m["unet"]["pos"][sel, k + 1])) * 1e3)
        sf, sm = np.array(sf), np.array(sm)
        ax = axes[k]; hi = max(sf.max(), sm.max()) * 1.1
        ax.plot([0, hi], [0, hi], color="#999999", lw=0.8)
        ax.scatter(sf, sm, s=12, color=C_M)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_xlabel("finder spread [us]"); ax.set_title(PEAKS[k], fontsize=11)
        if k == 0:
            ax.set_ylabel("ML spread [us]")
        ax.text(0.04, 0.96, f"median\nfinder {np.median(sf):.1f}\nML {np.median(sm):.1f}",
                transform=ax.transAxes, va="top", fontsize=8)
        out.append((np.median(sf), np.median(sm), len(sf)))
    fig.suptitle("Shot-to-shot repeatability: spread of the reported time within one run at one "
                 "voltage (held-out runs)", fontsize=12)
    fig.text(0.5, 0.06, textwrap.fill(
        f"Each point is one run at one voltage ({out[0][2]} groups of at least 5 shots); "
        f"standard deviation of the time over its shots. Below the line, the ML is more "
        f"repeatable than the finder. This needs no labels. The ML's 10 us grid limits how "
        f"small its spread can look.", 150), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.92)); footer(fig); pdf.savefig(fig); plt.close(fig)
    return out


def page_confidence(pdf, d, m):
    v, bad = m["unet"]["val"], np.flatnonzero(d["n_found"] == 3)
    fig, axes = plt.subplots(1, 2, figsize=A4)
    for k, c in zip(range(1, 5), ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"]):
        axes[0].hist(m["unet"]["conf"][v, k], bins=np.linspace(0, 1, 41), histtype="step",
                     color=c, label=PEAKS[k - 1], lw=1.4)
    axes[0].set_yscale("log"); axes[0].set_xlabel("ML confidence (peak height of the channel)")
    axes[0].set_ylabel("shots"); axes[0].legend(); axes[0].set_title("held-out clean runs")
    ca_clean = m["unet"]["conf"][v, 1]; ca_bad = m["unet"]["conf"][bad, 1]
    axes[1].hist(ca_clean, bins=np.linspace(0, 1, 41), density=True, alpha=0.6, color=C_M,
                 label=f"A1, held-out clean shots (n={len(v)})")
    axes[1].hist(ca_bad, bins=np.linspace(0, 1, 41), density=True, alpha=0.6, color=C_F,
                 label=f"A1, shots the finder failed (n={len(bad)})")
    axes[1].set_xlabel("ML confidence on A1"); axes[1].set_ylabel("density"); axes[1].legend()
    axes[1].set_title("A1: clean vs finder-failure shots")
    fig.suptitle("ML confidence: the finder's failure shots look ordinary to the ML", fontsize=13)
    fig.text(0.5, 0.05, f"A1 confidence: median {np.median(ca_clean):.3f} on clean shots, "
             f"{np.median(ca_bad):.3f} on the finder's failures; lowest on failures "
             f"{ca_bad.min():.3f}.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94)); footer(fig); pdf.savefig(fig); plt.close(fig)
    return np.median(ca_clean), np.median(ca_bad), ca_bad.min()


def sweep_runs():                                   # outputs/real/sweep*/<tag>[_s<seed>]/metrics.json
    runs = {}
    for p in sorted(glob.glob("outputs/real/sweep*/*/metrics.json")):
        tag = p.split("/")[-2].split("_s")[0]
        runs.setdefault(tag, []).append(json.load(open(p)))
    return runs


def page_sweep(pdf, runs):
    def name(r):
        return (f"U-Net base {r['base']}, {r['levels']} levels" if r["arch"] == "unet"
                else f"Transformer d={r['d']}, {r['tdepth']} layers")

    def span(x, f="%.1f"):
        x = np.asarray(x)
        return f % x.mean() if len(x) == 1 else f"{f % x.mean()} ({f % x.min()}-{f % x.max()})"

    order = sorted(runs, key=lambda t: runs[t][0]["params"])
    rows = []
    for t in order:
        rs = runs[t]
        rows.append([name(rs[0]), f"{rs[0]['params']/1e6:.2f}", str(len(rs)),
                     span([r["wall_s"] / 60 for r in rs]),
                     span([100 * r["val"]["all_five_within_0.02ms"] for r in rs]),
                     span([100 * r["val"]["all_five_within_0.05ms"] for r in rs]),
                     span([100 * r["val_subbin"]["all_five_within_0.02ms"] for r in rs]),
                     span([1e3 * r["val_subbin"]["per_peak"]["P1"]["median_ms"] for r in rs]),
                     span([100 * r["val"]["gross_any"] for r in rs])])
    fig = plt.figure(figsize=A4)
    fig.text(0.05, 0.93, "Model size: does a bigger network agree better?", fontsize=16,
             weight="bold")
    ax = fig.add_axes([0.03, 0.5, 0.94, 0.38]); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=[
        "model", "parameters\n[M]", "seeds", "training\n[min]",
        "all five within\n0.02 ms [%]\nargmax", "all five within\n0.05 ms [%]\nargmax",
        "all five within\n0.02 ms [%]\nsub-bin", "A1 median\nerror [us]\nsub-bin",
        "any landmark\n> 1 ms off [%]"], loc="upper center", cellLoc="center",
        colWidths=[0.2] + [0.1] * 8)
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.9)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_height(cell.get_height() * 1.9)
    ax2 = fig.add_axes([0.08, 0.1, 0.42, 0.3])
    for t in order:
        rs = runs[t]; c = C_M if rs[0]["arch"] == "unet" else C_T
        ax2.scatter([r["params"] / 1e6 for r in rs],
                    [100 * r["val_subbin"]["all_five_within_0.02ms"] for r in rs],
                    color=c, s=22)
    ax2.scatter([], [], color=C_M, label="U-Net"); ax2.scatter([], [], color=C_T, label="Transformer")
    ax2.set_xscale("log"); ax2.set_xlabel("parameters [M]")
    ax2.set_ylabel("all five within 0.02 ms [%], sub-bin"); ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    r0 = runs[order[0]][0]
    fig.text(0.56, 0.4, textwrap.fill(
        f"Every model is trained on the same run split ({r0['n_train']} training shots) for "
        f"{r0['epochs']} epochs and scored on the same {r0['n_val']} held-out shots, against the "
        f"training labels: P0 by rule, A1-B2 = the existing finder's times. Unlike pages 1-2, "
        f"these columns include P0. Where a model has more than one seed, the cell shows the "
        f"mean and, in brackets, the lowest and highest seed. Sub-bin = parabola through the "
        f"three highest bins of each channel instead of the highest bin alone. Each point in "
        f"the plot is one training run.", 62), fontsize=9, va="top")
    fig.text(0.02, 0.03, "The sub-bin read-out is also an unsigned step.", fontsize=6.5,
             color="#555555")
    footer(fig); pdf.savefig(fig); plt.close(fig)
    return rows


def main():
    out = sys.argv[1]
    d, m = load()
    v = m["unet"]["val"]; bad = np.flatnonzero(d["n_found"] == 3)
    rng = np.random.default_rng(3)
    clean = rng.choice(v, 4, replace=False)
    with PdfPages(out) as pdf:
        rows = page_summary(pdf, d, m)
        page_agreement(pdf, d, m)
        page_examples(pdf, d, m, clean, "Held-out clean shots: both finders",
                      "red x = existing finder (its labels)    blue o = ML")
        page_examples(pdf, d, m, bad[[0, 6, 11, 14, 16, 21]],
                      "Shots where the existing finder found only three peaks",
                      "red x = existing finder (labels P1-P3: A1 missed, the rest shifted)    "
                      "blue o = ML")
        rep = page_repeatability(pdf, d, m)
        conf = page_confidence(pdf, d, m)
        sw = page_sweep(pdf, sweep_runs())
        info = pdf.infodict(); info["Title"] = "ML vs existing peak finder, LVR traces"
        info["Author"] = "Santanu Mahata"
    print(out)
    for r in rows:
        print("  ", r)
    print("   repeatability median spread finder/ML [us]:",
          [f"{a:.1f}/{b:.1f}" for a, b, _ in rep])
    print("   A1 confidence clean / failures / min failures: %.3f %.3f %.3f" % conf)
    for r in sw:
        print("   sweep", r)


if __name__ == "__main__":
    main()
