"""Regenerate every figure from ``results/*.json``.

``python -m scripts.make_figures --paper-dir ../../qvk-forecast-paper/figures``
PNGs go to ``docs/figures`` (README) and, with ``--paper-dir``, PDFs go to the manuscript folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from app import config as C

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "figure.dpi": 130,
    "lines.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
RES, OUT = C.RESULTS_DIR, C.FIGURE_DIR
extra_dirs: list[Path] = []
PAL = {"A": "#E69F00", "B": "#0072B2", "C": "#CC79A7"}
MODEL_PAL = {"Persistence": "#999999", "ARIMA (AR(24))": "#0072B2", "XGBoost": "#009E73", "LSTM (2 layers)": "#E69F00",
             "Proposed: ARIMA + VQK": "#D55E00", "ARIMA + VQK (circuit kernel)": "#56B4E9", "ARIMA + RBF kernel": "#F0E442"}
LS = {"ARIMA (AR(24))": (0, (4, 1.5)), "XGBoost": "-", "LSTM (2 layers)": (0, (1, 1)), "Proposed: ARIMA + VQK": "-"}
HATCH = {"Persistence": "", "ARIMA (AR(24))": "//", "XGBoost": "..", "LSTM (2 layers)": "xx", "Proposed: ARIMA + VQK": "\\\\"}


def tag(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


MAIN = ["Persistence", "ARIMA (AR(24))", "XGBoost", "LSTM (2 layers)", "Proposed: ARIMA + VQK"]
RNAME = C.REGION_NAMES


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    for d in extra_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf")
    plt.close(fig)
    print("figure:", name)


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


# ---------------------------------------------------------------------------------------------------------
def fig_architecture():
    """Pipeline schematic: sources -> four stages (with the software engine of each) -> outcomes."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 40.6)
    ax.set_ylim(0, 24)
    ax.axis("off")
    INK, MUTE, EDGE = "#1d2733", "#52606d", "#9fb3c8"

    def card(x, y, w, h, fc, ec=INK, lw=0.8, r=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=2))

    def arrow(x0, y0, x1, y1, c=INK, lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=lw, color=c, zorder=6, shrinkA=0, shrinkB=0))

    def turbine(x, base, h, blade, lw=1.1):
        ax.plot([x, x], [base, base + h], color="#55606e", lw=lw + 0.3, zorder=3)
        for ang in (90, 210, 330):
            r = np.deg2rad(ang + 14)
            ax.plot([x, x + blade * np.cos(r)], [base + h, base + h + blade * np.sin(r)], color="#2b3a4a", lw=lw, zorder=3)
        ax.add_patch(Circle((x, base + h), 0.28, fc="#2b3a4a", ec="none", zorder=4))

    # ---- column headings -----------------------------------------------------------------------------
    ax.text(4.2, 23.2, "SOURCES", fontsize=6.4, weight="bold", color=MUTE, ha="center")
    ax.text(24.6, 23.2, "FOUR-STAGE PIPELINE", fontsize=6.4, weight="bold", color=MUTE, ha="center")

    # ---- sources card: solar | wind ------------------------------------------------------------------
    card(0.2, 12.0, 8.0, 10.6, "#f4f8fc", ec=EDGE)
    ax.add_patch(Circle((2.3, 20.5), 0.8, fc="#F6B73C", ec="none", zorder=3))
    for ang in range(0, 360, 45):
        r = np.deg2rad(ang)
        ax.plot([2.3 + 1.1 * np.cos(r), 2.3 + 1.5 * np.cos(r)], [20.5 + 1.1 * np.sin(r), 20.5 + 1.5 * np.sin(r)], color="#F6B73C", lw=0.9, zorder=3)
    ax.add_patch(Polygon([[0.8, 14.7], [3.9, 14.7], [4.5, 18.3], [1.4, 18.3]], fc="#27508f", ec="#d7e0ee", lw=0.5, zorder=3))
    for t in (1 / 3, 2 / 3):
        ax.plot([0.8 + 3.1 * t, 1.4 + 3.1 * t], [14.7, 18.3], color="#d7e0ee", lw=0.5, zorder=4)
    ax.plot([1.1, 4.2], [16.5, 16.5], color="#d7e0ee", lw=0.5, zorder=4)
    ax.text(2.6, 13.1, "Solar PV", fontsize=6.2, weight="bold", ha="center", color=INK)
    ax.plot([4.4, 4.4], [12.9, 21.7], color="#d7e0ee", lw=0.6, zorder=3)
    turbine(6.6, 14.3, 4.2, 1.5)
    turbine(5.2, 14.3, 2.3, 0.9, lw=0.9)
    ax.text(5.9, 13.1, "Wind", fontsize=6.2, weight="bold", ha="center", color=INK, zorder=5)

    # ---- region strip (the "engine" slot of the sources column) -------------------------------------
    card(0.2, 3.6, 8.0, 7.6, "#ffffff", ec=EDGE, lw=0.7)
    ax.text(4.2, 10.4, "Three regions", fontsize=5.6, ha="center", color=MUTE, style="italic")
    for i, (k, lab, pv) in enumerate((("A", "solar", 0.85), ("B", "wind", 0.03), ("C", "mixed", 0.50))):
        y = 9.0 - i * 1.75
        ax.text(0.7, y, k, fontsize=6.4, weight="bold", color=PAL[k], va="center")
        ax.add_patch(Rectangle((1.5, y - 0.45), 3.4 * pv, 0.9, fc="#F6B73C", ec="none", zorder=3))
        ax.add_patch(Rectangle((1.5 + 3.4 * pv, y - 0.45), 3.4 * (1 - pv), 0.9, fc="#4a7fb5", ec="none", zorder=3))
        ax.text(5.15, y, lab.split("-")[0], fontsize=4.8, va="center", color=INK, zorder=4)
    ax.text(4.2, 3.95, "bar: PV (gold) / wind (blue) share", fontsize=4.2, ha="center", color=MUTE, zorder=4)

    # ---- the four stages ----------------------------------------------------------------------------
    stages = [
        ("1", ["Data"], ["fixed-parameter", "synthetic scenarios", "with calm/stormy", "regimes; no live", "data API"], "#f3ead5", "NumPy / SciPy"),
        ("2", ["Classical", "baselines"], ["persistence, AR(24),", "XGBoost,", "LSTM (2 layers)", "-> point forecast", "and residual"], "#dbe6f4", "NumPy, XGBoost,\nPyTorch (CPU)"),
        ("3", ["Quantum-inspired", "residual corrector"], ["6 qubits, 3 layers,", "36 fixed angles;", "kernel ridge on", "the AR residual"], "#e8def8", "PennyLane\ndefault.qubit / .mixed"),
        ("4", ["Text report"], ["deterministic", "template filled", "from saved", "result JSON", "artifacts"], "#dff0e5", "Python template\n(no language model)"),
    ]
    x0, w, gap = 9.3, 6.95, 0.85
    for i, (num, title, lines, fc, engine) in enumerate(stages):
        x = x0 + i * (w + gap)
        card(x, 12.0, w, 10.6, fc)
        ax.add_patch(Circle((x + 1.05, 21.0), 0.7, fc=INK, ec="none", zorder=4))
        ax.text(x + 1.05, 21.0, num, color="white", fontsize=6.6, weight="bold", ha="center", va="center", zorder=5)
        for j, t in enumerate(title):
            ax.text(x + 2.1, 21.0 + (0.5 if len(title) > 1 else 0) - j * 1.1, t, fontsize=5.6, weight="bold", va="center", color=INK)
        ax.text(x + w / 2, 18.7, "\n".join(lines), ha="center", va="top", fontsize=5.5, color="#2a3441", linespacing=1.45)
        card(x, 3.6, w, 7.6, "#ffffff", ec=EDGE, lw=0.7)
        ax.text(x + w / 2, 10.4, "Engine", fontsize=5.6, ha="center", color=MUTE, style="italic")
        ax.text(x + w / 2, 6.9, engine, fontsize=5.5, ha="center", va="center", weight="bold", color=INK, linespacing=1.35)
        if i < 3:
            arrow(x + w + 0.05, 17.2, x + w + gap - 0.05, 17.2)
    arrow(8.25, 17.2, x0 - 0.05, 17.2)

    # ---- outcomes -----------------------------------------------------------------------------------
    ow, og = 9.45, 0.85
    outs = ["1-h-ahead capacity-factor\nforecast, three regions", "calm vs stormy regime\nseparability (Fisher ratio)",
            "cross-region transfer, 3x3,\nzero-shot and fine-tuned", "shot- and depolarising-noise\nstudy of the circuit kernel"]
    for i, t in enumerate(outs):
        x = 0.2 + i * (ow + og)
        card(x, 0.0, ow, 2.45, "#fff4e0", ec="#d9a441")
        ax.text(x + ow / 2, 1.22, t, fontsize=5.0, ha="center", va="center", color=INK, linespacing=1.3)
    ax.text(20.3, 2.95, "OUTCOMES   (CPU only; regenerated by one command)", fontsize=5.4, color=MUTE, ha="center", weight="bold")
    save(fig, "fig01_architecture")


def fig_regions(sim):
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.6), gridspec_kw={"width_ratios": [1.5, 1.5, 1.5, 1.0]})
    for ax, k in zip(axes[:3], C.REGIONS):
        y = sim["regions"][k]["week"]["y"]
        ax.plot(np.arange(len(y)) / 24, y, color=PAL[k], lw=1.2)
        ax.text(0.03, 0.93, f"{k}: {RNAME[k]}", transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_xlabel("day")
        tag(ax, "abc"[C.REGIONS.index(k)])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("capacity factor")
    for k in C.REGIONS:
        h = np.array(sim["regions"][k]["hist"])
        axes[3].plot(np.linspace(0.025, 0.975, len(h)), h, color=PAL[k], lw=1.4, label=k)
    tag(axes[3], "d")
    axes[3].set_xlabel("capacity factor")
    axes[3].set_ylabel("density")
    axes[3].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig02_regions")


def fig_forecast(fc):
    t = fc["week_traces"]["C"]
    n = 168
    x = np.arange(n) / 24
    y = np.array(t["y"])
    st = np.array(t["stormy"]).astype(bool)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [2.2, 1]})
    s_ = st[:n].astype(int)
    edges = np.where(np.diff(np.r_[0, s_, 0]) != 0)[0]
    for i0, i1 in edges.reshape(-1, 2):
        ax.axvspan(i0 / 24, i1 / 24, color="#BBBBBB", alpha=0.35, lw=0)
    ax.plot(x, y[:n], color="black", lw=1.5, label="observed")
    for m in ("ARIMA (AR(24))", "XGBoost", "LSTM (2 layers)", "Proposed: ARIMA + VQK"):
        ax.plot(x, t["pred"][m][:n], color=MODEL_PAL[m], ls=LS[m], lw=0.9, label=m.replace("Proposed: ARIMA + VQK", "AR + VQK"))
    ax.set_xlabel("day of the test window")
    ax.set_ylabel("capacity factor")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1.3)
    ax.legend(ncol=3, frameon=False, loc="upper left", columnspacing=1.0, handlelength=1.8)
    tag(ax, "a")
    names = ["Persistence", "ARIMA (AR(24))", "XGBoost", "LSTM (2 layers)", "Proposed: ARIMA + VQK"]
    w = 0.38
    for j, (lab, mask) in enumerate((("calm", ~st), ("stormy", st))):
        vals = []
        for m in names:
            pr = np.array(t["pred"][m]) if m != "Persistence" else np.r_[y[0], y[:-1]]
            vals.append(float(np.sqrt(np.mean((pr[mask] - y[mask]) ** 2))))
        bx.bar(np.arange(len(names)) + (j - 0.5) * w, vals, w, color=["#888888" if j == 0 else "#222222"] * len(names), alpha=0.55 if j == 0 else 0.9, label=lab)
    bx.set_xticks(range(len(names)), ["Pers.", "AR", "XGB", "LSTM", "AR+VQK"], rotation=40, ha="right")
    bx.set_ylabel("RMSE (capacity factor)")
    bx.legend(frameon=False)
    tag(bx, "b")
    fig.tight_layout()
    save(fig, "fig03_forecast")


def fig_kernel(kn):
    run = kn["regions"]["C"][0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, key, title in ((axes[0], "rbf_tuned", "Tuned RBF kernel"), (axes[1], "quantum_circuit", "Quantum circuit kernel")):
        z = np.array(run[key]["embedding"])
        lab = np.array(run[key]["labels"])
        ax.scatter(z[lab == 0, 0], z[lab == 0, 1], s=7, color="#0072B2", alpha=0.5, label="calm")
        ax.scatter(z[lab == 1, 0], z[lab == 1, 1], s=7, color="#D55E00", alpha=0.5, label="stormy")
        ax.set_title(title, fontsize=8)
        ax.text(0.03, 0.05, f"FDR (PC1) = {run[key]['fdr_pc1']:.2f}", transform=ax.transAxes, fontsize=7.5)
        tag(ax, "ab"[(ax is axes[1])])
        ax.set_xlabel("kernel PC 1")
    axes[0].set_ylabel("kernel PC 2")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig04_kernel")


def fig_circuit():
    from app.quantum.circuit import draw_circuit
    f = draw_circuit()
    f.set_size_inches(9.0, 3.6)
    save(f, "fig05_circuit")


def fig_transfer(fc):
    models = ["ARIMA (AR(24))", "XGBoost", "LSTM (2 layers)", "Proposed: ARIMA + VQK"]
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.2))
    for r, key in enumerate(("zero_shot", "fine_tune")):
        for c, m in enumerate(models):
            M = np.full((3, 3), np.nan)
            for i, s in enumerate(C.REGIONS):
                for j, t in enumerate(C.REGIONS):
                    v = fc["transfer"][f"{s}->{t}"][m][key]
                    if v:
                        M[i, j] = np.mean(v)
            ax = axes[r, c]
            im = ax.imshow(M, cmap="viridis_r", vmin=0.04, vmax=0.17)
            ax.grid(False)
            ax.set_xticks(range(3), C.REGIONS)
            ax.set_yticks(range(3), C.REGIONS)
            for i in range(3):
                for j in range(3):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > 0.1 else "black")
            if r == 0:
                ax.set_title(m.replace("Proposed: ARIMA + VQK", "AR + VQK").replace(" (2 layers)", "").replace("ARIMA (AR(24))", "AR(24)"), fontsize=8)
            if c == 0:
                ax.set_ylabel(("zero-shot" if key == "zero_shot" else "fine-tune") + "\nsource")
            if r == 1:
                ax.set_xlabel("target")
    fig.colorbar(im, ax=axes, shrink=0.7, label="NRMSE")
    save(fig, "fig06_transfer")


def fig_noise(nz):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    shots = sorted(nz["shots"], key=int)
    m = [ms(nz["shots"][s]) for s in shots]
    axes[0].errorbar([int(s) for s in shots], [a for a, _ in m], [b for _, b in m], marker="o", color="#D55E00", capsize=2)
    axes[0].axhline(np.mean(nz["exact"]), color="#555", ls="--", label="exact")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("shots per kernel entry")
    axes[0].set_ylabel("Fisher ratio (PC1)")
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1, loc="lower right")
    tag(axes[0], "a"); tag(axes[1], "b")
    ps = sorted(nz["depolarising"], key=float)
    m = [ms(nz["depolarising"][p]) for p in ps]
    axes[1].errorbar([float(p) * 100 for p in ps], [a for a, _ in m], [b for _, b in m], marker="o", color="#0072B2", capsize=2)
    axes[1].set_xlabel("depolarising probability per qubit per layer (%)")
    fig.tight_layout()
    save(fig, "fig07_noise")


def fig_sim_checks(sim):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for k in C.REGIONS:
        axes[0].plot(range(24), sim["regions"][k]["diurnal"], color=PAL[k], label=f"{k}")
        axes[1].plot(range(1, 97), sim["regions"][k]["acf"], color=PAL[k], label=f"{k}")
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("mean capacity factor")
    axes[1].set_xlabel("lag (h)")
    axes[1].set_ylabel("autocorrelation")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig08_sim_checks")


def fig_importance(fc):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True)
    for ax, k in zip(axes, C.REGIONS):
        imp = fc["importance"][k]
        x = np.arange(len(imp["covariates"]))
        ax.bar(x - 0.2, imp["xgboost"], 0.4, color=MODEL_PAL["XGBoost"], label="XGBoost")
        ax.bar(x + 0.2, imp["proposed_forecast"], 0.4, color=MODEL_PAL["Proposed: ARIMA + VQK"], label="Proposed")
        ax.set_xticks(x, ["wind", "gust", "press.", "temp.", "GHI", "dp/3h"], rotation=45, fontsize=7)
        ax.set_title(f"{k}: {RNAME[k]}", fontsize=8.5)
    axes[0].set_ylabel("permutation importance")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig09_importance")


def fig_benchmark(fc):
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    w = 0.16
    for i, m in enumerate(MAIN):
        v = [ms(fc["indomain"][k][m]["nrmse"]) for k in C.REGIONS]
        ax.bar(np.arange(3) + (i - 2) * w, [a for a, _ in v], w, yerr=[b for _, b in v], color=MODEL_PAL[m], edgecolor="black", linewidth=0.4, hatch=HATCH[m], label=m.replace("Proposed: ARIMA + VQK", "AR + VQK"), capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(range(3), [f"{k}: {RNAME[k]}" for k in C.REGIONS])
    ax.set_ylabel("NRMSE (lower is better)")
    ax.legend(ncol=3, frameon=False, fontsize=6.5)
    save(fig, "fig10_benchmark")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", type=str, default="")
    a = ap.parse_args()
    if a.paper_dir:
        extra_dirs.append(Path(a.paper_dir))
    fc, kn, nz, sim = load("forecast"), load("kernel"), load("noise"), load("simulator")
    fig_architecture()
    fig_circuit()
    if sim:
        fig_regions(sim)
        fig_sim_checks(sim)
    if fc:
        fig_forecast(fc)
        fig_transfer(fc)
        fig_importance(fc)
        fig_benchmark(fc)
    if kn:
        fig_kernel(kn)
    if nz:
        fig_noise(nz)


if __name__ == "__main__":
    main()
