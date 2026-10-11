"""Turn results/*.json into LaTeX tables and macros for the manuscript.

``python -m scripts.export_paper_numbers --out ../../qvk-forecast-paper/generated``
Nothing in the paper is typed by hand: every number is read from the JSON written by ``run_experiments``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app import config as C

RES = C.RESULTS_DIR
MAIN = ["Persistence", "ARIMA (AR(24))", "XGBoost", "LSTM (2 layers)", "Proposed: ARIMA + VQK"]
ABL = ["ARIMA + VQK (circuit kernel)", "ARIMA + RBF kernel"]
TEX_NAME = {"Persistence": "Persistence", "ARIMA (AR(24))": "AR(24) (ARIMA family)", "XGBoost": "XGBoost",
            "LSTM (2 layers)": "LSTM (2 layers)", "Proposed: ARIMA + VQK": "\\textbf{Proposed: AR + VQK (surrogate)}",
            "ARIMA + VQK (circuit kernel)": "AR + VQK (exact circuit)", "ARIMA + RBF kernel": "AR + RBF kernel (control)"}
RN = C.REGION_NAMES


def ms(v, d=3):
    a = np.asarray(v, float)
    return f"{a.mean():.{d}f}\\,$\\pm$\\,{a.std(ddof=1) if len(a) > 1 else 0:.{d}f}"


def mean(v):
    return float(np.mean(v))


def table(path: Path, caption: str, label: str, header: str, rows: list[str], spec: str) -> None:
    body = "\n".join(rows)
    path.write_text(
        "\\begin{table}[!t]\n\\centering\n\\caption{" + caption + "}\\label{" + label + "}\n\\small\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header}\\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../qvk-forecast-paper/generated")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    f = json.loads((RES / "forecast.json").read_text())
    k = json.loads((RES / "kernel.json").read_text())
    s = json.loads((RES / "simulator.json").read_text())
    nz = json.loads((RES / "noise.json").read_text())
    nseed = int(f["seeds"])
    mac: list[str] = []

    def m(name, val):
        mac.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    m("NSeeds", nseed)
    for key, val in s["software"].items():
        m("Ver" + key.replace("_", "").title().replace(" ", ""), val)
    m("NParams", s["n_circuit_parameters"])
    m("GenSeconds", f"{s['generation_seconds_three_regions_eight_years']:.1f}")

    # ---------- in-domain table --------------------------------------------------------------------------
    for r in C.REGIONS:
        rows = []
        for name in MAIN + ABL:
            d = f["indomain"][r][name]
            lat = d["latency_ms"]
            lat_s = f"{mean(lat):.2f}" if lat else "--"
            rows.append(f"{TEX_NAME[name]} & {ms(d['nrmse'], 4)} & {ms(d['crps'], 4)} & {ms(d['regret'], 4)} & {lat_s} \\\\")
            tag = {"Persistence": "Pers", "ARIMA (AR(24))": "AR", "XGBoost": "Xgb", "LSTM (2 layers)": "Lstm",
                   "Proposed: ARIMA + VQK": "Vqk", "ARIMA + VQK (circuit kernel)": "Circ", "ARIMA + RBF kernel": "Rbf"}[name]
            m(f"Nrmse{tag}{r}", f"{mean(d['nrmse']):.4f}")
            m(f"Crps{tag}{r}", f"{mean(d['crps']):.4f}")
            m(f"Regret{tag}{r}", f"{mean(d['regret']):.4f}")
            if lat:
                m(f"Lat{tag}{r}", f"{mean(lat):.2f}")
        table(out / f"tab_indomain_{r}.tex",
              f"In-domain one-hour-ahead results for region {r} ({RN[r]}) on the simulated 14-day test window: mean\\,$\\pm$\\,s.d. over {nseed} seeds. "
              "NRMSE is normalised by the observed range, CRPS equals MAE for point forecasts, regret uses $a=2$, $b=1$. Latency is per forecast.",
              f"tab:indomain_{r}", "Model & NRMSE & CRPS (MAE) & Regret & Latency (ms)", rows, "lrrrr")
    # all-regions compact table
    rows = []
    for name in MAIN:
        cells = " & ".join(ms(f["indomain"][r][name]["nrmse"], 3) for r in C.REGIONS)
        rows.append(f"{TEX_NAME[name]} & {cells} \\\\")
    table(out / "tab_indomain_all.tex",
          f"In-domain NRMSE for the three simulated regions (mean\\,$\\pm$\\,s.d., {nseed} seeds).", "tab:indomain_all",
          "Model & A (solar) & B (wind) & C (mixed)", rows, "lccc")
    # best model per region and gap between proposed and AR
    for r in C.REGIONS:
        best = min(MAIN, key=lambda n: mean(f["indomain"][r][n]["nrmse"]))
        m(f"Best{r}", TEX_NAME[best].replace("\\textbf{", "").replace("}", ""))
        ar, vq = mean(f["indomain"][r]["ARIMA (AR(24))"]["nrmse"]), mean(f["indomain"][r]["Proposed: ARIMA + VQK"]["nrmse"])
        m(f"GapVqkAr{r}", f"{100 * (ar - vq) / ar:.1f}")
        xg = mean(f["indomain"][r]["XGBoost"]["nrmse"])
        m(f"GapVqkXgb{r}", f"{100 * (xg - vq) / xg:.1f}")
        pe = mean(f["indomain"][r]["Persistence"]["nrmse"])
        m(f"GapVqkPers{r}", f"{100 * (pe - vq) / pe:.1f}")

    # ---------- transfer table ----------------------------------------------------------------------------
    models = ["ARIMA (AR(24))", "XGBoost", "LSTM (2 layers)", "Proposed: ARIMA + VQK"]
    rows = []
    for pair, d in f["transfer"].items():
        src, tgt = pair.split("->")
        kind = "in-domain" if src == tgt else "cross-region"
        cells = []
        for name in models:
            z = mean(d[name]["zero_shot"])
            ft = d[name]["fine_tune"]
            cells.append(f"{z:.3f}" + (f" / {mean(ft):.3f}" if ft else " / --"))
        rows.append(f"{src}\\,$\\to$\\,{tgt} ({kind}) & " + " & ".join(cells) + " \\\\")
    table(out / "tab_transfer.tex",
          "Cross-region transfer NRMSE, zero-shot / after fine-tuning on 28 days of target data (mean over seeds). "
          "Diagonal pairs are in-domain and are not fine-tuned.", "tab:transfer",
          "Source\\,$\\to$\\,target & AR(24) & XGBoost & LSTM & AR + VQK", rows, "lcccc")
    off = [p for p in f["transfer"] if p[0] != p[-1]]
    for name, tag in (("ARIMA (AR(24))", "AR"), ("XGBoost", "Xgb"), ("LSTM (2 layers)", "Lstm"), ("Proposed: ARIMA + VQK", "Vqk")):
        zs = np.mean([mean(f["transfer"][p][name]["zero_shot"]) for p in off])
        fs = np.mean([mean(f["transfer"][p][name]["fine_tune"]) for p in off])
        m(f"TransZero{tag}", f"{zs:.3f}")
        m(f"TransFine{tag}", f"{fs:.3f}")
    best_t = min(models, key=lambda n: np.mean([mean(f["transfer"][p][n]["fine_tune"]) for p in off]))
    m("BestTransfer", TEX_NAME[best_t].replace("\\textbf{", "").replace("}", ""))


    # ---------- transfer summary statistics ---------------------------------------------------------------
    diag = [p for p in f["transfer"] if p[0] == p[-1]]
    for name, tag in (("ARIMA (AR(24))", "AR"), ("XGBoost", "Xgb"), ("LSTM (2 layers)", "Lstm"), ("Proposed: ARIMA + VQK", "Vqk")):
        zi = np.mean([mean(f["transfer"][p][name]["zero_shot"]) for p in diag])
        zo = np.mean([mean(f["transfer"][p][name]["zero_shot"]) for p in off])
        fo = np.mean([mean(f["transfer"][p][name]["fine_tune"]) for p in off])
        m(f"TransIn{tag}", f"{zi:.3f}")
        m(f"TransRise{tag}", f"{100 * (zo - zi) / zi:.0f}")
        m(f"TransGain{tag}", f"{100 * (zo - fo) / zo:.0f}")

    # ---------- permutation importance ---------------------------------------------------------------------
    lab = {"wind_ms": "wind speed", "gust_ms": "gust", "pressure_hpa": "pressure", "temp_c": "temperature",
           "ghi_wm2": "irradiance (GHI)", "dp3h_hpa": "3-h pressure change"}
    rows = []
    for r in C.REGIONS:
        imp = f["importance"][r]
        for key, nm in (("xgboost", "XGBoost"), ("proposed_forecast", "AR + VQK")):
            v = np.array(imp[key], float)
            sh = v / v.sum()
            top = np.argsort(-sh)[:3]
            cells = " & ".join(f"{lab[imp['covariates'][j]]} ({sh[j]:.2f})" for j in top)
            rows.append(f"{r} & {nm} & {cells} \\\\")
            m(f"ImpTop{r}{'Xgb' if key == 'xgboost' else 'Vqk'}", lab[imp["covariates"][int(top[0])]])
    table(out / "tab_importance.tex",
          "Top three covariates by permutation importance (Eq.~\\eqref{eq:perm}) per region, with each covariate's share of the total importance "
          "(first seed, test window). AR(24) uses no covariates and is not shown.", "tab:importance",
          "Region & Model & Rank 1 & Rank 2 & Rank 3", rows, "llccc")

    # ---------- noise ----------------------------------------------------------------------------------------
    m("NoiseSeeds", len(nz["exact"]))
    m("FdrExactMin", f"{min(nz['exact']):.2f}")
    m("FdrExactMax", f"{max(nz['exact']):.2f}")

    # ---------- simulator characteristics ----------------------------------------------------------------
    rows = []
    for r in C.REGIONS:
        d = s["regions"][r]
        rows.append(f"{r}: {d['name']} & {d['mean_cf']:.3f} & {d['std_cf']:.3f} & {d['acf1']:.3f} & {d['acf24']:.3f} & "
                    f"{100 * d['storm_fraction']:.1f} & {d['mean_storm_hours']:.0f} \\\\")
        m(f"SimStormFrac{r}", f"{100 * d['storm_fraction']:.1f}")
        m(f"SimMeanCF{r}", f"{d['mean_cf']:.3f}")
        m(f"SimHours{r}", f"{d['hours']:,}".replace(",", "\\,"))
    table(out / "tab_sim.tex", "Characteristics of the simulated regions (hourly, 2018--2025). ACF is the autocorrelation of the capacity factor at the given lag; "
          "storm columns give the share of hours in the stormy regime and the mean spell length.", "tab:sim",
          "Region & mean CF & s.d. & ACF(1) & ACF(24) & stormy (\\%) & spell (h)", rows, "lrrrrrr")
    sv = s["surrogate_vs_circuit"]
    m("SurrCorrLo", f"{min(sv.values()):.2f}")
    m("SurrCorrHi", f"{max(sv.values()):.2f}")

    # ---------- kernel separability ----------------------------------------------------------------------
    names = [("quantum_circuit", "Quantum circuit kernel (exact)"), ("quantum_surrogate", "Closed-form surrogate (Eq.~\\eqref{eq:surrogate})"),
             ("rbf_tuned", "RBF kernel (tuned $\\gamma$)")]
    rows = []
    wins = 0
    tot = 0
    for r in C.REGIONS:
        for key, lab in names:
            fd = [x[key]["fdr_pc1"] for x in k["regions"][r]]
            sv_ = [x[key]["svm_acc"] for x in k["regions"][r] if x[key]["svm_acc"] is not None]
            rows.append(f"{r} & {lab} & {ms(fd, 2)} & {ms(sv_, 3) if sv_ else '--'} \\\\")
        for x in k["regions"][r]:
            tot += 1
            wins += x["quantum_circuit"]["fdr_pc1"] > x["rbf_tuned"]["fdr_pc1"]
        m(f"FdrCirc{r}", f"{mean([x['quantum_circuit']['fdr_pc1'] for x in k['regions'][r]]):.2f}")
        m(f"FdrRbf{r}", f"{mean([x['rbf_tuned']['fdr_pc1'] for x in k['regions'][r]]):.2f}")
        m(f"FdrSurr{r}", f"{mean([x['quantum_surrogate']['fdr_pc1'] for x in k['regions'][r]]):.2f}")
        m(f"SvmCirc{r}", f"{mean([x['quantum_circuit']['svm_acc'] for x in k['regions'][r]]):.3f}")
        m(f"SvmRbf{r}", f"{mean([x['rbf_tuned']['svm_acc'] for x in k['regions'][r]]):.3f}")
    table(out / "tab_kernel.tex",
          "Calm-versus-stormy separability: Fisher discriminant ratio (FDR) on the leading kernel-PCA component and 4-fold SVM accuracy "
          "(mean\\,$\\pm$\\,s.d. over seeds). Kernel hyper-parameters are tuned by SVM accuracy.", "tab:kernel",
          "Region & Kernel & FDR (PC1) & SVM accuracy", rows, "llrr")
    m("KernelWins", wins)
    m("KernelTrials", tot)

    # ---------- noise -------------------------------------------------------------------------------------
    rows = []
    for sh, v in nz["shots"].items():
        cnt = f"{int(sh):,}".replace(",", "\\,")
        rows.append(f"Shot noise, {cnt} shots & {ms(v, 3) if len(v) > 1 else f'{v[0]:.3f}'} \\\\")
    for p, v in nz["depolarising"].items():
        rows.append(f"Depolarising $p={p}$ & {ms(v, 3) if len(v) > 1 else f'{v[0]:.3f}'} \\\\")
    ex = nz["exact"]
    m("FdrExactNoise", f"{mean(ex):.3f}")
    m("FdrDepolHigh", f"{mean(nz['depolarising']['0.03']):.3f}")
    m("FdrShotLow", f"{mean(nz['shots']['50']):.3f}")
    table(out / "tab_noise.tex", f"Effect of hardware-like noise on the Fisher ratio of the circuit kernel (region C, $\\alpha=0.5$, {len(ex)} seeds, 150 samples per class; shot-noise entries average 5 repetitions). The exact kernel gives {mean(ex):.3f}\\,$\\pm$\\,{np.std(ex, ddof=1):.3f}.", "tab:noise",
          "Noise setting & FDR (PC1)", rows, "lr")

    from app.explain.narrator import explain_from_files
    (out / "explainer.tex").write_text(explain_from_files().replace("%", "\\%").replace("+", "{+}") + "\n")
    (out / "numbers.tex").write_text("% generated by scripts/export_paper_numbers.py -- do not edit\n" + "\n".join(mac) + "\n")
    print(f"wrote {len(mac)} macros and tables to {out}")


if __name__ == "__main__":
    main()
