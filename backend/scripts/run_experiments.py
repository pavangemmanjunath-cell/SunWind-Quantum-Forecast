"""Reproduce every number in the manuscript.

Usage (from ``backend/``)::

    python -m scripts.run_experiments forecast --seeds 5     # in-domain benchmark + cross-region transfer + traces
    python -m scripts.run_experiments kernel   --seeds 5     # calm-vs-stormy kernel separability
    python -m scripts.run_experiments noise                  # finite-shot and depolarising noise on the kernel
    python -m scripts.run_experiments simulator              # characterisation of the simulated regions
    python -m scripts.run_experiments all

Each stage writes JSON into ``../results``; ``make_figures`` and ``export_paper_numbers`` read only those files.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from app import config as C
from app.data.synthetic import SPECS, generate_all
from app.forecasting.protocol import (
    ALPHAS,
    evaluate_in_domain,
    permutation_importance,
    prepare,
    regime_study,
    train_region,
    transfer_matrix,
)
from app.quantum import kernel as Q
from app.quantum.circuit import init_params, n_parameters

RES = C.RESULTS_DIR
RES.mkdir(parents=True, exist_ok=True)
TRANSFER_MODELS = ["ARIMA (AR(24))", "XGBoost", "LSTM (2 layers)", "Proposed: ARIMA + VQK"]


def _dump(name: str, obj) -> None:
    (RES / name).write_text(json.dumps(obj, indent=1))
    print(f"[saved] {RES / name}", flush=True)


# ---------------------------------------------------------------------------------------------------------
def stage_forecast(seeds: int) -> None:
    indomain = {k: {} for k in C.REGIONS}
    transfer: dict[str, dict[str, dict[str, list]]] = {}
    extra: dict = {}
    for seed in range(seeds):
        t0 = time.time()
        data = prepare(seed)
        trained = {}
        for k in C.REGIONS:
            models = train_region(data[k][1], seed)
            trained[k] = models
            res = evaluate_in_domain(models, data[k][1], measure_latency=(seed == 0))
            for name, r in res.items():
                for metric, v in r.items():
                    indomain[k].setdefault(name, {}).setdefault(metric, []).append(v)
            if seed == 0:
                sp = data[k][1]
                extra.setdefault("week_traces", {})[k] = {
                    "t": [str(t) for t in sp.test.index], "y": sp.test.y.tolist(),
                    "stormy": sp.test.stormy.tolist(), "ghi": sp.test.cov[:, 4].tolist(), "wind": sp.test.cov[:, 0].tolist(),
                    "pred": {m.name: m.predict(sp.test).tolist() for m in models}}
                prop = next(m for m in models if m.name == "Proposed: ARIMA + VQK")
                xgb = next(m for m in models if m.name == "XGBoost")
                extra.setdefault("importance", {})[k] = {
                    "covariates": list(C.COVARIATES),
                    "proposed_forecast": permutation_importance(prop, sp.test, target="forecast"),
                    "proposed_correction": permutation_importance(prop, sp.test, target="correction"),
                    "xgboost": permutation_importance(xgb, sp.test, target="forecast"),
                    "lambda": prop.lam}
        tm = transfer_matrix(trained, data, TRANSFER_MODELS)
        for pair, d in tm.items():
            for name, r in d.items():
                slot = transfer.setdefault(pair, {}).setdefault(name, {"zero_shot": [], "fine_tune": []})
                slot["zero_shot"].append(r["zero_shot"])
                if r["fine_tune"] is not None:
                    slot["fine_tune"].append(r["fine_tune"])
        print(f"[forecast] seed {seed} done in {time.time() - t0:.0f}s", flush=True)
        _dump("forecast.json", {"seeds": seeds, "indomain": indomain, "transfer": transfer, **extra})


def stage_kernel(seeds: int) -> None:
    out = {"seeds": seeds, "regions": {k: [] for k in C.REGIONS}}
    for seed in range(seeds):
        data = prepare(seed)
        for k in C.REGIONS:
            out["regions"][k].append(regime_study(*data[k], seed=seed, keep_embedding=(seed == 0 and k == "C")))
        print(f"[kernel] seed {seed} done", flush=True)
        _dump("kernel.json", out)


def stage_noise() -> None:
    out = {"shots": {}, "depolarising": {}, "exact": []}
    for seed in range(3):
        data = prepare(seed)
        df, sp = data["C"]
        rng = np.random.default_rng(seed)
        from app.forecasting.dataset import Scaler
        from app.forecasting.protocol import _balanced_sample
        X, y = _balanced_sample(sp.val, 150, rng)
        X = Scaler(sp.train.cov)(X)
        params = init_params(seed=seed)
        alpha = 0.5
        K = Q.circuit_kernel(X, X, params, alpha)
        out["exact"].append(Q.fisher_ratio(Q.kernel_pca_scores(K, 1)[:, 0], y))
        for shots in (50, 200, 1000, 5000, 20000):
            f = [Q.fisher_ratio(Q.kernel_pca_scores(Q.shot_noise_kernel(K, shots, np.random.default_rng(100 * seed + r)), 1)[:, 0], y)
                 for r in range(5)]
            out["shots"].setdefault(str(shots), []).append(float(np.mean(f)))
        for p in (0.0, 0.001, 0.003, 0.01, 0.03):
            Kn = Q.depolarising_kernel(X, params, alpha, p)
            out["depolarising"].setdefault(str(p), []).append(Q.fisher_ratio(Q.kernel_pca_scores(Kn, 1)[:, 0], y))
        print(f"[noise] seed {seed} done", flush=True)
        _dump("noise.json", out)


def stage_simulator() -> None:
    from scipy.stats import pearsonr

    data = generate_all(seed=C.DEFAULT_SEED)
    out = {"software": {"python": platform.python_version(), "numpy": np.__version__}, "regions": {}, "specs": {}}
    import pennylane
    import sklearn
    import xgboost
    out["software"].update({"pennylane": pennylane.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__})
    try:
        import torch
        out["software"]["torch"] = torch.__version__.split("+")[0]
    except Exception:  # noqa: BLE001
        pass
    t = time.time()
    generate_all(seed=C.DEFAULT_SEED + 1)
    out["generation_seconds_three_regions_eight_years"] = time.time() - t
    for k, df in data.items():
        y = df["y"]
        storm = df["stormy"].to_numpy()
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, storm, 0])))[::2]
        ramp = np.abs(y.diff().dropna())
        diurnal = y.groupby(df.index.hour).mean()
        out["regions"][k] = {
            "name": SPECS[k].name, "hours": int(len(df)), "mean_cf": float(y.mean()), "std_cf": float(y.std()),
            "p95_ramp": float(ramp.quantile(0.95)), "acf1": float(y.autocorr(1)), "acf24": float(y.autocorr(24)),
            "acf72": float(y.autocorr(72)), "storm_fraction": float(storm.mean()),
            "mean_storm_hours": float(runs.mean()) if len(runs) else 0.0, "n_storms": int(len(runs)),
            "mean_wind_ms": float(df["wind_ms"].mean()), "mean_ghi": float(df["ghi_wm2"].mean()),
            "zero_fraction": float((y < 0.005).mean()),
            "diurnal": diurnal.tolist(),
            "acf": [float(y.autocorr(l)) for l in range(1, 97)],
            "hist": np.histogram(y, bins=20, range=(0, 1), density=True)[0].tolist(),
            "week": {"t": [str(i) for i in df.loc["2025-06-02":"2025-06-08"].index], "y": df.loc["2025-06-02":"2025-06-08", "y"].tolist()},
            "storm_vs_calm_wind": [float(df.loc[df.stormy == 0, "wind_ms"].mean()), float(df.loc[df.stormy == 1, "wind_ms"].mean())],
            "storm_vs_calm_dp": [float(df.loc[df.stormy == 0, "dp3h_hpa"].mean()), float(df.loc[df.stormy == 1, "dp3h_hpa"].mean())],
        }
    # circuit vs closed-form surrogate agreement
    from app.forecasting.dataset import Scaler, make_supervised, split
    sp = split(make_supervised(data["C"]))
    X = Scaler(sp.train.cov)(sp.train.cov[np.random.default_rng(0).choice(len(sp.train), 400, replace=False)])
    params = init_params(seed=0)
    iu = np.triu_indices(400, 1)
    out["surrogate_vs_circuit"] = {a: float(pearsonr(Q.circuit_kernel(X, X, params, a)[iu], Q.surrogate_kernel(X, X, a)[iu])[0])
                                   for a in ALPHAS}
    out["n_circuit_parameters"] = n_parameters(params)
    _dump("simulator.json", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["forecast", "kernel", "noise", "simulator", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    if a.stage in ("simulator", "all"):
        stage_simulator()
    if a.stage in ("kernel", "all"):
        stage_kernel(a.seeds)
    if a.stage in ("noise", "all"):
        stage_noise()
    if a.stage in ("forecast", "all"):
        stage_forecast(a.seeds)
