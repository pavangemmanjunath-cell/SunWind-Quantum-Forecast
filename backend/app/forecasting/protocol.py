"""Experiment protocol shared by the scripts and the API: training, in-domain evaluation, transfer, regime study."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from .. import config as C
from ..data.synthetic import generate_region
from ..quantum import kernel as Q
from ..quantum.circuit import init_params
from .baselines import Forecaster, make_models
from .dataset import Scaler, Splits, make_supervised, split
from .metrics import all_metrics, latency_ms


def prepare(seed: int) -> dict[str, tuple[pd.DataFrame, Splits]]:
    out = {}
    for k in C.REGIONS:
        df = generate_region(k, seed)
        out[k] = (df, split(make_supervised(df)))
    return out


def train_region(sp: Splits, seed: int, with_extras: bool = True) -> list[Forecaster]:
    models = make_models(seed, with_extras)
    for m in models:
        m.fit(sp.train, sp.val)
    return models


def evaluate_in_domain(models: list[Forecaster], sp: Splits, measure_latency: bool = False) -> dict[str, dict]:
    res = {}
    for m in models:
        p = m.predict(sp.test)
        r = all_metrics(sp.test.y, p)
        if measure_latency:
            rows = [sp.test.take(np.arange(len(sp.test)) == i) for i in range(min(120, len(sp.test)))]
            r["latency_ms"] = latency_ms(lambda i: m.predict(rows[i % len(rows)]), n=120)
        res[m.name] = r
    return res


def transfer_matrix(trained: dict[str, list[Forecaster]], data: dict[str, tuple[pd.DataFrame, Splits]],
                    names: list[str]) -> dict[str, dict]:
    """NRMSE on the target region's test week for every ordered (source, target) pair.

    in-domain: source == target; zero-shot: source model applied unchanged; fine-tune: model adapted on the 28 days
    of target-region data that precede the test window.
    """
    out: dict[str, dict] = {}
    for src in C.REGIONS:
        for tgt in C.REGIONS:
            tsp = data[tgt][1]
            for m in trained[src]:
                if m.name not in names:
                    continue
                zs = all_metrics(tsp.test_week.y, m.predict(tsp.test_week))["nrmse"]
                ft = None
                if src != tgt:
                    mm = m.clone()
                    mm.finetune(tsp.finetune)
                    ft = all_metrics(tsp.test_week.y, mm.predict(tsp.test_week))["nrmse"]
                out.setdefault(f"{src}->{tgt}", {})[m.name] = {"zero_shot": zs, "fine_tune": ft}
    return out


def permutation_importance(model: Forecaster, s, n_rep: int = 20, seed: int = 0, target: str = "forecast") -> list[float]:
    """Eq. 8: mean L1 change of the forecast (or of the correction term) when covariate j is permuted, over sigma_y."""
    rng = np.random.default_rng(seed)
    base = model.correction(s) if target == "correction" else model.predict(s)
    sig = float(np.std(s.y))
    imp = []
    for j in range(C.N_COV):
        acc = 0.0
        for _ in range(n_rep):
            s2 = copy.copy(s)
            cov = s.cov.copy()
            cov[:, j] = rng.permutation(cov[:, j])
            s2.cov = cov
            p = model.correction(s2) if target == "correction" else model.predict(s2)
            acc += float(np.mean(np.abs(p - base))) / sig
        imp.append(acc / n_rep)
    return imp


# ---------------------------------------------------------------------------------------------------------
# weather-regime separability (calm vs stormy)
# ---------------------------------------------------------------------------------------------------------
def _balanced_sample(sup, n_per_class: int, rng) -> tuple[np.ndarray, np.ndarray]:
    lab = sup.stormy
    idx = np.concatenate([rng.choice(np.where(lab == c)[0], n_per_class, replace=False) for c in (0, 1)])
    return sup.cov[idx], lab[idx]


def _cv_accuracy(K: np.ndarray, y: np.ndarray, C_svm: float = 1.0, folds: int = 4) -> float:
    accs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(K, y):
        clf = SVC(kernel="precomputed", C=C_svm).fit(K[np.ix_(tr, tr)], y[tr])
        accs.append(clf.score(K[np.ix_(te, tr)], y[te]))
    return float(np.mean(accs))


ALPHAS = (0.25, 0.5, 1.0, 2.0)
GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0)


def regime_study(df: pd.DataFrame, sp: Splits, seed: int, n_tune: int = 200, n_eval: int = 400,
                 keep_embedding: bool = False) -> dict:
    """Calm-vs-stormy separability: leading-component Fisher ratio and SVM accuracy for the quantum and RBF kernels.

    Each kernel's single hyper-parameter (angle scale alpha, bandwidth gamma) is tuned by cross-validated SVM accuracy
    on a tuning sample from the training years; the Fisher ratio is then measured on a disjoint sample from 2024-2025.
    """
    rng = np.random.default_rng(seed)
    scaler = Scaler(sp.train.cov)
    params = init_params(seed=seed)
    Xt, yt = _balanced_sample(sp.train, n_tune, rng)
    ev = sp.val   # 2024 sample for the evaluation set
    Xe, ye = _balanced_sample(ev, n_eval, rng)
    Xt, Xe = scaler(Xt), scaler(Xe)

    def tune(fn, grid):
        scores = {g: _cv_accuracy(fn(Xt, Xt, g), yt) for g in grid}
        best = max(scores, key=scores.get)
        return best, scores

    a_best, a_scores = tune(lambda A, B, a: Q.circuit_kernel(A, B, params, a), ALPHAS)
    g_best, g_scores = tune(lambda A, B, g: Q.rbf_kernel(A, B, g), GAMMAS)
    out = {"alpha": a_best, "gamma": g_best, "alpha_cv": a_scores, "gamma_cv": g_scores}
    kernels = {
        "quantum_circuit": Q.circuit_kernel(Xe, Xe, params, a_best),
        "quantum_surrogate": Q.surrogate_kernel(Xe, Xe, a_best),
        "rbf_tuned": Q.rbf_kernel(Xe, Xe, g_best),
        "quantum_circuit_alpha1": Q.circuit_kernel(Xe, Xe, params, 1.0),
    }
    for name, K in kernels.items():
        z = Q.kernel_pca_scores(K, 2)
        fdr1 = Q.fisher_ratio(z[:, 0], ye)
        fdr2 = Q.fisher_ratio(z[:, 1], ye)
        out[name] = {"fdr_pc1": fdr1, "fdr_pc2": fdr2, "fdr_best_of_2": max(fdr1, fdr2),
                     "svm_acc": _cv_accuracy(K + 0.0, ye) if name != "quantum_surrogate" else None}
        if keep_embedding and name in ("quantum_circuit", "rbf_tuned"):
            out[name]["embedding"] = z.tolist()
            out[name]["labels"] = ye.tolist()
    return out
