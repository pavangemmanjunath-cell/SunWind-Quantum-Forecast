"""Lazy, cached access to simulated data and the fitted proposed model (used by the API)."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from .. import config as C
from ..data.synthetic import generate_region
from ..forecasting.baselines import ARPlusVQK
from ..forecasting.dataset import Scaler, make_supervised, split
from ..quantum import kernel as Q
from ..quantum.circuit import init_params


@lru_cache(maxsize=16)
def region_frame(region: str, seed: int = C.DEFAULT_SEED):
    return generate_region(region, seed)


@lru_cache(maxsize=8)
def region_splits(region: str, seed: int = C.DEFAULT_SEED):
    return split(make_supervised(region_frame(region, seed)))


@lru_cache(maxsize=4)
def proposed_model(region: str) -> ARPlusVQK:
    sp = region_splits(region)
    return ARPlusVQK("surrogate", seed=0).fit(sp.train, sp.val)


def results(name: str) -> dict | None:
    p = C.RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def embedding(region: str, alpha: float, gamma: float, n_per_class: int, seed: int) -> dict:
    """Live calm-vs-stormy kernel-PCA embedding for the interactive Kernel Lab."""
    from ..forecasting.protocol import _balanced_sample
    sp = region_splits(region)
    rng = np.random.default_rng(seed)
    X, y = _balanced_sample(sp.val, n_per_class, rng)
    X = Scaler(sp.train.cov)(X)
    params = init_params(seed=seed)
    out = {"labels": y.tolist()}
    for name, K in (("quantum", Q.circuit_kernel(X, X, params, alpha)), ("rbf", Q.rbf_kernel(X, X, gamma))):
        z = Q.kernel_pca_scores(K, 2)
        out[name] = {"points": z.tolist(), "fdr": Q.fisher_ratio(z[:, 0], y)}
    return out
