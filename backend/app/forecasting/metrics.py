"""Forecast metrics: NRMSE, CRPS (MAE surrogate for point forecasts), decision regret, latency."""
from __future__ import annotations

import time

import numpy as np

from .. import config as C


def nrmse(y: np.ndarray, p: np.ndarray) -> float:
    """RMSE normalised by the range of the observed series in the evaluation window."""
    rng = float(np.max(y) - np.min(y))
    return float(np.sqrt(np.mean((y - p) ** 2)) / max(rng, 1e-9))


def crps_surrogate(y: np.ndarray, p: np.ndarray) -> float:
    """For a deterministic forecast the continuous ranked probability score equals the MAE."""
    return float(np.mean(np.abs(y - p)))


def curtailment_regret(y: np.ndarray, p: np.ndarray, a: float = C.COST_SHORTFALL, b: float = C.COST_CURTAIL) -> float:
    """Mean regret of committing ``p`` when ``y`` is realised.

    Over-forecast leaves a shortfall bought back at ``a`` per unit; under-forecast curtails surplus at ``b`` per unit.
    The perfect-foresight cost is zero, so the regret equals the realised cost (per-unit capacity-factor scale).
    """
    return float(np.mean(a * np.maximum(p - y, 0.0) + b * np.maximum(y - p, 0.0)))


def all_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"nrmse": nrmse(y, p), "crps": crps_surrogate(y, p), "regret": curtailment_regret(y, p)}


def latency_ms(step_fn, n: int = 200) -> float:
    """Average wall-clock time of one single-step forecast call, in milliseconds."""
    step_fn(0)
    t0 = time.perf_counter()
    for i in range(n):
        step_fn(i)
    return (time.perf_counter() - t0) / n * 1e3
