"""Supervised one-step-ahead dataset: features at time t, target y(t+1) (only past true values are used)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C


@dataclass
class Supervised:
    index: pd.DatetimeIndex     # timestamp of the *target* y(t+1)
    lags: np.ndarray            # (n, p)  y(t), y(t-1), ..., y(t-p+1)
    cov: np.ndarray             # (n, 6)  covariates at time t (raw units)
    y: np.ndarray               # (n,)    y(t+1)
    stormy: np.ndarray          # (n,)    latent regime label at time t

    def take(self, mask: np.ndarray) -> "Supervised":
        return Supervised(self.index[mask], self.lags[mask], self.cov[mask], self.y[mask], self.stormy[mask])

    def __len__(self) -> int:
        return len(self.y)


def make_supervised(df: pd.DataFrame, p: int = C.AR_ORDER) -> Supervised:
    y = df["y"].to_numpy()
    n = len(y)
    t = np.arange(p - 1, n - 1)
    lags = np.stack([y[t - k] for k in range(p)], axis=1)
    cov = df[list(C.COVARIATES)].to_numpy()[t]
    return Supervised(df.index[t + 1], lags, cov, y[t + 1], df["stormy"].to_numpy()[t])


@dataclass
class Splits:
    train: Supervised
    val: Supervised
    test: Supervised            # in-domain 14-day window
    test_week: Supervised       # first 7 days of the test window (cross-region transfer)
    finetune: Supervised        # 28 days of data preceding the test window


def split(sup: Supervised) -> Splits:
    idx = sup.index
    tr = idx <= pd.Timestamp(C.TRAIN_END)
    va = (idx > pd.Timestamp(C.TRAIN_END)) & (idx <= pd.Timestamp(C.VAL_END))
    t0 = pd.Timestamp(C.TEST_START)
    te = (idx >= t0) & (idx < t0 + pd.Timedelta(hours=C.TEST_HOURS_IN_DOMAIN))
    tw = (idx >= t0) & (idx < t0 + pd.Timedelta(hours=C.TEST_HOURS_TRANSFER))
    ft = (idx >= t0 - pd.Timedelta(hours=C.FINETUNE_HOURS)) & (idx < t0)
    return Splits(sup.take(tr), sup.take(va), sup.take(te), sup.take(tw), sup.take(ft))


class Scaler:
    def __init__(self, x: np.ndarray):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd


def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    h, d = index.hour.to_numpy(), index.dayofyear.to_numpy()
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)], axis=1)
