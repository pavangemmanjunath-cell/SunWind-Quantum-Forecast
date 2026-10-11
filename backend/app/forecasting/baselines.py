"""Forecasters: persistence, AR(24) ('ARIMA' in the paper's tables), XGBoost, 2-layer LSTM, and the proposed AR + VQK."""
from __future__ import annotations

import copy

import numpy as np

from .. import config as C
from ..quantum import kernel as Q
from ..quantum.circuit import init_params
from .dataset import Scaler, Supervised, time_features


class Forecaster:
    name = "base"

    def fit(self, tr: Supervised, va: Supervised):
        return self

    def predict(self, s: Supervised) -> np.ndarray:
        raise NotImplementedError

    def finetune(self, ft: Supervised):
        return self

    def clone(self):
        return copy.deepcopy(self)


class Persistence(Forecaster):
    name = "Persistence"

    def predict(self, s):
        return s.lags[:, 0].copy()


class AR(Forecaster):
    """Autoregressive baseline of order p = 24 solved by least squares (Eq. 2) with an intercept.

    The stored result identifier retains the historical ``ARIMA (AR(24))`` key; this implementation is AR(24)
    without differencing or a moving-average component.
    """
    name = "ARIMA (AR(24))"

    def __init__(self, p: int = C.AR_ORDER):
        self.p = p
        self.coef = np.zeros(p + 1)

    @staticmethod
    def _design(s):
        return np.hstack([np.ones((len(s), 1)), s.lags])

    def fit(self, tr, va=None):
        self.coef, *_ = np.linalg.lstsq(self._design(tr), tr.y, rcond=None)
        return self

    def predict(self, s):
        return np.clip(self._design(s) @ self.coef, 0.0, 1.0)

    def finetune(self, ft, lam: float = 50.0):
        """Ridge towards the source coefficients: argmin ||y - Xc||^2 + lam ||c - c_src||^2."""
        X = self._design(ft)
        A = X.T @ X + lam * np.eye(X.shape[1])
        self.coef = np.linalg.solve(A, X.T @ ft.y + lam * self.coef)
        return self


class XGB(Forecaster):
    name = "XGBoost"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.booster = None

    @staticmethod
    def _feat(s):
        return np.hstack([s.lags, s.cov, time_features(s.index)])

    def fit(self, tr, va):
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=30, n_jobs=2, random_state=self.seed, verbosity=0)
        m.fit(self._feat(tr), tr.y, eval_set=[(self._feat(va), va.y)], verbose=False)
        self.booster = m.get_booster()[: m.best_iteration + 1]      # keep exactly the early-stopped ensemble
        return self

    def predict(self, s):
        import xgboost as xgb
        return np.clip(self.booster.predict(xgb.DMatrix(self._feat(s))), 0.0, 1.0)

    def finetune(self, ft):
        """Continue boosting for 100 rounds (learning rate 0.1) on the target-region fine-tuning window."""
        import xgboost as xgb
        params = {"max_depth": 4, "eta": 0.1, "subsample": 0.8, "nthread": 2, "seed": self.seed, "verbosity": 0}
        self.booster = xgb.train(params, xgb.DMatrix(self._feat(ft), label=ft.y), num_boost_round=100, xgb_model=self.booster)
        return self


class LSTM(Forecaster):
    """Two-layer LSTM over the last 24 hourly values, with the covariates at time t injected before the output head.

    The network predicts the increment y(t+1) - y(t) (a persistence-residual parameterisation that trains stably).
    """
    name = "LSTM (2 layers)"

    def __init__(self, seed: int = 0, hidden: int = 32, epochs: int = 8, max_windows: int = 30000):
        self.seed, self.hidden, self.epochs, self.max_windows = seed, hidden, epochs, max_windows
        self.net = None
        self.scaler = None

    def _build(self):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(s, n_cov, hid):
                super().__init__()
                s.lstm = nn.LSTM(1, hid, num_layers=2, batch_first=True)
                s.cov = nn.Linear(n_cov, 8)
                s.head = nn.Sequential(nn.Linear(hid + 8, 16), nn.ReLU(), nn.Linear(16, 1))

            def forward(s, seq, cov):
                o, _ = s.lstm(seq)
                return s.head(torch.cat([o[:, -1], torch.relu(s.cov(cov))], dim=1)).squeeze(-1)

        torch.manual_seed(self.seed)
        return Net(C.N_COV, self.hidden)

    def _tensors(self, s):
        import torch
        seq = torch.tensor(s.lags[:, ::-1].copy(), dtype=torch.float32).unsqueeze(-1)   # oldest -> newest
        cov = torch.tensor(self.scaler(s.cov), dtype=torch.float32)
        return seq, cov

    def _train(self, s, epochs, lr, val=None, patience=3):
        import torch
        rng = np.random.default_rng(self.seed)
        seq, cov = self._tensors(s)
        tgt = torch.tensor(s.y - s.lags[:, 0], dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        best, best_state, bad = 1e9, None, 0
        for ep in range(epochs):
            self.net.train()
            idx = rng.permutation(len(tgt))[:self.max_windows]
            for b in range(0, len(idx), 512):
                j = torch.tensor(idx[b:b + 512])
                opt.zero_grad()
                loss = torch.mean((self.net(seq[j], cov[j]) - tgt[j]) ** 2)
                loss.backward()
                opt.step()
            if val is not None:
                err = float(np.sqrt(np.mean((self.predict(val) - val.y) ** 2)))
                if err < best:
                    best, best_state, bad = err, copy.deepcopy(self.net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def fit(self, tr, va):
        import torch
        torch.set_num_threads(2)
        self.scaler = Scaler(tr.cov)
        self.net = self._build()
        self._train(tr, self.epochs, 2e-3, val=va)
        return self

    def predict(self, s):
        import torch
        self.net.eval()
        with torch.no_grad():
            seq, cov = self._tensors(s)
            out = self.net(seq, cov).numpy()
        return np.clip(s.lags[:, 0] + out, 0.0, 1.0)

    def finetune(self, ft):
        self._train(ft, 5, 1e-3)
        return self


class ARPlusVQK(Forecaster):
    """Proposed configuration: AR(24) forecast + kernel-ridge correction of its residual on a quantum-inspired kernel.

    ``kernel='surrogate'`` uses the closed-form Eq. 5 (as in the manuscript); ``'circuit'`` evaluates the exact
    six-qubit state-vector kernel (Eq. 4) through PennyLane; ``'rbf'`` is a classical control.
    """

    def __init__(self, kernel: str = "surrogate", alpha: float = 1.0, lam: float | None = None, n_sub: int = 2000,
                 seed: int = 0):
        self.kind, self.alpha, self.lam_fixed, self.n_sub, self.seed = kernel, alpha, lam, n_sub, seed
        self.ar = AR()
        self.params = init_params(seed=seed)
        self.krr = None
        self.lam = lam
        self.name = {"surrogate": "Proposed: ARIMA + VQK", "circuit": "ARIMA + VQK (circuit kernel)",
                     "rbf": "ARIMA + RBF kernel"}[kernel]

    def _kernel(self):
        if self.kind == "surrogate":
            return lambda A, B: Q.surrogate_kernel(A, B, self.alpha)
        if self.kind == "circuit":
            return Q.CachedCircuitKernel(self.params, self.alpha)
        return lambda A, B: Q.rbf_kernel(A, B, 0.1)

    def _fit_krr(self, s, lam, rng):
        eps = s.y - self.ar.predict(s)
        sel = rng.choice(len(s), size=min(self.n_sub, len(s)), replace=False)
        self.krr = Q.KernelRidge(self._kernel(), lam).fit(self.scaler(s.cov[sel]), eps[sel])

    def fit(self, tr, va):
        rng = np.random.default_rng(self.seed)
        self.ar.fit(tr)
        self.scaler = Scaler(tr.cov)
        grid = [self.lam_fixed] if self.lam_fixed else [1.0, 3.0, 10.0, 30.0, 100.0]
        vsel = rng.choice(len(va), size=min(3000, len(va)), replace=False)
        vs = va.take(np.isin(np.arange(len(va)), vsel))
        best = (1e9, grid[0])
        for lam in grid:                              # lambda chosen on the validation year
            self._fit_krr(tr, lam, np.random.default_rng(self.seed))
            err = float(np.sqrt(np.mean((self.predict(vs) - vs.y) ** 2)))
            best = min(best, (err, lam))
        self.lam = best[1]
        self._fit_krr(tr, self.lam, np.random.default_rng(self.seed))
        return self

    def predict(self, s):
        return np.clip(self.ar.predict(s) + self.krr.predict(self.scaler(s.cov)), 0.0, 1.0)

    def finetune(self, ft):
        self.ar.finetune(ft)
        self.scaler = Scaler(ft.cov)
        self._fit_krr(ft, self.lam, np.random.default_rng(self.seed))
        return self

    def correction(self, s) -> np.ndarray:
        return self.krr.predict(self.scaler(s.cov))


def make_models(seed: int = 0, with_extras: bool = True) -> list[Forecaster]:
    models: list[Forecaster] = [Persistence(), AR(), XGB(seed), LSTM(seed), ARPlusVQK("surrogate", seed=seed)]
    if with_extras:
        models += [ARPlusVQK("circuit", seed=seed), ARPlusVQK("rbf", seed=seed)]
    return models
