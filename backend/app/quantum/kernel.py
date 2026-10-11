"""Kernels: exact circuit kernel, closed-form surrogate (Eq. 5), tuned RBF, kernel ridge, kernel PCA and the Fisher ratio."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, solve

from .. import config as C
from .circuit import make_state_qnode

_STATE = None


def _qnode():
    global _STATE
    if _STATE is None:
        _STATE = make_state_qnode()
    return _STATE


def prep(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Clip standardised covariates to [-clip, clip]."""
    return np.clip(np.asarray(x, dtype=float), -clip, clip)


def quantum_states(X: np.ndarray, params: dict[str, np.ndarray], alpha: float = 1.0, batch: int = 4096) -> np.ndarray:
    X = prep(X)
    out = []
    for i in range(0, len(X), batch):
        out.append(np.asarray(_qnode()(X[i:i + batch], params["theta"], params["phi"], alpha)))
    return np.concatenate(out, axis=0)


def circuit_kernel(XA, XB, params, alpha: float = 1.0) -> np.ndarray:
    """K_q(x, x') = |<phi(x)|phi(x')>|^2 from exact state-vector simulation (Eq. 4)."""
    A = quantum_states(XA, params, alpha)
    B = A if XB is XA else quantum_states(XB, params, alpha)
    return np.abs(A.conj() @ B.T) ** 2


class CachedCircuitKernel:
    """Circuit kernel that remembers the state vectors of arrays it has already simulated (fast repeated inference)."""

    def __init__(self, params, alpha: float = 1.0):
        self.params, self.alpha, self._cache = params, alpha, {}

    def _states(self, X):
        key = (X.shape, hash(np.ascontiguousarray(X).tobytes()))
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = quantum_states(X, self.params, self.alpha)
        return self._cache[key]

    def __call__(self, XA, XB):
        return np.abs(self._states(XA).conj() @ self._states(XB).T) ** 2


def surrogate_kernel(XA, XB, alpha: float = 1.0) -> np.ndarray:
    """Closed-form surrogate of Eq. 5: exp(-||x-x'||^2 / 2) * cos^2(pi <x,x'> / 2) on alpha-scaled inputs."""
    A, B = alpha * prep(XA), alpha * prep(XB)
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-0.5 * np.maximum(d2, 0.0)) * np.cos(0.5 * np.pi * (A @ B.T)) ** 2


def rbf_kernel(XA, XB, gamma: float) -> np.ndarray:
    A, B = prep(XA), prep(XB)
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


class KernelRidge:
    """alpha* = (K + lambda I)^{-1} eps  (Eq. 6, closed form)."""

    def __init__(self, kernel, lam: float = 10.0):
        self.kernel, self.lam = kernel, lam

    def fit(self, X: np.ndarray, eps: np.ndarray):
        self.X = X
        K = self.kernel(X, X)
        self.coef = solve(K + self.lam * np.eye(len(X)), eps, assume_a="gen")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.kernel(X, self.X) @ self.coef


def kernel_pca_scores(K: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project the samples used to build ``K`` onto its leading kernel principal components."""
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    w, v = eigh((Kc + Kc.T) / 2.0, subset_by_index=[n - n_components, n - 1])
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    return v * np.sqrt(np.maximum(w, 0.0))


def fisher_ratio(z: np.ndarray, labels: np.ndarray) -> float:
    """Two-class Fisher discriminant ratio of a 1-D projection: (m1 - m0)^2 / (s1^2 + s0^2)."""
    a, b = z[labels == 1], z[labels == 0]
    return float((a.mean() - b.mean()) ** 2 / (a.var(ddof=1) + b.var(ddof=1) + 1e-12))


# ---------------------------------------------------------------------------------------------------------
# hardware-noise emulation of the circuit kernel
# ---------------------------------------------------------------------------------------------------------
def shot_noise_kernel(K: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Estimate every kernel entry from ``shots`` measurements of the compute-uncompute circuit (binomial sampling)."""
    n = len(K)
    Kc = np.clip(K, 0.0, 1.0)
    est = rng.binomial(shots, Kc[np.triu_indices(n, 1)]) / shots
    out = np.eye(n)
    out[np.triu_indices(n, 1)] = est
    return out + np.triu(out, 1).T


def depolarising_kernel(X: np.ndarray, params: dict[str, np.ndarray], alpha: float, p: float) -> np.ndarray:
    """Tr(rho(x) rho(x')) for states prepared with a depolarising channel of strength ``p`` after every layer, per qubit."""
    import pennylane as qml


    dev = qml.device("default.mixed", wires=C.N_QUBITS)

    @qml.qnode(dev, interface="numpy", diff_method=None)
    def rho(x, theta, phi, a):
        for l in range(C.N_LAYERS):
            for i in range(C.N_QUBITS):
                qml.RY(theta[l, i] + a * x[..., i], wires=i)
            for i in range(C.N_QUBITS):
                qml.RZ(phi[l, i], wires=i)
            for i in range(C.N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
            if p > 0:
                for i in range(C.N_QUBITS):
                    qml.DepolarizingChannel(p, wires=i)
        return qml.density_matrix(wires=range(C.N_QUBITS))

    R = np.asarray(rho(prep(X), params["theta"], params["phi"], alpha))
    n = len(R)
    flat = R.reshape(n, -1)
    flat_t = np.transpose(R, (0, 2, 1)).reshape(n, -1)
    return np.real(flat @ flat_t.T)
