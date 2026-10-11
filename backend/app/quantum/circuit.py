"""Six-qubit, three-layer hardware-efficient ansatz (PennyLane) used as the feature map.

Layer ``l`` applies  R_Y(theta[l,i] + alpha * x_i)  then  R_Z(phi[l,i])  on every qubit i, followed by a
nearest-neighbour CNOT chain 0->1->...->5. Parameters: 6 qubits x 2 rotations x 3 layers = 36 angles, drawn once at random and kept fixed (only alpha is tuned).
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

from .. import config as C


def init_params(n_qubits: int = C.N_QUBITS, n_layers: int = C.N_LAYERS, seed: int = 0) -> dict[str, np.ndarray]:
    """Xavier-style uniform initialisation, a = sqrt(6 / (fan_in + fan_out)) with fan_in = fan_out = n_qubits."""
    rng = np.random.default_rng(seed)
    a = np.sqrt(6.0 / (2 * n_qubits))
    return {"theta": rng.uniform(-a, a, (n_layers, n_qubits)), "phi": rng.uniform(-a, a, (n_layers, n_qubits))}


def n_parameters(params: dict[str, np.ndarray]) -> int:
    return int(params["theta"].size + params["phi"].size)


def _ansatz(x, theta, phi, alpha, n_qubits, n_layers):
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(theta[l, i] + alpha * x[..., i], wires=i)
        for i in range(n_qubits):
            qml.RZ(phi[l, i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


def make_state_qnode(n_qubits: int = C.N_QUBITS, n_layers: int = C.N_LAYERS):
    """QNode returning the full state vector; ``x`` of shape (B, n_qubits) is broadcast over the batch."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy", diff_method=None)
    def state(x, theta, phi, alpha):
        _ansatz(x, theta, phi, alpha, n_qubits, n_layers)
        return qml.state()

    return state


def make_overlap_qnode(n_qubits: int = C.N_QUBITS, n_layers: int = C.N_LAYERS):
    """Compute-uncompute circuit: P(0...0) of U(x')^dagger U(x) |0> equals the kernel value K_q(x, x')."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy", diff_method=None)
    def overlap(x1, x2, theta, phi, alpha):
        _ansatz(x1, theta, phi, alpha, n_qubits, n_layers)
        qml.adjoint(_ansatz)(x2, theta, phi, alpha, n_qubits, n_layers)
        return qml.probs(wires=range(n_qubits))

    return overlap


def draw_circuit(path: str | None = None):
    """Matplotlib drawing of the ansatz for the manuscript (Figure 5)."""
    qn = make_state_qnode()
    p = init_params()
    fig, _ = qml.draw_mpl(qn, decimals=None, style="pennylane", fontsize=11)(np.zeros(C.N_QUBITS), p["theta"], p["phi"], 1.0)
    return fig
