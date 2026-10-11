"""Stage four: interpretation layer.

``TemplateExplainer`` turns the saved experiment result files into a structured paragraph. Every sentence is a deterministic
function of the numbers, so the text can be audited line by line against ``results/*.json``. The ``Explainer``
protocol makes it possible to plug in a generative-AI backend later; none is used for the text reported in the paper.
"""
from __future__ import annotations

import json
from typing import Protocol

import numpy as np

from .. import config as C


class Explainer(Protocol):
    def explain(self, forecast: dict, kernel: dict | None) -> str: ...


def _m(v) -> float:
    return float(np.mean(v))


class TemplateExplainer:
    def explain(self, forecast: dict, kernel: dict | None = None, region: str = "C") -> str:
        ind = forecast["indomain"][region]
        names = ["ARIMA (AR(24))", "XGBoost", "LSTM (2 layers)", "Proposed: ARIMA + VQK"]
        nr = {n: _m(ind[n]["nrmse"]) for n in names}
        best = min(nr, key=nr.get)
        prop = nr["Proposed: ARIMA + VQK"]
        ar = nr["ARIMA (AR(24))"]
        labels = {
            "ARIMA (AR(24))": "AR(24)",
            "Proposed: ARIMA + VQK": "Proposed: AR(24) + VQK",
        }
        s = [f"On the present simulated test corpus ({C.REGION_NAMES[region]}, one-step-ahead, {forecast['seeds']} simulated seeds) the "
             f"lowest-NRMSE model among those listed is {labels.get(best, best)} ({nr[best]:.4f})."]
        d = (prop - ar) / ar * 100
        s.append(f"The proposed AR(24) + quantum-inspired kernel configuration has NRMSE {prop:.4f}, "
                 f"numerically {abs(d):.1f}% {'higher' if d > 0 else 'lower'} than its AR(24) base ({ar:.4f}); "
                 "this descriptive difference does not establish statistical significance or real-world performance.")
        if kernel:
            q = [r["quantum_circuit"]["fdr_pc1"] for k in kernel["regions"] for r in kernel["regions"][k]]
            c = [r["rbf_tuned"]["fdr_pc1"] for k in kernel["regions"] for r in kernel["regions"][k]]
            ratio = _m(q) / max(_m(c), 1e-9)
            s.append(f"On the calm-versus-stormy regime task the leading-component Fisher ratio is {_m(q):.2f} for the circuit "
                     f"kernel and {_m(c):.2f} for the tuned radial-basis-function kernel (ratio {ratio:.2f}).")
            s.append("The quantum-inspired kernel shows clearer separability than the classical baseline." if ratio > 1.2 else
                     "The quantum-inspired kernel does not show clearer separability than the tuned classical kernel.")
        s.append("All results come from simulation; no claim of quantum advantage on physical hardware is made.")
        return " ".join(s)


def explain_from_files(results_dir=C.RESULTS_DIR) -> str:
    fc = json.loads((results_dir / "forecast.json").read_text())
    kp = results_dir / "kernel.json"
    return TemplateExplainer().explain(fc, json.loads(kp.read_text()) if kp.exists() else None)
