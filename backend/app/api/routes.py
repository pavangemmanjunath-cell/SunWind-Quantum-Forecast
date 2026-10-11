from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from .. import config as C
from ..explain.narrator import TemplateExplainer
from ..forecasting.dataset import Supervised
from ..schemas import PredictRequest, PredictResponse
from ..services import registry

router = APIRouter(prefix="/api")


def _region(r: str) -> str:
    if r not in C.REGIONS:
        raise HTTPException(404, f"unknown region {r!r}; choose one of {C.REGIONS}")
    return r


@router.get("/health")
def health():
    return {"status": "ok", "regions": C.REGIONS, "qubits": C.N_QUBITS, "layers": C.N_LAYERS, "parameters": C.N_PARAMS}


@router.get("/regions")
def regions():
    return [{"key": k, "name": C.REGION_NAMES[k]} for k in C.REGIONS]


@router.get("/series")
def series(region: str = "C", seed: int = C.DEFAULT_SEED, start: str = C.TEST_START, hours: int = Query(168, ge=24, le=24 * 60)):
    df = registry.region_frame(_region(region), seed)
    s = df.loc[start:].iloc[:hours]
    return {"region": region, "t": [str(i) for i in s.index], **{c: s[c].round(4).tolist() for c in
            ("y", "pv_cf", "wind_cf", "wind_ms", "gust_ms", "pressure_hpa", "temp_c", "ghi_wm2", "dp3h_hpa", "stormy")}}


@router.get("/results/{name}")
def get_results(name: str):
    if name not in {"forecast", "kernel", "noise", "simulator"}:
        raise HTTPException(404, "unknown result set")
    r = registry.results(name)
    if r is None:
        raise HTTPException(404, f"{name}.json not found; run scripts.run_experiments {name}")
    return r


@router.get("/forecast/week")
def forecast_week(region: str = "C"):
    r = registry.results("forecast")
    if r is None:
        raise HTTPException(404, "forecast.json not found")
    return r["week_traces"][_region(region)]


@router.get("/kernel/embedding")
def kernel_embedding(region: str = "C", alpha: float = Query(0.5, gt=0, le=4), gamma: float = Query(0.1, gt=0, le=5),
                     n: int = Query(150, ge=40, le=300), seed: int = 7):
    return registry.embedding(_region(region), alpha, gamma, n, seed)


@router.post("/predict/{region}", response_model=PredictResponse)
def predict(region: str, body: PredictRequest):
    m = registry.proposed_model(_region(region))
    cov = np.array([body.covariates])
    s = Supervised(None, np.array([body.lags]), cov, np.zeros(1), np.zeros(1))
    ar = float(m.ar.predict(s)[0])
    corr = float(m.correction(s)[0])
    return PredictResponse(region=region, ar=ar, correction=corr, forecast=float(np.clip(ar + corr, 0, 1)))


@router.get("/explain")
def explain(region: str = "C"):
    fc, kp = registry.results("forecast"), registry.results("kernel")
    if fc is None:
        raise HTTPException(404, "forecast.json not found")
    return {"text": TemplateExplainer().explain(fc, kp, _region(region))}
