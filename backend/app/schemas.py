from __future__ import annotations

from pydantic import BaseModel, Field

from . import config as C


class PredictRequest(BaseModel):
    lags: list[float] = Field(..., min_length=C.AR_ORDER, max_length=C.AR_ORDER, description="y(t), y(t-1), ..., y(t-23)")
    covariates: list[float] = Field(..., min_length=C.N_COV, max_length=C.N_COV, description=", ".join(C.COVARIATES))


class PredictResponse(BaseModel):
    region: str
    ar: float
    correction: float
    forecast: float
