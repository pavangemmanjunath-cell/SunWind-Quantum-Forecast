from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api.routes import router

app = FastAPI(title="SunWind-Quantum-Forecast API", version=__version__,
              description="Quantum-inspired variational kernel framework for cross-region solar and wind forecasting.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("QVK_CORS", "http://localhost:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
