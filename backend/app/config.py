"""Central constants shared by the simulator, models, experiments and API."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # backend/
RESULTS_DIR = Path(os.environ["QVK_RESULTS_DIR"]) if "QVK_RESULTS_DIR" in os.environ else ROOT.parent / "results"
FIGURE_DIR = ROOT.parent / "docs" / "figures"
ARTIFACT_DIR = ROOT / "artifacts"
RAW_DATA_DIR = ROOT.parent / "data" / "raw"           # drop real CSV files here (see app/data/loaders.py)

REGIONS = ("A", "B", "C")
REGION_NAMES = {"A": "Iberian solar", "B": "North-Sea wind", "C": "Mixed Texas"}

# --- calendar (hourly resolution) -----------------------------------------
START = "2018-01-01 00:00"
END = "2025-12-31 23:00"
TRAIN_END = "2023-12-31 23:00"      # training: 2018-2023
VAL_END = "2024-12-31 23:00"        # validation: 2024
TEST_START = "2025-06-02 00:00"     # test window inside 2025
TEST_HOURS_IN_DOMAIN = 14 * 24      # fourteen-day in-domain test window
TEST_HOURS_TRANSFER = 7 * 24        # one-week cross-region test window
FINETUNE_HOURS = 28 * 24            # target-region data available for fine-tuning

# --- forecasting ------------------------------------------------------------
AR_ORDER = 24
COVARIATES = ("wind_ms", "gust_ms", "pressure_hpa", "temp_c", "ghi_wm2", "dp3h_hpa")
N_COV = len(COVARIATES)

# --- quantum-inspired kernel ------------------------------------------------
N_QUBITS = 6
N_LAYERS = 3
N_PARAMS = N_QUBITS * 2 * N_LAYERS      # R_Y and R_Z per qubit per layer = 36

# --- decision problem (curtailment minimisation) -----------------------------
COST_SHORTFALL = 2.0   # a: over-forecast -> unit shortfall must be bought back
COST_CURTAIL = 1.0     # b: under-forecast -> surplus is curtailed

DEFAULT_SEED = 7


# CPU-only by design: no GPU and no quantum hardware are used anywhere in this project.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
DEVICE = "cpu"
