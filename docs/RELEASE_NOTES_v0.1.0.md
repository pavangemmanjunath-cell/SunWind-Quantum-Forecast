# SunWind-Quantum-Forecast v0.1.0

**First experimental release · 11 October 2026**

## Overview

SunWind-Quantum-Forecast is an experimental research prototype for comparing renewable-energy forecasting models across three region-parameterized synthetic scenarios. This release provides the backend, interactive dashboard, saved experiment results, reproducibility material, and a Windows setup launcher.

## What is included

- FastAPI backend serving synthetic series, saved experiment outputs, forecast traces, and kernel embeddings.
- React dashboard for regional replay, model comparisons, saved benchmark results, and interactive kernel exploration.
- Comparisons involving persistence, AR(24), XGBoost, LSTM, and AR(24) with variational quantum-inspired kernel methods.
- Committed experiment results and figures, with instructions for reproducing experiments.
- Windows `start.bat` launcher for setting up and starting the backend and dashboard.
- GitHub Actions checks for backend tests and linting, and frontend type-checking and production build.

## Potential use case

Short-horizon renewable-generation forecasts can potentially help solar and wind operators, utilities, and grid planners consider decisions such as storage scheduling, supply-demand balancing, and reserve planning. These are potential applications of forecasting in general; this experimental release does not demonstrate those operational benefits.

## Research status and limitations

This is ongoing experimental research software, not a validated operational forecasting system. The included benchmark results and dashboard replay use synthetic data; they are not observations and do not establish performance on real weather or power-generation data. The dashboard does not fetch live weather or grid data.

Real-world accuracy, generalization, and operational benefits have not been demonstrated. Further integration of appropriately licensed real-world data, simulator calibration, independent validation, and model improvement are required before operational use can be assessed.

The kernel circuit is evaluated using classical simulation. This release makes no claim of quantum advantage or physical quantum-hardware performance.

## Verification

For this release commit, the backend suite passed **17 tests**, Ruff lint passed, the frontend TypeScript check and production build passed, and the GitHub Actions CI workflow completed successfully.

## Authors

- Pavan Manjunath — [ORCID: 0000-0002-2694-8807](https://orcid.org/0000-0002-2694-8807)
- Thomas Prüfer

No Zenodo record or DOI is associated with this release.
