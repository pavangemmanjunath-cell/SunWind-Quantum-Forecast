# SunWind-Quantum-Forecast

**Quantum-inspired kernels for solar and wind forecasting across regions.**

SunWind-Quantum-Forecast is ongoing experimental research software for comparing a variational quantum-inspired kernel residual model with classical forecasting baselines across three fixed-parameter synthetic renewable-energy scenarios.

> **Experimental research — ongoing work:** this repository is a research prototype, not a validated operational forecasting system. The committed results and interactive replay use synthetic data; they are not measurements or evidence of real-world forecast accuracy. Further data integration, calibration, independent validation, and improvement are still required. The kernel is evaluated by classical simulation; this project does not use quantum hardware or claim quantum advantage.

## Overview

The code studies one-step-ahead renewable capacity-factor forecasts from six weather covariates and 24 autoregressive lags. It includes persistence, AR(24), XGBoost, a two-layer LSTM, a kernel-ridge residual model, and kernel/separability controls. Its audience is researchers and developers who need to inspect, test, or reproduce this specific experimental pipeline.

The repository includes a FastAPI service, an interactive React dashboard, deterministic synthetic time series, stored experiment outputs, plots, and tests. The dashboard replays stored results and runs the kernel-embedding experiment on request; it does not stream live weather or grid data.

## Why this use case matters

For organizations that operate or plan around renewable generation—such as solar and wind farm operators, utilities, and grid planners—better short-horizon estimates of available generation could help inform decisions like scheduling storage, balancing supply and demand, or planning reserve capacity. These are potential applications of renewable forecasting generally, not capabilities or benefits demonstrated by this prototype.

This project is useful today as a transparent research and learning environment: it makes a forecasting pipeline inspectable, compares the implemented models under a controlled synthetic setup, and exposes where further data integration and validation are needed. Its current results do **not** show that it can improve operational decisions, reduce costs, or increase renewable-energy use. Those outcomes would require evaluation with appropriately licensed, representative real-world weather and generation data, independent validation, and assessment within the intended operational workflow.

## Research question and motivation

Weather-driven variability makes solar and wind generation difficult to forecast, while the usefulness of additional model structure must be tested against strong classical baselines. This repository asks whether the implemented quantum-inspired kernel can add useful residual information or improve calm-versus-stormy separability across distinct simulated regions. It is an experimental comparison, not a claim that the proposed method is superior.

## What is implemented

- **Scientific:** simulated cross-region forecasting and calm/stormy kernel-separability studies, with limitations stated alongside results.
- **Algorithmic:** an AR(24) baseline, standard classical comparators, and a kernel-ridge residual model using a fixed, six-qubit, three-layer feature map. The circuit angles are fixed random features, not trained parameters.
- **Software:** a Python/FastAPI backend and a React/TypeScript/Vite dashboard with a 3-D regional replay, model comparisons, experiment tables, and an interactive kernel lab.
- **Reproducibility:** seeded simulation and experiment scripts, committed JSON outputs, figures, and a backend test suite.

## Results snapshot

The committed five-seed, simulated-data results for region C (Mixed Texas) report these mean NRMSE values:

| Model | Mean NRMSE |
|---|---:|
| XGBoost | 0.0515 |
| AR(24) | 0.0587 |
| AR(24) + VQK (surrogate kernel) | 0.0586 |
| AR(24) + RBF kernel control | 0.0582 |

Lower NRMSE is better. These figures describe the committed simulator run only; they do not establish performance on measured data or a statistically meaningful advantage for VQK. See [`results/`](./results/) for the full stored output and [`docs/reproducibility.md`](./docs/reproducibility.md) for rerunning experiments.

## Architecture and methodology

The pipeline is:

1. Generate hourly region-parameterized synthetic series for three scenarios. A CSV-reading utility exists, but it is not wired into the API or experiment runner.
2. Create a supervised forecasting set from 24 target lags and six covariates, with chronological train/validation/test partitions.
3. Fit and evaluate the classical baselines and the AR-plus-kernel-ridge residual model.
4. Persist experiment summaries as JSON. The API serves these artifacts and computes live kernel embeddings for the lab.
5. Display the regional replay, model comparisons, and kernel results in the browser.

See [`docs/architecture.md`](./docs/architecture.md), [`data/README.md`](./data/README.md), and [`docs/reproducibility.md`](./docs/reproducibility.md) for component, dataset, and reproduction details.

## Quick start

### Windows

Install Python 3.10 or newer and Node.js 20 or newer, then run `start.bat` from the repository root (or double-click it). The script creates a local Python virtual environment, installs the backend requirements, installs frontend dependencies when they are missing, starts both services, waits for their health checks, and opens the dashboard.

- Dashboard: <http://127.0.0.1:5173>
- API health: <http://127.0.0.1:8000/api/health>
- Interactive API docs: <http://127.0.0.1:8000/docs>

The first run requires an internet connection for dependency installation. Two terminal windows stay open for the API and frontend logs; close those windows to stop the services.

### macOS and Linux

From the repository root, use two terminals:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements-dev.txt
cd frontend && npm ci
```

Terminal 1:

```bash
cd backend
../backend/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
npm run dev -- --host 127.0.0.1
```

Open <http://127.0.0.1:5173>. To use a different Python executable, create the virtual environment with that interpreter.

## Using the dashboard and API

The **Regions** view animates a synchronized 14-day test-window replay and compares forecasts. **Kernel Lab** computes an embedding on request. **Experiments** reads the stored benchmark, transfer, noise, and simulator JSON. **About** summarizes the method and data provenance.

Example API calls (with the backend running):

```text
GET  http://127.0.0.1:8000/api/health
GET  http://127.0.0.1:8000/api/regions
GET  http://127.0.0.1:8000/api/series?region=C&hours=168
GET  http://127.0.0.1:8000/api/results/forecast
```

Forecast and experiment output files are in [`results/`](./results/). Synthetic series are generated by [`backend/app/data/synthetic.py`](./backend/app/data/synthetic.py); they are generated in memory rather than downloaded as a bundled raw dataset. Optional CSV input is described in [`data/README.md`](./data/README.md).

## Reproducing the experiments

```bash
cd backend
python -m pytest -q
python -m scripts.run_experiments forecast --seeds 5
python -m scripts.run_experiments kernel --seeds 5
python -m scripts.run_experiments noise
python -m scripts.run_experiments simulator
```

To run all stages, use `python -m scripts.run_experiments all --seeds 5`. Experiments can be CPU-intensive; the dashboard does not start them automatically. Full instructions and expected artifacts are in [`docs/reproducibility.md`](./docs/reproducibility.md).

## Tests and continuous integration

Run backend tests and linting with:

```bash
cd backend
python -m pytest -q
ruff check .
```

Validate the frontend with:

```bash
cd frontend
npm ci
npm run build
```

GitHub Actions runs these checks for pushes and pull requests. Tests cover API behavior and core forecasting/kernel invariants; they do not validate performance on real-world data.

## Experimental status and limitations

- The three named regional series are stylized simulations with fixed parameters, not data measured at those locations and not fitted to a documented observational archive.
- The test/validation scores measure behavior within the synthetic setup only. They do not establish accuracy, generalization, or operational utility on real weather or power data.
- The dashboard's local API does not fetch external weather or grid data. XGBoost and LSTM benchmark traces are saved from an offline experiment run; only the proposed AR(24) + VQK model has an interactive prediction endpoint.
- This is ongoing experimental research. Results and model choices remain subject to improvement and independent validation; do not treat them as deployment recommendations.
- Absolute errors and model rankings may differ on observations. External data must be obtained and licensed by the user; the CSV utility is not yet connected to the experiments.
- Circuit angles are fixed random features; only the documented scaling parameter is tuned.
- The dashboard's explanation text is a deterministic template over saved result files.
- All kernel/circuit calculations run classically. No quantum advantage is claimed.

See [`docs/experimental-status.md`](./docs/experimental-status.md) for the model-by-model input contract, API data flow, references consulted, and a concrete external-validation plan.

## Citation

If you use this software, cite it using the metadata in [`CITATION.cff`](./CITATION.cff). GitHub can render the citation when the file is present. The associated manuscript is identified in the source project as *A Quantum-Inspired Variational Kernel Framework for Cross-Region Solar and Wind Energy Forecasting*. Publication and DOI details are not documented here, so do not cite an invented DOI.

## Contributing and support

Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) before opening a pull request. Use the repository's issue forms for bugs, feature proposals, or research questions. Security reporting guidance is in [`SECURITY.md`](./SECURITY.md).

## License and project metadata

The project is distributed under the MIT License; see [`LICENSE`](./LICENSE). The GitHub repository is [pavangemmanjunath-cell/SunWind-Quantum-Forecast](https://github.com/pavangemmanjunath-cell/SunWind-Quantum-Forecast).

Suggested GitHub topics: `renewable-energy`, `solar-forecasting`, `wind-forecasting`, `time-series`, `machine-learning`, `quantum-kernel`, `quantum-inspired`, `energy-systems`, `reproducible-research`.
