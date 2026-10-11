# Reproducing the experiments

This guide describes the code paths that generate the committed results. The experiments are CPU-based and may take substantially longer than starting the dashboard. Runtime depends on hardware and package versions.

## Requirements

- Python 3.10 or newer
- Node.js 20 or newer only for the browser dashboard
- A CPU; GPU and quantum hardware are not required
- Internet access to install packages on a new environment

Python dependency constraints are in `backend/requirements*.txt`; frontend dependencies are pinned in `frontend/package-lock.json`.

## Environment setup

### Windows

Run `start.bat` from the repository root to set up the backend virtual environment and frontend dependencies, start both services, and open the dashboard. The launcher keeps the Python environment under `%LOCALAPPDATA%\SunWind-Quantum-Forecast\venv` to avoid Windows path-length problems in deeply nested checkout folders.

### macOS and Linux

```bash
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements-dev.txt
cd frontend && npm ci
```

## Tests and static checks

```bash
cd backend
python -m pytest -q
ruff check .
```

Frontend type-check and production build:

```bash
cd frontend
npm ci
npm run build
```

## Data preparation and configuration

No external dataset download is needed for the default run. `backend/app/data/synthetic.py` creates deterministic hourly series for regions A, B, and C from seed values. The default experiment configuration, chronological splits, covariates, and seeds are in `backend/app/config.py` and the experiment scripts.

To try real observations, see `data/README.md` and the optional CSV loader in `backend/app/data/loaders.py`. The user is responsible for data provenance, terms of use, credentials, and compatibility; no real dataset is bundled or fetched automatically.

## Run stages

Run commands from `backend/`:

```bash
python -m scripts.run_experiments forecast --seeds 5
python -m scripts.run_experiments kernel --seeds 5
python -m scripts.run_experiments noise
python -m scripts.run_experiments simulator
```

Or run all four stages:

```bash
python -m scripts.run_experiments all --seeds 5
```

The `forecast` stage includes the in-domain comparison, transfer evaluation, importance summaries, and saved test-window prediction traces. `kernel` measures calm-versus-stormy separability. `noise` evaluates finite-shot and depolarising-noise emulations. `simulator` summarizes generated-series properties and checks the circuit/surrogate relationship.

## Outputs

By default, JSON results are written to the repository's `results/` directory:

- `forecast.json`
- `kernel.json`
- `noise.json`
- `simulator.json`

These outputs feed the dashboard and figure/export scripts. To write results to a separate location, set `QVK_RESULTS_DIR` before running the scripts; this also changes which results the API serves. Generated figures are written under `docs/figures/`. Some paper-export commands accept an explicit external paper directory; do not assume that directory exists in this repository.

## Seeds and interpretation

The simulator and experiment protocols use seeded NumPy generators. Re-running with the same software stack and seed is intended to reproduce the same simulated series and deterministic outputs; timing fields and numerical-library differences may vary. The saved forecasting tables aggregate five seeds, while the noise study uses its own three-seed protocol.

Reported errors are for fixed, region-parameterized synthetic scenarios and the documented chronological test windows. They are not estimates of real-world operational performance, and the scenario parameters are not fitted to a documented observational archive. The CSV utility is not wired into this runner. See [`experimental-status.md`](experimental-status.md) and the limitations in the main README before interpreting any result.
