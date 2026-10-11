# Contributing

Thank you for helping improve SunWind-Quantum-Forecast. Changes should preserve scientific traceability and distinguish simulations from observed data.

## Before opening an issue or pull request

- Check existing issues and describe the problem or research question clearly.
- For bugs, provide a minimal reproduction, environment, expected behavior, actual behavior, and relevant logs.
- For result or methodology changes, identify the affected protocol, seed handling, data provenance, and expected generated artifacts.
- Do not include credentials, private datasets, or personal information in issues or commits.

## Development setup

Requirements are Python 3.10+ and, for the dashboard, Node.js 20+. On Windows, `start.bat` creates the backend virtual environment and installs dependencies. For targeted backend development:

```bash
python -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements-dev.txt
cd backend
python -m pytest -q
ruff check .
```

For the frontend:

```bash
cd frontend
npm ci
npm run build
```

## Pull request expectations

- Keep changes focused and explain why they are needed.
- Add or update tests for behavior changes.
- Run the relevant backend tests/linter and frontend type-check/build.
- Update documentation when commands, data assumptions, metrics, or outputs change.
- Never make benchmark claims without identifying data, protocol, seeds, and metric; do not replace committed results merely to improve them.
- Keep generated or externally sourced data out of commits unless its provenance and redistribution license are clear.
- Do not add DOI, publication, adoption, or performance claims without verifiable evidence.

By submitting a contribution, you agree that it may be distributed under the repository's MIT license.
