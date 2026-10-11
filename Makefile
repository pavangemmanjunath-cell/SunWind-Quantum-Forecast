.PHONY: install test lint frontend-install frontend-build experiments

install:
	python -m pip install -r backend/requirements-dev.txt

test:
	cd backend && python -m pytest -q

lint:
	cd backend && ruff check .

frontend-install:
	cd frontend && npm ci

frontend-build:
	cd frontend && npm run build

experiments:
	cd backend && python -m scripts.run_experiments all --seeds 5
