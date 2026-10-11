# Experimental status, model inputs, and validation

## Research status

SunWind-Quantum-Forecast is ongoing experimental research software. It is not a validated operational forecasting service. The committed benchmark results are reproducible outputs of this repository's synthetic generator; they do not establish real-world accuracy, generalization, or operational usefulness. The model pipeline, data integration, and simulator still require improvement and independent validation.

The three scenario names (Iberian solar, North-Sea wind, and mixed Texas) describe intended scenario profiles. They are not evidence that the series were measured at those locations. The simulator parameters are fixed in `backend/app/data/synthetic.py`; this code does not fit them to a documented observed weather or generation archive. Its checks cover generated-series bounds, distributions, diurnal patterns, autocorrelation, and reproducibility. Those are internal consistency checks, not empirical validation.

## Forecast inputs and target alignment

Every supervised sample uses the information available at hour `t` to forecast the next hour's capacity factor `y(t+1)`. In `backend/app/forecasting/dataset.py`, the 24 target lags are `y(t), y(t-1), ..., y(t-23)`, and the six covariates are read at `t`: `wind_ms`, `gust_ms`, `pressure_hpa`, `temp_c`, `ghi_wm2`, and `dp3h_hpa`. The target timestamp is `t+1`; its hour-of-day and day-of-year are known calendar features, not future measurements. The chronological partitions train through 2023, validate in 2024, and evaluate 2025 windows. Tests assert the lag/covariate alignment and model feature shapes.

| Model | Inputs passed to the implementation | Output |
|---|---|---|
| Persistence | `y(t)` | `y(t+1) = y(t)` |
| AR(24) | 24 lags and an intercept; no weather covariates | Least-squares prediction of `y(t+1)` |
| XGBoost | 24 lags + six covariates at `t` + four sine/cosine calendar features for target time `t+1` = 34 features | One-step point prediction |
| Two-layer LSTM | 24 lagged values ordered oldest-to-newest + six covariates at `t`, scaled using training-period statistics | Predicted increment `y(t+1)-y(t)`, added to `y(t)` |
| AR(24) + VQK | AR(24) uses the 24 lags; kernel-ridge residual correction uses the six covariates at `t` | AR prediction plus the learned correction |

The string `ARIMA (AR(24))` is retained in saved result JSON as a legacy identifier. The implementation is an autoregression of order 24 with an intercept; it does not difference the series and does not include a moving-average term. The browser presents the mathematically accurate label `AR(24)`.

## How the baselines are trained

- **AR(24):** least-squares fit with an intercept on the training-period lags.
- **XGBoost:** `XGBRegressor` with at most 500 trees, depth 4, learning rate 0.05, row and column subsampling 0.8, and early stopping after 30 rounds without validation improvement. `fit` receives the chronological 2024 validation split as `eval_set`; the implementation retains trees through the best iteration. The official XGBoost guide describes this validation-set/early-stopping pattern.
- **LSTM:** two recurrent layers with 32 hidden units, a separate linear projection of the six scaled covariates, and a prediction head for the persistence increment. It uses Adam at learning rate 0.002, batches of 512, at most eight epochs, and at most 30,000 randomly selected training windows per epoch. Validation RMSE chooses the best checkpoint, with patience three. Fine-tuning, when used in the transfer experiment, uses five epochs at learning rate 0.001.

## What the local API does (and does not do)

The application's `/api` is an interface between the local React dashboard and the local FastAPI backend. It is **not** a connection to an external weather, solar, wind, or grid-data provider.

- `GET /api/series` returns generated synthetic time series.
- `GET /api/results/{name}` and `GET /api/forecast/week` serve committed JSON benchmark outputs and saved forecast traces.
- `POST /api/predict/{region}` accepts 24 lags and six covariates and runs the proposed AR(24) + VQK predictor only.
- The XGBoost and LSTM forecasts displayed in the regional replay were produced by the offline experiment runner and saved in `results/`; those two models are not run by interactive prediction API endpoints.
- `backend/app/data/loaders.py` can parse user-provided hourly CSVs, but it is not connected to the registry, API, or experiment runner. The current results therefore do not consume API-fetched or user-supplied real data.

## References consulted and what they support

The following references were consulted during this review as methodological context for describing component forms and future validation sources. They do **not** validate the implementation, the scenario parameter values, the turbine curve, the simulated records, or the forecasting results. The simulator is not a full implementation of any one referenced package or model.

1. Dobos, A. P. (2014). *PVWatts Version 5 Manual*. NREL/TP-6A20-62641. [https://doi.org/10.2172/1158421](https://doi.org/10.2172/1158421). Context for an irradiance-scaled photovoltaic DC model with temperature-coefficient adjustment. This project uses a simplified PVWatts-style formula; it does not implement the complete PVWatts model.
2. Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). pvlib python: a Python package for modeling solar energy systems. *Journal of Open Source Software, 3*(29), 884. [https://doi.org/10.21105/joss.00884](https://doi.org/10.21105/joss.00884). Reference for the open-source PV modeling ecosystem, not evidence that this project calls pvlib.
3. Justus, C. G., Hargraves, W. R., Mikhail, A., & Graber, D. (1978). Methods for estimating wind speed frequency distributions. *Journal of Applied Meteorology, 17*(3), 350–353. [https://doi.org/10.1175/1520-0450(1978)017%3C0350%3AMFEWSF%3E2.0.CO%3B2](https://doi.org/10.1175/1520-0450(1978)017%3C0350%3AMFEWSF%3E2.0.CO%3B2). Context for Weibull wind-speed distributions; it does not determine or validate this generator's regional parameters.
4. [NOAA/NCEI Integrated Surface Database (ISD)](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) describes a potential source for future weather validation. Its station coverage and record continuity vary; no ISD data were used here.
5. The official [XGBoost scikit-learn estimator guide](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html) documents the estimator and validation-set-based early stopping pattern. The implementation passes its chronological 2024 validation split as `eval_set`; the guide is an implementation reference, not a comparison or performance claim.
6. The official [PyTorch `torch.nn.LSTM` reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) documents the multilayer recurrent module used by the two-layer baseline. It does not establish that this project's architecture is optimal.

## What would be needed for stronger accuracy claims

Before claiming that the simulator or forecasts are accurate for a physical region, future work should:

1. Connect documented, licensed, source-specific weather and power observations or archive APIs; record data versions, station/region identifiers, units, missing-data rules, and licenses.
2. Fit simulator parameters only on declared calibration periods, then compare distributions, seasonal/diurnal profiles, autocorrelation, storm durations, wind power curves, and solar output against held-out observed periods.
3. Evaluate forecasting models on independent sites and years with time-aware splits and leakage checks; compare against transparent persistence and standard statistical baselines.
4. Report uncertainty across regions/seeds, sensitivity to parameters, and failed as well as successful checks.
5. Keep conclusions explicitly limited to the studied data and protocol; repeat validation after material code or data changes.
