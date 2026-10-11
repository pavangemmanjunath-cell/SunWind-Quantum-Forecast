# Dataset and provenance

## Default dataset

No external or measured dataset is bundled. By default, series are produced deterministically in memory by [`backend/app/data/synthetic.py`](../backend/app/data/synthetic.py). The generator models stylized hourly solar and wind behavior for three region-parameterized scenarios:

| Key | Scenario |
|---|---|
| A | Iberian solar |
| B | North-Sea wind |
| C | Mixed Texas |

The configured date range is 2018-01-01 through 2025-12-31 at hourly resolution. The fixed scenario parameters are not fitted to a documented observational dataset. The series are simulations intended to represent qualitative features; they are not measurements and should not be described as an operational or public dataset. No external data license applies to the generated series independently of this repository's MIT-licensed code. Do not infer validation against actual regional weather or power records.

Forecast and experiment result snapshots are stored separately in [`results/`](../results/). The dashboard replays forecast traces from `results/forecast.json`; experiment scripts can regenerate result JSON files.

## Optional CSV input

The CSV-reading utility is [`backend/app/data/loaders.py`](../backend/app/data/loaders.py). It expects user-supplied files at `data/raw/region_A.csv`, `data/raw/region_B.csv`, or `data/raw/region_C.csv`, with an hourly timestamp column `time` and these columns:

- `y`: per-unit capacity factor, clipped to `[0, 1]`
- `wind_ms`
- `gust_ms`
- `pressure_hpa`
- `temp_c`
- `ghi_wm2`
- `dp3h_hpa` (derived from pressure if absent)
- `stormy` (derived from a gust threshold if absent)

Rows are sorted by timestamp, resampled hourly, and gaps of up to six hours are interpolated. Review the loader implementation before using this preprocessing for research.

No CSVs are downloaded automatically. Before adding or distributing any real data, document its source, version, license, access requirements, and preprocessing. Keep private or restricted data out of version control; `data/raw/` is ignored by Git.

**Integration status:** this utility is not currently called by the API registry or experiment runner. All committed result files and browser replay traces use the synthetic generator. Connecting real sources, harmonizing schemas, and rerunning external validation are ongoing research/software tasks, not implemented features.
