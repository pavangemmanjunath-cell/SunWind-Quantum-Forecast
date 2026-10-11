"""CSV-loading utility for user-supplied data; it is not wired into the API or experiment runner.

To use the utility directly, place one CSV per region in ``data/raw/`` named ``region_<A|B|C>.csv`` with an hourly ISO timestamp column ``time`` and
the columns ``y`` (per-unit capacity factor, 0-1), ``wind_ms``, ``gust_ms``, ``pressure_hpa``, ``temp_c``, ``ghi_wm2``;
``dp3h_hpa`` (3-hour pressure change) is derived when missing and ``stormy`` (0/1 regime label) is derived from a gust
threshold when missing. Public sources that provide these quantities are listed in the manuscript (ENTSO-E Transparency,
NREL NSRDB, NOAA ISD, Open Power System Data, ARPA-E PERFORM, Global Wind Atlas); downloading and harmonising them is
left to the user because it depends on licences and API keys. No public weather or power API is queried by this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


def load_region_csv(region: str, path: str | None = None) -> pd.DataFrame:
    p = path or str(C.RAW_DATA_DIR / f"region_{region}.csv")
    df = pd.read_csv(p, parse_dates=["time"], index_col="time").sort_index()
    df = df.asfreq("h").interpolate(limit=6)
    missing = [c for c in ("y", "wind_ms", "gust_ms", "pressure_hpa", "temp_c", "ghi_wm2") if c not in df]
    if missing:
        raise ValueError(f"{p}: missing columns {missing}")
    if "dp3h_hpa" not in df:
        df["dp3h_hpa"] = df["pressure_hpa"].diff(3).fillna(0.0)
    if "stormy" not in df:
        df["stormy"] = (df["gust_ms"].rolling(6, min_periods=1).mean() > df["gust_ms"].quantile(0.8)).astype(np.int8)
    df["y"] = df["y"].clip(0.0, 1.0)
    return df.dropna()
