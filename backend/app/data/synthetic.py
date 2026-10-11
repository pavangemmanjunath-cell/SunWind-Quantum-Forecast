"""Stylized, region-parameterized synthetic generator for three renewable scenarios.

The generator couples a latent two-state *weather regime* (calm / stormy, a Markov chain with multi-hour
persistence) to wind speed, gusts, pressure, temperature and the clear-sky index, then converts the weather to
per-unit capacity factors with a PV model (clear-sky GHI, cell-temperature derating) and a turbine power curve.
Three regions with disjoint synoptic character are provided:

* ``A`` Iberian solar     - strong diurnal / seasonal cycle, few storms, high clear-sky index
* ``B`` North-Sea wind    - synoptic-scale wind persistence, frequent long storms, high mean wind
* ``C`` Mixed Texas       - both signals plus abrupt convective cloud transients and frequent fronts

The scenario parameters are fixed in this module and are not fitted to a documented observational dataset. Everything
is seeded; the same ``(region, seed)`` always gives the same series. This is a *simulation* for software and method
experiments, not measured regional weather or power data (see ``loaders.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .. import config as C


@dataclass(frozen=True)
class RegionSpec:
    key: str
    name: str
    lat_deg: float
    solar_share: float
    wind_share: float
    # wind (Weibull marginal on an AR(1) Gaussian copula)
    wind_c: float
    wind_k: float
    wind_phi: float
    wind_season_amp: float      # winter maximum (fraction of mean)
    wind_diurnal_amp: float
    # clouds
    cloud_gamma: float          # larger -> clearer sky
    cloud_phi: float
    convective: float           # fast cloud-transient amplitude
    # storms (Markov regime)
    calm_mean_h: float
    storm_mean_h: float
    storm_wind_boost: float
    storm_cloud_boost: float
    storm_dp_hpa: float
    # climate
    temp_mean: float
    temp_year_amp: float
    temp_day_amp: float
    pressure_mean: float


SPECS: dict[str, RegionSpec] = {
    "A": RegionSpec("A", "Iberian solar", 39.0, 0.85, 0.15, 6.2, 2.0, 0.975, 0.25, 0.12, 3.2, 0.985, 0.03,
                    300.0, 28.0, 0.55, 0.65, 9.0, 16.0, 9.0, 7.0, 1016.0),
    "B": RegionSpec("B", "North-Sea wind", 55.0, 0.03, 0.97, 10.4, 2.3, 0.992, 0.28, 0.03, 1.1, 0.992, 0.02,
                    70.0, 46.0, 0.70, 0.60, 14.0, 9.5, 6.0, 2.0, 1011.0),
    "C": RegionSpec("C", "Mixed Texas", 31.0, 0.50, 0.50, 7.6, 2.1, 0.965, 0.12, 0.20, 1.9, 0.975, 0.14,
                    95.0, 24.0, 0.85, 0.80, 11.0, 19.5, 11.0, 9.0, 1015.0),
}


def _hann_smooth(x: np.ndarray, width: int = 9) -> np.ndarray:
    w = np.hanning(width + 2)[1:-1]
    return np.convolve(x, w / w.sum(), mode="same")


def _markov_regime(n: int, calm_mean: float, storm_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Alternating calm/stormy spells with geometric durations -> 0/1 indicator of length ``n``."""
    out = np.zeros(n, dtype=np.int8)
    t, state = 0, 0
    while t < n:
        dur = int(rng.geometric(1.0 / (storm_mean if state else calm_mean)))
        out[t:t + dur] = state
        t += dur
        state ^= 1
    return out


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(n) * np.sqrt(1.0 - phi**2)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):          # tiny loop, 70k steps
        x[i] = phi * x[i - 1] + e[i]
    return x


def _power_curve(w: np.ndarray) -> np.ndarray:
    cin, rated, cout = 3.0, 12.0, 25.0
    p = np.where(w < cin, 0.0, np.where(w < rated, ((w - cin) / (rated - cin)) ** 3, 1.0))
    p = np.where(w >= cout, 0.0, p)
    k = np.array([0.25, 0.5, 0.25])
    return np.clip(np.convolve(p, k, mode="same") * 0.97, 0.0, 1.0)   # farm-level smoothing + availability


def _clear_sky_ghi(lat_deg: float, doy: np.ndarray, hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    dec = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    ha = np.deg2rad(15.0 * (hour + 0.5 - 12.0))
    cosz = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    cosz = np.clip(cosz, 0.0, 1.0)
    ghi = np.where(cosz > 0.02, 1098.0 * cosz * np.exp(-0.057 / np.maximum(cosz, 0.05)), 0.0)
    return ghi, cosz


def generate_region(key: str, seed: int = C.DEFAULT_SEED, start: str = C.START, end: str = C.END) -> pd.DataFrame:
    """Hourly frame with ``y`` (capacity factor), the six covariates, and the latent ``stormy`` label."""
    spec = SPECS[key]
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)
    rng = np.random.default_rng([seed, ord(key)])
    doy = idx.dayofyear.to_numpy().astype(float)
    hour = idx.hour.to_numpy().astype(float)
    year_phase = np.cos(2 * np.pi * (doy - 15.0) / 365.0)             # +1 in mid-winter (north)

    stormy = _markov_regime(n, spec.calm_mean_h, spec.storm_mean_h, rng)
    s = np.clip(_hann_smooth(stormy.astype(float), 11), 0.0, 1.0)      # frontal ramp in/out

    # ---- wind ------------------------------------------------------------------
    z = _ar1(n, spec.wind_phi, rng)
    u = np.clip(ndtr(z), 1e-6, 1 - 1e-6)
    w = spec.wind_c * (-np.log(1.0 - u)) ** (1.0 / spec.wind_k)
    w *= 1.0 + spec.wind_season_amp * year_phase
    w *= 1.0 + spec.wind_diurnal_amp * np.cos(2 * np.pi * (hour - 15.0) / 24.0)
    w *= 1.0 + spec.storm_wind_boost * s
    w = np.maximum(w, 0.0)
    gust = w * (1.12 + 0.10 * np.abs(rng.standard_normal(n)) + 0.55 * s * np.abs(rng.standard_normal(n)))

    # ---- pressure / temperature -------------------------------------------------
    syn = _ar1(n, 0.997, rng) * 6.0
    press = spec.pressure_mean + 4.0 * year_phase + syn - spec.storm_dp_hpa * s + 0.4 * rng.standard_normal(n)
    temp = (spec.temp_mean - spec.temp_year_amp * year_phase
            + spec.temp_day_amp * (1.0 - 0.6 * s) * np.cos(2 * np.pi * (hour - 15.0) / 24.0)
            + 2.0 * _ar1(n, 0.98, rng) - 2.5 * s)

    # ---- solar ---------------------------------------------------------------------
    ghi_cs, cosz = _clear_sky_ghi(spec.lat_deg, doy, hour)
    cloud = ndtr(_ar1(n, spec.cloud_phi, rng))
    cloud = np.clip(cloud + (1.0 - cloud) * spec.storm_cloud_boost * s, 0.0, 1.0)
    kt = 1.0 - 0.88 * cloud ** spec.cloud_gamma
    kt *= 1.0 + spec.convective * _ar1(n, 0.55, rng)
    kt = np.clip(kt, 0.04, 1.05)
    ghi = ghi_cs * kt
    t_cell = temp + 0.025 * ghi
    pv = np.clip(ghi / 1000.0 * (1.0 - 0.0040 * (t_cell - 25.0)) / 0.92, 0.0, 1.0)

    w_eff = np.maximum(w * (1.0 + 0.07 * rng.standard_normal(n)), 0.0)       # sub-hourly turbulence seen by the farm
    wind_cf = _power_curve(w_eff)
    pv = np.clip(pv * (1.0 + 0.05 * rng.standard_normal(n)), 0.0, 1.0)         # sub-hourly cloud edge effects
    y = np.clip(spec.solar_share * pv + spec.wind_share * wind_cf + 0.006 * rng.standard_normal(n), 0.0, 1.0)

    # covariates are *observations* (station / NWP-like), not the hidden truth: add measurement error
    w_obs = np.maximum(w * (1.0 + 0.09 * rng.standard_normal(n)) + 0.3 * rng.standard_normal(n), 0.0)
    gust_obs = np.maximum(gust * (1.0 + 0.10 * rng.standard_normal(n)), w_obs)
    ghi_obs = np.maximum(ghi * (1.0 + 0.10 * rng.standard_normal(n)), 0.0)
    press_obs = press + 0.6 * rng.standard_normal(n)
    dp3 = np.concatenate([np.zeros(3), press_obs[3:] - press_obs[:-3]])
    temp_obs = temp + 0.5 * rng.standard_normal(n)

    return pd.DataFrame({"y": y, "pv_cf": pv, "wind_cf": wind_cf, "wind_ms": w_obs, "gust_ms": gust_obs,
                         "pressure_hpa": press_obs, "temp_c": temp_obs, "ghi_wm2": ghi_obs, "dp3h_hpa": dp3,
                         "stormy": stormy.astype(np.int8), "storm_intensity": s}, index=idx)


def generate_all(seed: int = C.DEFAULT_SEED, **kw) -> dict[str, pd.DataFrame]:
    return {k: generate_region(k, seed, **kw) for k in C.REGIONS}
