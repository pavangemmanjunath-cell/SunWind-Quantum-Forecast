"""Correctness checks: these guard the claims made in the manuscript."""
import numpy as np
import pandas as pd
import pytest

from app import config as C
from app.data.synthetic import generate_region
from app.forecasting import metrics as M
from app.forecasting.baselines import AR, LSTM, XGB
from app.forecasting.dataset import Scaler, make_supervised, split, time_features
from app.quantum import kernel as Q
from app.quantum.circuit import init_params, make_overlap_qnode, n_parameters

RNG = np.random.default_rng(0)


def test_circuit_has_36_parameters_and_six_qubits():
    assert n_parameters(init_params()) == 36 and C.N_QUBITS == 6 and C.N_LAYERS == 3


def test_circuit_kernel_matches_compute_uncompute():
    p = init_params(seed=1)
    X = RNG.normal(size=(4, 6))
    K = Q.circuit_kernel(X, X, p, 1.0)
    ov = make_overlap_qnode()
    for i in range(4):
        for j in range(4):
            v = float(ov(Q.prep(X[i]), Q.prep(X[j]), p["theta"], p["phi"], 1.0)[0])
            assert abs(v - K[i, j]) < 1e-9


def test_kernels_are_symmetric_psd_with_unit_diagonal():
    X = RNG.normal(size=(40, 6))
    for K in (Q.circuit_kernel(X, X, init_params(), 1.0), Q.rbf_kernel(X, X, 0.1)):
        assert np.allclose(K, K.T, atol=1e-10) and np.allclose(np.diag(K), 1.0, atol=1e-9)
        assert np.linalg.eigvalsh(K).min() > -1e-8


def test_surrogate_is_not_claimed_psd_but_is_bounded_and_symmetric():
    X = RNG.normal(size=(30, 6))
    K = Q.surrogate_kernel(X, X, 1.0)
    assert np.allclose(K, K.T) and K.max() <= 1.0 + 1e-9 and K.min() >= -1e-9


def test_depolarising_zero_noise_equals_exact():
    X = RNG.normal(size=(8, 6))
    p = init_params()
    assert np.allclose(Q.depolarising_kernel(X, p, 1.0, 0.0), Q.circuit_kernel(X, X, p, 1.0), atol=1e-8)


def test_kernel_ridge_interpolates_with_tiny_lambda():
    X = RNG.normal(size=(30, 6)); y = RNG.normal(size=30)
    m = Q.KernelRidge(lambda A, B: Q.rbf_kernel(A, B, 0.1), 1e-8).fit(X, y)
    assert np.allclose(m.predict(X), y, atol=1e-3)


def test_metrics_known_values():
    y = np.array([0.0, 1.0]); p = np.array([0.5, 0.5])
    assert M.nrmse(y, p) == pytest.approx(0.5)
    assert M.crps_surrogate(y, p) == pytest.approx(0.5)
    assert M.curtailment_regret(y, p) == pytest.approx(0.5 * (2 * 0.5 + 1 * 0.5))
    assert M.nrmse(y, y) == 0.0


@pytest.fixture(scope="module")
def df():
    return generate_region("C", seed=3, start="2018-01-01", end="2019-12-31 23:00")


def test_simulator_bounds_and_reproducibility(df):
    assert df["y"].between(0, 1).all() and df.isna().sum().sum() == 0
    assert len(df) == 24 * 730
    again = generate_region("C", seed=3, start="2018-01-01", end="2019-12-31 23:00")
    pd.testing.assert_frame_equal(df, again)
    assert not np.allclose(df["y"], generate_region("C", seed=4, start="2018-01-01", end="2019-12-31 23:00")["y"])


def test_solar_region_is_zero_at_night():
    d = generate_region("A", seed=1, start="2019-06-01", end="2019-06-30 23:00")
    night = d[d.index.hour.isin([0, 1, 2, 23])]
    assert night["pv_cf"].max() == 0.0   # PV is zero at night; any night output comes from the region's wind share


def test_supervised_has_no_lookahead(df):
    s = make_supervised(df)
    i = 100
    assert s.y[i] == df["y"].iloc[24 + i]            # target is y(t+1) ...
    assert s.lags[i, 0] == df["y"].iloc[23 + i]      # ... and the newest feature is y(t)
    assert s.index[i] == df.index[24 + i]
    np.testing.assert_allclose(s.cov[i], df.loc[df.index[23 + i], list(C.COVARIATES)].to_numpy())


def test_classical_model_feature_contracts(df):
    sup = make_supervised(df)
    mask = np.zeros(len(sup), dtype=bool)
    mask[100:103] = True
    samples = sup.take(mask)

    features = XGB._feat(samples)
    assert features.shape == (3, C.AR_ORDER + C.N_COV + 4)
    np.testing.assert_allclose(features[:, :C.AR_ORDER], samples.lags)
    np.testing.assert_allclose(features[:, C.AR_ORDER:C.AR_ORDER + C.N_COV], samples.cov)
    np.testing.assert_allclose(features[:, -4:], time_features(samples.index))
    np.testing.assert_allclose(AR._design(samples), np.column_stack([np.ones(len(samples)), samples.lags]))


def test_lstm_inputs_contain_ordered_lags_and_scaled_covariates(df):
    sup = make_supervised(df)
    mask = np.zeros(len(sup), dtype=bool)
    mask[100:103] = True
    samples = sup.take(mask)
    model = LSTM()
    model.scaler = Scaler(samples.cov)
    sequence, covariates = model._tensors(samples)

    assert tuple(sequence.shape) == (3, C.AR_ORDER, 1)
    assert tuple(covariates.shape) == (3, C.N_COV)
    np.testing.assert_allclose(sequence[0, :, 0].numpy(), samples.lags[0, ::-1])
    np.testing.assert_allclose(covariates.numpy(), model.scaler(samples.cov), atol=1e-6)


def test_split_dates_do_not_overlap():
    full = generate_region("C", seed=2)
    sp = split(make_supervised(full))
    assert sp.train.index.max() < sp.val.index.min() <= sp.val.index.max() < sp.test.index.min()
    assert sp.test.index.min() == pd.Timestamp(C.TEST_START) + pd.Timedelta(hours=0)
    assert sp.finetune.index.max() < sp.test.index.min()
    assert len(sp.test) == C.TEST_HOURS_IN_DOMAIN
