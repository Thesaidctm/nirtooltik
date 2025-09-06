import pathlib
import sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.preprocessors.spectro import (
    Detrend,
    EMSC,
    MSC,
    SNV,
    SavitzkyGolayTransformer,
    EPO,
)


rng = np.random.RandomState(0)
n_samples, n_features = 5, 50

# Data for SNV
X_snv = rng.randn(n_samples, n_features)

# Data for MSC
base_msc = rng.randn(n_features)
slopes = 1 + rng.uniform(-0.5, 0.5, size=n_samples)
slopes -= slopes.mean() - 1
offsets = rng.uniform(-0.5, 0.5, size=n_samples)
offsets -= offsets.mean()
X_msc = slopes[:, None] * base_msc + offsets[:, None]

# Data for Detrend
x = np.arange(n_features)
X_detrend = np.tile(0.5 * x + 1.0, (n_samples, 1))

# Data for Savitzky-Golay
x_sg = np.arange(n_features)
X_sg = np.tile(x_sg**2, (n_samples, 1))
sg_expected = np.tile(2 * x_sg, (n_samples, 1))

# Data for EMSC
x_emsc = np.linspace(0, 1, n_features)
base_emsc = rng.randn(n_features)
a = rng.uniform(-0.5, 0.5, size=n_samples)
a -= a.mean()
b = rng.uniform(-0.5, 0.5, size=n_samples)
b -= b.mean()
c = rng.uniform(-0.5, 0.5, size=n_samples)
c -= c.mean()
baseline = a[:, None] + b[:, None] * x_emsc + c[:, None] * x_emsc**2
X_emsc = base_emsc + baseline


def check_snv(X, Xt):
    np.testing.assert_allclose(Xt.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(Xt.std(axis=1), 1, atol=1e-12)


def check_msc(X, Xt):
    expected = np.tile(base_msc, (X.shape[0], 1))
    np.testing.assert_allclose(Xt, expected, atol=1e-12)


def check_detrend(X, Xt):
    np.testing.assert_allclose(Xt, 0, atol=1e-12)


def check_sg(X, Xt):
    np.testing.assert_allclose(Xt, sg_expected, atol=1e-12)


def check_emsc(X, Xt):
    expected = np.tile(base_emsc, (X.shape[0], 1))
    np.testing.assert_allclose(Xt, expected, atol=1e-12)


@pytest.mark.parametrize(
    "transformer, X, check",
    [
        pytest.param(SNV(), X_snv, check_snv, id="SNV"),
        pytest.param(MSC(), X_msc, check_msc, id="MSC"),
        pytest.param(Detrend(order=1), X_detrend, check_detrend, id="Detrend"),
        pytest.param(
            SavitzkyGolayTransformer(window_length=5, polyorder=2, deriv=1),
            X_sg,
            check_sg,
            id="SavitzkyGolay",
        ),
        pytest.param(EMSC(degree=2), X_emsc, check_emsc, id="EMSC"),
    ],
)
def test_preprocessors_basic(transformer, X, check):
    transformer.fit(X)
    Xt = transformer.transform(X)
    assert Xt.shape == X.shape
    assert not np.isnan(Xt).any()
    check(X, Xt)


def test_epo_preserves_shape_and_removes_z_component():
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5
    Z = rng.randn(n_features)
    X = rng.randn(n_samples, n_features)
    coeffs = rng.randn(n_samples, 1)
    X = X + coeffs * Z

    epo = EPO(Z=Z)
    epo.fit(X)
    Xt = epo.transform(X)

    assert Xt.shape == X.shape
    assert np.allclose(Xt @ Z, 0, atol=1e-10)
