import pathlib
import sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.processing.preprocessors.spectro import (
    AdaptiveQuantileSegmentNormalizer,
    Detrend,
    EMSC,
    MSC,
    SNV,
    SavitzkyGolayTransformer,
    EPO,
)


rng = np.random.RandomState(0)
n_samples, n_features = 5, 50
X = rng.randn(n_samples, n_features)


@pytest.mark.parametrize(
    "transformer",
    [
        SNV(),
        MSC(),
        Detrend(order=1),
        SavitzkyGolayTransformer(window_length=5, polyorder=2, deriv=1),
        EMSC(degree=2),
        AdaptiveQuantileSegmentNormalizer(min_segment_size=10, blend_strength=2.0),
    ],
    ids=["SNV", "MSC", "Detrend", "SavitzkyGolay", "EMSC", "AQS"],
)
def test_preprocessors_output_has_no_nans_and_preserves_shape(transformer):
    Xt = transformer.fit_transform(X)
    assert Xt.shape == X.shape
    assert np.isfinite(Xt).all()


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


def test_adaptive_quantile_segment_normalizer_learns_segments():
    aqs = AdaptiveQuantileSegmentNormalizer(min_segment_size=8)
    aqs.fit(X)
    assert hasattr(aqs, "_references_")
    assert len(aqs._references_) >= 2
    Xt = aqs.transform(X)
    assert Xt.shape == X.shape
    assert np.isfinite(Xt).all()

