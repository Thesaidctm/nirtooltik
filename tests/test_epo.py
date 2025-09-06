import pathlib
import sys
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.preprocessors.spectro import EPO


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

