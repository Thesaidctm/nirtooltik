import pathlib
import sys

import numpy as np
import pytest

# Ensure the package can be imported when tests run from source
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from nircls.processing.models.neural import SpectralCNN, SpectralLSTM  # noqa: E402


@pytest.fixture
def dummy_data():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 10).astype(np.float32)
    y = (np.arange(20) % 2).astype(int)
    return X, y


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spectralcnn_fit_predict_device(dummy_data, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    X, y = dummy_data
    clf = SpectralCNN(
        n_epochs=1,
        batch_size=5,
        n_filters=4,
        kernel_size=3,
        hidden_dim=4,
        random_state=0,
        device=device,
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(X),)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spectrallstm_fit_predict_device(dummy_data, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    X, y = dummy_data
    clf = SpectralLSTM(
        n_epochs=1,
        batch_size=5,
        hidden_dim=4,
        num_layers=1,
        random_state=0,
        device=device,
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(X),)
