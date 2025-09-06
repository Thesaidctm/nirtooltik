import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.utils.split import spxy_split


def test_spxy_split_distributes_spectral_clusters():
    # Two well-separated spectral clusters
    cluster_a = np.linspace(0, 0.09, 10)
    cluster_b = np.linspace(1, 1.09, 10)
    X = np.concatenate([cluster_a, cluster_b])[:, None]
    y = np.array([0] * 10 + [1] * 10)
    splits = spxy_split(X, y, n_splits=2)
    for _, val_idx in splits:
        vals = X[val_idx, 0]
        # Each fold should contain samples from both clusters
        assert vals.min() < 0.5
        assert vals.max() > 0.5
