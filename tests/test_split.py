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


def test_spxy_split_handles_identical_samples_with_seed():
    X = np.ones((6, 2))
    y = np.zeros(6)
    splits_a = spxy_split(X, y, n_splits=3, random_state=42)
    splits_b = spxy_split(X, y, n_splits=3, random_state=42)
    normalized_a = [(
        tuple(sorted(tr.tolist())),
        tuple(sorted(val.tolist())),
    ) for tr, val in splits_a]
    normalized_b = [(
        tuple(sorted(tr.tolist())),
        tuple(sorted(val.tolist())),
    ) for tr, val in splits_b]
    assert normalized_a == normalized_b
    # Every index must appear exactly once in validation sets
    val_indices = np.concatenate([val for _, val in splits_a])
    assert sorted(val_indices.tolist()) == list(range(6))


def test_spxy_split_breaks_ties_randomly_when_needed():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    splits = spxy_split(X, y, n_splits=2, random_state=7)
    # Ensure both folds are non-empty and reproducible
    assert all(len(val) > 0 for _, val in splits)
    splits_again = spxy_split(X, y, n_splits=2, random_state=7)
    normalized = [(
        tuple(sorted(tr.tolist())),
        tuple(sorted(val.tolist())),
    ) for tr, val in splits]
    normalized_again = [(
        tuple(sorted(tr.tolist())),
        tuple(sorted(val.tolist())),
    ) for tr, val in splits_again]
    assert normalized == normalized_again
