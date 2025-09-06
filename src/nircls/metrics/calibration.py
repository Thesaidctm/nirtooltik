
import numpy as np


def expected_calibration_error(y_true, y_proba, n_bins=15):
    r"""
    Compute the expected calibration error (ECE).

    ECE is defined as the weighted average over *n_bins* of the absolute
    difference between the mean predicted confidence and the empirical
    accuracy for samples whose confidence falls in each bin.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_proba : array-like of shape (n_samples, n_classes)
        Predicted class probabilities for each sample.
    n_bins : int, default=15
        Number of equally spaced bins used to discretize the [0, 1] interval.

    Returns
    -------
    float
        The expected calibration error:

        .. math::
            \sum_{b=1}^{n_{bins}}
            \left| \text{acc}_b - \text{conf}_b \right|
            \frac{n_b}{n}

        where :math:`\text{acc}_b` and :math:`\text{conf}_b` are the mean
        accuracy and confidence of predictions in bin :math:`b`, respectively.

    Raises
    ------
    ValueError
        If the input arrays do not have compatible shapes or contain invalid
        probabilities.
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_proba.ndim != 2:
        raise ValueError("y_proba must be a 2D array.")
    if y_proba.shape[0] != y_true.shape[0]:
        raise ValueError(
            "y_true and y_proba must have the same number of samples."
        )
    if not np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each row of y_proba must sum to 1.")

    conf = y_proba.max(axis=1)
    preds = y_proba.argmax(axis=1)
    acc = (preds == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = (conf >= lower) & (conf < upper if i < n_bins - 1 else conf <= upper)
        if not np.any(mask):
            continue
        ece += abs(acc[mask].mean() - conf[mask].mean()) * mask.mean()
    return float(ece)
