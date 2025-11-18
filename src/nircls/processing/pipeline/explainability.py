from __future__ import annotations

import logging
import gc
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import traceback
import warnings
from functools import lru_cache
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from collections import OrderedDict
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
import psutil
import sklearn
import yaml  # type: ignore[import-untyped]
from joblib import dump, load
from rich.console import Console
from rich.progress import Progress
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneOut,
    ParameterGrid,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

from ..config_schema import ConfigModel, ValidationError
from .feature_selection.vip import compute_vip
from .io import export_xy_xlsx
from .io import prepare_inputs as _prepare_inputs
from .io import read_tabular, read_xlsx_xy, summarize_dataset
from ..results.metrics.classification import expected_calibration_error
from ..results.metrics.regression import (
    compute_mae_percent,
    compute_rpd,
    compute_rmsecv,
    compute_rmsep,
    compute_rmse,
    rmse_with_ci,
)
from .models.factory import make_models_from_config
from .preprocessors.spectro import (
    EMSC,
    EPO,
    MSC,
    SNV,
    AdaptiveQuantileSegmentNormalizer,
    Detrend,
    SavitzkyGolayTransformer,
)
from ..results.report import generate_scientific_report
from ..utils.compliance import check_astm_classification
from ..utils.email import enviar_email
from ..utils.errors import DataError, ModelError
from ..utils.logging import log_versions
from ..utils.misc import file_hash, get_git_commit
from ..utils.palettes import Palette
from ..utils.plots import (
    plot_confusion,
    plot_pls_loadings,
    plot_pls_weights,
    plot_pls_y_loadings,
    plot_reliability,
    plot_roc_ovr,
    plot_vip,
)
from ..utils.plots import plot_confusion, plot_pca_scores, plot_reliability, plot_roc_ovr
from ..utils.qc import detect_identical_spectra, find_duplicates, pca_outliers
from ..utils.summarize import summarize_runs
from ..utils.split import spxy_split
from ..utils.y_scramble import y_scramble_scores

warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet",
    category=FutureWarning,
)

logger = logging.getLogger(__name__)


_WINDOWS_WAIT_HANDLE_LIMIT = 63
_WINDOWS_MAX_LOKY_WORKERS = max(1, (_WINDOWS_WAIT_HANDLE_LIMIT - 1) // 2)


def explain_model(model_path: Path, data_path: Path, out_dir: Path) -> None:
    """Generate SHAP explanations for a saved model and dataset.

    Parameters
    ----------
    model_path:
        Path to a previously saved model in ``joblib`` format.
    data_path:
        Dataset in tabular or Excel format. Only the feature matrix ``X`` is
        required; target columns will be ignored if present.
    out_dir:
        Directory where the generated figures will be persisted.
    """

    try:
        import shap  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "shap is required for model explanation. Install with 'pip install nircls[shap]'"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and feature matrix
    est = load(model_path)
    try:
        if data_path.suffix.lower() in [".xlsx", ".xls"]:
            try:
                X, _, _ = read_xlsx_xy(data_path)
            except DataError:
                X, _, _ = read_tabular(data_path)
        else:
            X, _, _ = read_tabular(data_path)
    except DataError as e:
        raise DataError(f"Erro ao processar dados: {e}") from e

    # Build explainer according to model and compute SHAP values
    explainer = shap.Explainer(est, X)
    shap_values = explainer(X)

    import matplotlib.pyplot as plt

    # Summary plot (PNG + interactive HTML)
    shap.plots.summary(shap_values, show=False)
    shap.save_html(out_dir / "shap_summary.html", shap_values)
    plt.savefig(out_dir / "shap_summary.png", bbox_inches="tight")
    plt.close()

    values = shap_values.values
    if values.ndim == 3:  # multi-class -> use first class
        values = values[:, :, 0]
    mean_abs = np.abs(values).mean(axis=0)
    top_idx = int(mean_abs.argmax())
    feature_name = X.columns[top_idx]

    # Dependence plot for the most important feature
    shap.plots.scatter(shap_values[:, top_idx], color=shap_values, show=False)
    shap.save_html(
        out_dir / f"shap_dependence_{feature_name}.html", shap_values[:, top_idx]
    )
    plt.savefig(out_dir / f"shap_dependence_{feature_name}.png", bbox_inches="tight")
    plt.close()

    # Waterfall plot for first sample
    shap.plots.waterfall(shap_values[0], show=False)
    shap.save_html(out_dir / "shap_waterfall.html", shap_values[0])
    plt.savefig(out_dir / "shap_waterfall.png", bbox_inches="tight")
    plt.close()
