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
from ..utils.misc import RESULTS_DIR, file_hash, get_git_commit
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


def _validate_split_capacity(allowed: int, *, kind: str) -> None:
    """Raise a :class:`DataError` when ``allowed`` splits are insufficient."""

    if allowed >= 2:
        return
    if kind == "groups":
        raise DataError(
            "Número insuficiente de grupos distintos para validação cruzada. "
            "Forneça pelo menos dois grupos no conjunto de treino."
        )
    raise DataError(
        "Número insuficiente de amostras para validação cruzada. "
        "Forneça pelo menos duas amostras no conjunto de treino."
    )
