from __future__ import annotations

import ast
from importlib.metadata import entry_points
from typing import Any, Dict, List

from .base import BaseModelBuilder
from ...utils.errors import ConfigError


def _apply_random_state(estimator: Any, random_state: int | None) -> Any:
    if random_state is None:
        return estimator
    if not hasattr(estimator, "get_params"):
        return estimator
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        return estimator
    if "random_state" in params:
        try:
            estimator.set_params(random_state=random_state)
        except ValueError:
            pass
    return estimator


def _apply_n_jobs(estimator: Any, n_jobs: int | None) -> Any:
    """Set ``n_jobs``-style parameters on ``estimator`` when supported."""

    if n_jobs is None:
        return estimator
    if not hasattr(estimator, "get_params"):
        return estimator
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        return estimator
    updated: Dict[str, Any] = {}
    if "n_jobs" in params:
        updated["n_jobs"] = n_jobs
    if "nthread" in params:
        updated["nthread"] = n_jobs
    if "thread_count" in params:
        updated["thread_count"] = n_jobs
    if not updated:
        return estimator
    try:
        estimator.set_params(**updated)
    except ValueError:
        return estimator
    return estimator

from .classification import (
    CLASSIFICATION_DEFAULT_GRIDS,
    CLASSIFICATION_MODEL_BUILDERS,
    CLASSIFICATION_MODEL_DESCRIPTIONS,
)
from .regression import (
    REGRESSION_DEFAULT_GRIDS,
    REGRESSION_MODEL_BUILDERS,
    REGRESSION_MODEL_DESCRIPTIONS,
)


MODEL_DESCRIPTIONS: Dict[str, str] = CLASSIFICATION_MODEL_DESCRIPTIONS

_ENTRYPOINT_GROUP = "nircls.model_builders"

MODEL_BUILDERS = CLASSIFICATION_MODEL_BUILDERS
DEFAULT_GRIDS = CLASSIFICATION_DEFAULT_GRIDS

TASK_MODEL_BUILDERS: Dict[str, Dict[str, Any]] = {
    "classification": CLASSIFICATION_MODEL_BUILDERS,
    "regression": REGRESSION_MODEL_BUILDERS,
}

TASK_DEFAULT_GRIDS: Dict[str, Dict[str, Dict[str, List[Any]]]] = {
    "classification": CLASSIFICATION_DEFAULT_GRIDS,
    "regression": REGRESSION_DEFAULT_GRIDS,
}

TASK_MODEL_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "classification": CLASSIFICATION_MODEL_DESCRIPTIONS,
    "regression": REGRESSION_MODEL_DESCRIPTIONS,
}

TASK_SUFFIXES = {"classification": "_CLS", "regression": "_REG"}

_PLS_REGRESSION_KEYS = {"PLSREGRESSION", "PLSR", "PLS_REG"}


_EP_LOADED = False


def _load_entry_point_builders() -> None:  # pragma: no cover - simple metadata loader
    """Populate builder mappings from third-party entry points."""

    global _EP_LOADED
    if _EP_LOADED:
        return
    eps = entry_points()
    if hasattr(eps, "select"):
        group_eps = eps.select(group=_ENTRYPOINT_GROUP)
    else:  # Python<3.10
        group_eps = eps.get(_ENTRYPOINT_GROUP, [])
    for ep in group_eps:
        builder_cls = ep.load()
        try:
            builder: BaseModelBuilder = builder_cls()  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive
            continue
        name = builder.name.upper()
        MODEL_BUILDERS[name] = lambda grid, b=builder: b.build(grid)
        grid = builder.default_grid() or {}
        if grid:
            DEFAULT_GRIDS[name] = grid
        if getattr(builder, "description", ""):
            MODEL_DESCRIPTIONS[name] = builder.description
    _EP_LOADED = True


def _normalize_task_type(task_type: str | None) -> str:
    normalized = str(task_type or "classification").strip().lower()
    if normalized == "regression":
        return "regression"
    return "classification"


def get_model_builders(task_type: str = "classification") -> Dict[str, Any]:
    """Return mapping of available model builders for ``task_type``."""

    normalized = _normalize_task_type(task_type)
    if normalized == "classification":
        _load_entry_point_builders()
    return TASK_MODEL_BUILDERS[normalized]


def get_model_descriptions(task_type: str = "classification") -> Dict[str, str]:
    """Return mapping from model identifier to human readable description."""

    normalized = _normalize_task_type(task_type)
    if normalized == "classification":
        _load_entry_point_builders()
    return TASK_MODEL_DESCRIPTIONS[normalized]


def _normalize_grid_values(grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Coerce YAML-loaded grid values to proper Python types.

    Nested lists representing tuples (e.g. ``[100]``) are converted to
    :class:`tuple`. String values are interpreted where possible using
    :func:`ast.literal_eval` so that numeric parameters provided as quoted
    values (e.g. ``"1e-09"``) are coerced to proper Python types. The grid is
    modified in-place and returned for convenience.
    """

    for param, values in list(grid.items()):
        normalized: List[Any] = []
        i = 0
        while i < len(values):
            val = values[i]
            if isinstance(val, list):
                normalized.append(tuple(val))
                i += 1
                continue
            if isinstance(val, str):
                if val.startswith("("):
                    # Handle tuple representations possibly split across
                    # multiple YAML tokens. Concatenate subsequent strings
                    # until a closing parenthesis is found before attempting
                    # to evaluate.
                    parts = [val]
                    if not val.endswith(")"):
                        i += 1
                        while i < len(values):
                            part = values[i]
                            parts.append(part)
                            if isinstance(part, str) and part.endswith(")"):
                                break
                            i += 1
                    joined = " ".join(str(p) for p in parts)
                    try:
                        normalized.append(ast.literal_eval(joined))
                    except Exception:
                        normalized.extend(parts)
                    i += 1
                    continue
                # Attempt to parse numeric strings (e.g. "1e-09")
                try:
                    normalized.append(ast.literal_eval(val))
                except Exception:
                    normalized.append(val)
                i += 1
                continue
            normalized.append(val)
            i += 1
        grid[param] = normalized
    return grid


def make_models_from_config(
    models_cfg: List[Dict[str, Any]],
    *,
    task_type: str = "classification",
    random_state: int | None = None,
    n_jobs: int | None = None,
):
    normalized_type = _normalize_task_type(task_type)
    builders = get_model_builders(normalized_type)
    defaults = TASK_DEFAULT_GRIDS[normalized_type]
    suffix = TASK_SUFFIXES[normalized_type]
    other_suffix = (
        TASK_SUFFIXES["regression"]
        if normalized_type == "classification"
        else TASK_SUFFIXES["classification"]
    )
    out = []
    for m in models_cfg:
        m.setdefault("overfit_check", True)
        name = m["name"]
        upper = name.upper()

        resolved_key = upper
        builder = builders.get(resolved_key)

        params = m.get("params") or {}

        def _attempt(candidate_key: str) -> tuple[str, Any] | tuple[None, None]:
            candidate_builder = builders.get(candidate_key)
            if candidate_builder is None:
                return None, None
            return candidate_key, candidate_builder

        if builder is None and not resolved_key.endswith(suffix):
            candidate_key, candidate_builder = _attempt(f"{resolved_key}{suffix}")
            if candidate_builder is not None:
                resolved_key, builder = candidate_key, candidate_builder
        if builder is None and other_suffix:
            if resolved_key.endswith(other_suffix):
                base = resolved_key[: -len(other_suffix)]
                candidate_key, candidate_builder = _attempt(base)
                if candidate_builder is None and suffix:
                    candidate_key, candidate_builder = _attempt(f"{base}{suffix}")
                if candidate_builder is not None:
                    resolved_key, builder = candidate_key, candidate_builder
            else:
                candidate_key, candidate_builder = _attempt(f"{resolved_key}{other_suffix}")
                if candidate_builder is not None:
                    resolved_key, builder = candidate_key, candidate_builder
        if builder is None:
            raise ValueError(f"Modelo não suportado: {name}")

        grid = _normalize_grid_values(m.get("grid", {}) or {})
        default_grid = defaults.get(resolved_key, {})
        for k, v in default_grid.items():
            grid.setdefault(k, v)

        if resolved_key in _PLS_REGRESSION_KEYS:
            if "n_components" not in params:
                raise ConfigError(
                    "For PLS_REG, 'n_components' must be set by the user in model params."
                )
            grid["n_components"] = [params["n_components"]]

        est = builder(grid)
        est = _apply_random_state(est, random_state)
        est = _apply_n_jobs(est, n_jobs)

        if params:
            # Coerce any quoted numeric values to their Python equivalents
            for k, v in list(params.items()):
                if isinstance(v, str):
                    try:
                        params[k] = ast.literal_eval(v)
                    except Exception:
                        pass
            est.set_params(**params)

        out.append({"name": name, "estimator": est, "grid": grid})

    return out
