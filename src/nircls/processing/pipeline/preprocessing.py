from __future__ import annotations

import logging
import inspect
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..models.factory import make_models_from_config
from ..preprocessors.spectro import (
    EMSC,
    EPO,
    MSC,
    SNV,
    AdaptiveQuantileSegmentNormalizer,
    Detrend,
    SavitzkyGolayTransformer,
)
from ...utils.errors import DataError, ModelError
from ...config_schema import ConfigModel, ValidationError

logger = logging.getLogger(__name__)


def _infer_task_type(cfg: Mapping[str, Any]) -> str:
    """Infer the task type (classification or regression) from ``cfg``."""

    task_type = cfg.get("task_type")
    if not task_type:
        data_cfg = cfg.get("data") if isinstance(cfg, Mapping) else None
        if isinstance(data_cfg, Mapping):
            task_type = data_cfg.get("task_type")
        if not task_type and isinstance(data_cfg, Mapping):
            task_type = data_cfg.get("target_type")
    normalized = str(task_type or "classification").lower()
    if normalized not in {"classification", "regression"}:
        raise DataError(f"Tipo de tarefa inválido: {task_type}")
    return normalized


def _validate_cfg(
    cfg: Dict[str, Any],
    *,
    need_data: bool = False,
    need_cv: bool = False,
    need_models: bool = True,
) -> ConfigModel:
    """Validate configuration dictionary using :class:`ConfigModel`."""

    try:
        model = ConfigModel(**cfg)
    except ValidationError as exc:
        raise DataError(
            f"Configuração inválida: {exc}. Revise os campos informados."
        ) from exc
    if need_data and model.data is None:
        raise DataError("Seção 'data' é obrigatória. Informe o caminho dos dados.")
    if need_models and (not model.models):
        raise ModelError("Seção 'models' é obrigatória. Adicione pelo menos um modelo.")
    if need_cv and (model.cv is None or not model.cv.strategies):
        raise ModelError(
            "Seção 'cv' é obrigatória. Defina ao menos uma estratégia de validação."
        )
    return model


def _make_preprocess(
    steps_cfg: Optional[Sequence[str]], cfg: Dict[str, Any]
) -> Tuple[List[Tuple[str, Any]], Dict[str, Any]]:
    """Create preprocessing steps and optional parameter grid."""

    steps = []
    grid = {}
    for name in steps_cfg or []:
        if name is None:
            continue
        if isinstance(name, str):
            cleaned = name.strip()
            if not cleaned:
                continue
            if cleaned.lower() in {"none", "raw"}:
                continue
            n = cleaned.upper()
        else:
            n = str(name).upper()
        if n == "SNV":
            steps.append(("snv", SNV()))
        elif n == "MSC":
            steps.append(("msc", MSC()))
        elif n == "DETREND":
            steps.append(("detrend", Detrend()))
        elif n == "SG":
            sg = cfg.get("sg", {"window": 15, "polyorder": 2, "deriv": 1})
            wl = sg.get("window", 15)
            po = sg.get("polyorder", 2)
            dv = sg.get("deriv", 1)
            if isinstance(wl, (list, tuple)):
                grid["sg__window_length"] = [int(w) for w in wl]
                wl = wl[0]
            if isinstance(po, (list, tuple)):
                grid["sg__polyorder"] = [int(p) for p in po]
                po = po[0]
            if isinstance(dv, (list, tuple)):
                grid["sg__deriv"] = [int(d) for d in dv]
                dv = dv[0]
            steps.append(("sg", SavitzkyGolayTransformer(wl, po, dv)))
        elif n == "EMSC":
            steps.append(("emsc", EMSC()))
        elif n == "EPO":
            steps.append(("epo", EPO()))
        elif n == "AQS":
            aqs_cfg = cfg.get("aqs", {})
            min_seg = aqs_cfg.get("min_segment_size", 25)
            blend_strength = aqs_cfg.get("blend_strength", 4.0)
            if isinstance(min_seg, (list, tuple)):
                grid["aqs__min_segment_size"] = [int(v) for v in min_seg]
                min_seg = min_seg[0]
            if isinstance(blend_strength, (list, tuple)):
                grid["aqs__blend_strength"] = [float(v) for v in blend_strength]
                blend_strength = blend_strength[0]
            steps.append(
                (
                    "aqs",
                    AdaptiveQuantileSegmentNormalizer(
                        min_segment_size=int(min_seg),
                        blend_strength=float(blend_strength),
                    ),
                )
            )
        elif n == "SCALE":
            steps.append(("scaler", StandardScaler()))
        else:
            raise ModelError(
                f"Pré-processo desconhecido: {name}. Verifique a lista de etapas suportadas."
            )
    return steps, grid


def _get_model_step_name(pipe: Pipeline) -> str:
    """Return the name of the final estimator step in ``pipe``."""

    if hasattr(pipe, "named_steps") and pipe.named_steps:
        for candidate in ("clf", "reg"):
            if candidate in pipe.named_steps:
                return candidate
        try:
            return next(reversed(pipe.named_steps))
        except StopIteration:
            pass
    if getattr(pipe, "steps", None):
        return pipe.steps[-1][0]
    return "clf"


def _get_preprocess_steps_from_pipeline(pipe: Pipeline) -> List[Any]:
    """Return preprocessing steps stored in ``pipe`` excluding the estimator."""

    if not hasattr(pipe, "named_steps"):
        return []
    final_name = _get_model_step_name(pipe)
    return [v for k, v in pipe.named_steps.items() if k != final_name]


def build_pipeline_from_config(
    cfg: Dict[str, Any],
    *,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> List[Tuple[str, Pipeline, Dict[str, Any]]]:
    _validate_cfg(cfg, need_models=True)
    task_type = _infer_task_type(cfg)
    final_step_name = "clf" if task_type == "classification" else "reg"
    pre_cfg = cfg.get("preprocess", {})
    steps_cfg = pre_cfg.get("steps", [])
    evaluate_flag = pre_cfg.get("evaluate_individual")
    if evaluate_flag is None:
        evaluate_ind = len(steps_cfg) > 1
    else:
        evaluate_ind = bool(evaluate_flag)
    out: List[Tuple[str, Pipeline, Dict[str, Any]]] = []

    def _build_models(
        pre_steps: List[Tuple[str, Any]], pre_grid: Dict[str, Any], suffix: str = ""
    ) -> None:
        params: Dict[str, Any] = {}
        try:
            signature = inspect.signature(make_models_from_config)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            models = make_models_from_config(cfg["models"])
        else:
            if "task_type" in signature.parameters:
                params["task_type"] = task_type
            if "random_state" in signature.parameters:
                params["random_state"] = random_state
            if "n_jobs" in signature.parameters and n_jobs is not None:
                params["n_jobs"] = n_jobs
            models = make_models_from_config(cfg["models"], **params)
        for m in models:
            est = m["estimator"]
            grid = pre_grid.copy()
            model_grid = m.get("grid", {}) or {}
            if task_type == "classification" and not hasattr(est, "predict_proba"):
                calib_cfg = m.get("calibration") or {}
                method = calib_cfg.get("method", "sigmoid")
                cv = calib_cfg.get("cv", 5)
                calib_params = {"estimator": est, "method": method, "cv": cv}
                if n_jobs is not None:
                    calib_params["n_jobs"] = n_jobs
                est = CalibratedClassifierCV(**calib_params)
                grid.update(
                    {
                        f"{final_step_name}__estimator__{k}": v
                        for k, v in model_grid.items()
                    }
                )
            else:
                grid.update({f"{final_step_name}__{k}": v for k, v in model_grid.items()})
            name = m["name"] + suffix
            pipe = Pipeline(pre_steps + [(final_step_name, est)])
            out.append((name, pipe, grid))

    if evaluate_ind and len(steps_cfg) > 1:
        for step in steps_cfg:
            pre_steps, pre_grid = _make_preprocess([step], pre_cfg)
            _build_models(pre_steps, pre_grid, suffix=f"__{step.lower()}")
    else:
        pre_steps, pre_grid = _make_preprocess(steps_cfg, pre_cfg)
        _build_models(pre_steps, pre_grid)

    return out


def _call_build_pipeline(
    cfg: Dict[str, Any],
    *,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> List[Tuple[str, Pipeline, Dict[str, Any]]]:
    """Invoke ``build_pipeline_from_config`` respecting its signature."""

    try:
        signature = inspect.signature(build_pipeline_from_config)
    except (TypeError, ValueError):
        return build_pipeline_from_config(cfg)
    kwargs: Dict[str, Any] = {}
    if "random_state" in signature.parameters:
        kwargs["random_state"] = random_state
    if "n_jobs" in signature.parameters and n_jobs is not None:
        kwargs["n_jobs"] = n_jobs
    return build_pipeline_from_config(cfg, **kwargs)


def _format_preprocess_step(step: Any) -> str:
    """Return a concise string representation for a preprocessing ``step``."""

    if step is None:
        return "None"
    if isinstance(step, str):
        cleaned = step.strip()
        if not cleaned:
            return ""
        if cleaned.endswith("()"):
            cleaned = cleaned[:-2]
        upper = cleaned.upper()
        mapping = {
            "SNV": "SNV",
            "MSC": "MSC",
            "DETREND": "Detrend",
            "SG": "Savitzky-Golay",
            "EMSC": "EMSC",
            "EPO": "EPO",
            "AQS": "Adaptive QSN",
            "SCALE": "Scaler",
        }
        return mapping.get(upper, cleaned)
    return step.__class__.__name__


def _format_preprocess_steps(steps: Sequence[Any]) -> str:
    """Return string representation for a sequence of preprocessing ``steps``."""

    formatted = [_format_preprocess_step(s) for s in steps if s is not None]
    return " + ".join(s for s in formatted if s)


def _normalize_preprocess_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _coerce_param_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            logger.debug("Could not parse parameter list: %s", value)
    return value


__all__ = [
    "_call_build_pipeline",
    "_coerce_param_value",
    "_format_preprocess_step",
    "_format_preprocess_steps",
    "_get_model_step_name",
    "_get_preprocess_steps_from_pipeline",
    "_infer_task_type",
    "_make_preprocess",
    "_normalize_preprocess_value",
    "_validate_cfg",
    "build_pipeline_from_config",
]
