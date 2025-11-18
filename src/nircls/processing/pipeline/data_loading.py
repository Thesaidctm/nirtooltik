from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from joblib import load

from ..io import prepare_inputs as _prepare_inputs
from ..io import read_tabular, read_xlsx_xy
from ...utils.errors import DataError
from ...utils.logging import logger
from ...utils.misc import RESULTS_DIR


@lru_cache(maxsize=8)
def _prepare_inputs_cache_key(cfg_json: str) -> str:
    """Return an opaque cache key for prepared inputs.

    The key is intentionally cached separately from the main logic to avoid
    the overhead of loading the actual dataset in ``lru_cache``.
    """

    return hashlib.sha256(cfg_json.encode()).hexdigest()


def _compute_prepare_inputs_key(
    cfg: Dict[str, Any],
    files: Sequence[Path],
    *,
    spectral_range: Optional[Tuple[float, float]],
    drop_nans: bool,
    target_type: str,
    target_cols: Optional[Sequence[str] | str],
    group_col: Optional[str],
    use_cache: bool,
) -> Optional[str]:
    """Build a stable cache key for prepared inputs.

    The payload includes file metadata and configuration fields that influence
    how data is parsed and filtered, ensuring cache hits are only returned when
    the underlying data and relevant options are unchanged.
    """

    if not use_cache:
        return None

    try:
        files_payload = [
            {
                "path": str(p.resolve()),
                "mtime": p.stat().st_mtime,
                "size": p.stat().st_size,
            }
            for p in files
        ]
    except OSError:
        return None

    if isinstance(target_cols, str) or target_cols is None:
        normalized_targets: Optional[str | Sequence[str]] = target_cols
    else:
        normalized_targets = list(target_cols)

    payload = {
        "decimal": cfg.get("decimal"),
        "drop_nans": drop_nans,
        "encoding": cfg.get("encoding"),
        "files": files_payload,
        "group_col": group_col,
        "input_path": cfg.get("input_path"),
        "sep": cfg.get("sep"),
        "sheet_name": cfg.get("sheet_name"),
        "spectral_range": spectral_range,
        "target_cols": normalized_targets,
        "target_type": target_type,
        "use_cache": use_cache,
        "x_sheet": cfg.get("x_sheet"),
        "y_sheet": cfg.get("y_sheet"),
    }

    cfg_json = json.dumps(payload, sort_keys=True, default=str)
    return _prepare_inputs_cache_key(cfg_json)


_PREPARED_INPUTS_CACHE: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
_PREPARED_INPUTS_CACHE_MAX = 4


def _get_cached_prepared_inputs(cache_key: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Return a shallow copy of cached inputs and refresh LRU order."""

    if not cache_key:
        return None
    cached = _PREPARED_INPUTS_CACHE.get(cache_key)
    if cached is None:
        return None
    _PREPARED_INPUTS_CACHE.move_to_end(cache_key)
    return [dict(ds) for ds in cached]


def _set_cached_prepared_inputs(
    cache_key: Optional[str], data: List[Dict[str, Any]]
) -> None:
    """Store prepared inputs while bounding cache growth."""

    if not cache_key:
        return
    _PREPARED_INPUTS_CACHE[cache_key] = [dict(ds) for ds in data]
    _PREPARED_INPUTS_CACHE.move_to_end(cache_key)
    while len(_PREPARED_INPUTS_CACHE) > _PREPARED_INPUTS_CACHE_MAX:
        _PREPARED_INPUTS_CACHE.popitem(last=False)


def prepare_inputs(
    cfg: Dict[str, Any], *, use_cache: bool = True
) -> List[Dict[str, Any]]:
    """Load datasets, using cached features when available."""

    files = [Path(p) for p in glob(cfg["input_path"])]
    files.sort()
    sp = cfg.get("spectral_range", None)
    sp = (float(sp[0]), float(sp[1])) if sp else None
    out_dir = Path(cfg.get("out_dir", RESULTS_DIR))
    cache_dir = out_dir / "cache"
    drop_nans = bool(cfg.get("drop_nans", True))
    tgt = cfg.get("target_cols")
    if tgt is None and "target_col" in cfg:
        tgt = cfg["target_col"]
    grp = cfg.get("group_col")
    target_type = str(cfg.get("target_type", "classification")).lower()
    if target_type not in {"classification", "regression"}:
        raise DataError(f"Tipo de alvo inválido: {cfg.get('target_type')}")

    cache_key = _compute_prepare_inputs_key(
        cfg,
        files,
        spectral_range=sp,
        drop_nans=drop_nans,
        target_type=target_type,
        target_cols=tgt,
        group_col=grp,
        use_cache=use_cache,
    )
    cached_inputs = _get_cached_prepared_inputs(cache_key)
    if cached_inputs is not None:
        logger.debug(
            "Reusing prepared inputs from in-memory cache for %s", cfg.get("input_path")
        )
        return cached_inputs

    if not use_cache:
        result = _prepare_inputs(cfg, use_cache=False)
        return result

    cached_items: List[Tuple[Path, Dict[str, Any]]] = []
    for f in files:
        cpath = cache_dir / f"{f.stem}.joblib"
        fingerprint = {
            "mtime": f.stat().st_mtime,
            "spectral_range": sp,
            "drop_nans": drop_nans,
        }
        if not cpath.exists():
            cached_items = []
            break
        try:
            cached = load(cpath)
        except Exception:
            cached_items = []
            break
        if cached.get("fingerprint") != fingerprint:
            cached_items = []
            break
        cached_items.append((f, cached))

    if not cached_items:
        result = _prepare_inputs(cfg, use_cache=use_cache)
        _set_cached_prepared_inputs(cache_key, result)
        return result

    out: List[Dict[str, Any]] = []
    for f, cached in cached_items:
        keep_idx = cached["keep_idx"]
        X = cached["X"]
        n_nans = cached.get("n_nans", 0)
        if f.suffix.lower() in [".xlsx", ".xls"] and (
            "x_sheet" in cfg or "y_sheet" in cfg
        ):
            _, y_list, groups = read_xlsx_xy(
                f,
                cfg.get("x_sheet", "X"),
                cfg.get("y_sheet", "Y"),
                tgt,
                grp,
                target_type=target_type,
            )
        else:
            _, y_list, groups = read_tabular(
                f, tgt, grp, target_type=target_type
            )
        groups_reset = None
        if groups is not None:
            groups_reset = groups.loc[keep_idx].reset_index(drop=True)
        for name, y in y_list:
            y = y.loc[keep_idx].reset_index(drop=True)
            ds = {
                "file": f.as_posix(),
                "target": name,
                "X": X,
                "Y": y,
                "n_nans": n_nans,
                "target_type": target_type,
                "task_type": target_type,
            }
            if groups_reset is not None:
                ds["groups"] = groups_reset
            out.append(ds)
    _set_cached_prepared_inputs(cache_key, out)
    return out
