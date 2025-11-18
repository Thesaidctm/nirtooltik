from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast, Literal

import numpy as np
import pandas as pd
from joblib import dump, load

from ..utils.errors import DataError
from ..utils.misc import RESULTS_DIR

_MISSING_OPENPYXL_MSG = (
    "Dependência opcional 'openpyxl' ausente. Instale com `pip install "
    "\"nircls[excel]\"` ou `pip install \"nircls[full]\"`."
)

_CACHE_VERSION = 1


def _iter_input_files(pattern: str | Iterable[str]) -> List[Path]:
    """Expand ``pattern`` into a sorted list of files.

    Parameters
    ----------
    pattern:
        Either a glob pattern or an iterable of glob patterns. ``Path.glob``
        is not used because ``input_path`` may point to files outside the
        current working tree.
    """

    if isinstance(pattern, (str, Path)):
        patterns: Iterable[str | Path] = [pattern]
    else:
        patterns = pattern

    files: List[Path] = []
    for item in patterns:
        for match in glob(str(item)):
            files.append(Path(match))
    files = [p for p in files if not p.name.startswith("~$")]
    files.sort()
    return files


def _dataset_fingerprint(
    path: Path,
    spectral_range: Optional[Tuple[float, float]],
    drop_nans: bool,
) -> Dict[str, Any]:
    """Return a dictionary describing the cache-relevant dataset state."""

    return {
        "version": _CACHE_VERSION,
        "path": str(path.resolve()),
        "mtime": path.stat().st_mtime,
        "spectral_range": spectral_range,
        "drop_nans": drop_nans,
    }


def _load_cached_features(
    cache_path: Path, fingerprint: Dict[str, Any]
) -> Optional[Tuple[pd.DataFrame, pd.Index, int]]:
    """Return cached feature matrix, kept indices and NaN counts if valid."""

    try:
        payload = load(cache_path)
    except FileNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive path
        cache_path.unlink(missing_ok=True)
        return None

    if payload.get("fingerprint") != fingerprint:
        return None

    X = payload.get("X")
    keep_idx = payload.get("keep_idx")
    n_nans = payload.get("n_nans")
    if not isinstance(X, pd.DataFrame) or keep_idx is None:
        return None
    index = pd.Index(keep_idx)
    return X, index, int(n_nans)


def _store_cached_features(
    cache_path: Path,
    fingerprint: Dict[str, Any],
    X: pd.DataFrame,
    keep_idx: pd.Index,
    n_nans: int,
) -> None:
    """Persist processed feature matrix for reuse across runs."""

    try:  # pragma: no cover - best effort cache write
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "fingerprint": fingerprint,
                "X": X,
                "keep_idx": keep_idx.tolist(),
                "n_nans": int(n_nans),
            },
            cache_path,
        )
    except Exception:
        cache_path.unlink(missing_ok=True)


def _normalize_columns_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with numeric spectral columns converted to float.

    Columns that cannot be converted to ``float`` are dropped. The original
    column order is preserved for the remaining columns.
    """
    keep_cols = []
    rename_map = {}
    for c in df.columns:
        try:
            rename_map[c] = float(c)
            keep_cols.append(c)
        except (ValueError, TypeError):
            continue
    if not keep_cols:
        return df.iloc[:, 0:0]
    return df[keep_cols].rename(columns=rename_map)


def _filter_range(
    df: pd.DataFrame, spectral_range: Optional[Tuple[float, float]]
) -> pd.DataFrame:
    """Normalize spectral columns and optionally slice by range."""
    df = _normalize_columns_to_float(df)
    if not spectral_range:
        return df
    lo, hi = spectral_range
    keep = [c for c in df.columns if lo <= c <= hi]
    return df.loc[:, keep]


TargetType = Literal["classification", "regression"]


def _extract_targets_and_groups(
    df: pd.DataFrame,
    target_cols: Optional[Sequence[str] | str],
    group_col: Optional[str],
    target_type: TargetType = "classification",
) -> Tuple[List[str], List[Tuple[str, pd.Series]], Optional[pd.Series]]:
    source = df.attrs.get("_source_path")
    source_str = f" em {source}" if source else ""
    default_to_all = bool(df.attrs.get("_default_all_target_cols"))
    tt_normalized = str(target_type).lower()
    if tt_normalized not in {"classification", "regression"}:
        raise DataError(f"Tipo de alvo inválido: {target_type}")
    is_classification = tt_normalized == "classification"
    if target_cols in [None, "all"]:
        if default_to_all:
            target_cols_list = [c for c in df.columns if c != group_col]
        else:
            if is_classification:
                target_cols_list = [
                    c
                    for c in df.columns
                    if c != group_col and not pd.api.types.is_numeric_dtype(df[c])
                ]
            else:
                numeric_candidates = [
                    c
                    for c in df.columns
                    if c != group_col and pd.api.types.is_numeric_dtype(df[c])
                ]
                if len(numeric_candidates) == 1:
                    target_cols_list = numeric_candidates
                else:
                    target_cols_list = []
                    if len(numeric_candidates) > 1:
                        raise DataError(
                            "Não foi possível inferir colunas alvo numéricas. Informe 'target_cols'."
                        )
    elif isinstance(target_cols, str):
        target_cols_list = [target_cols]
    else:
        target_cols_list = list(cast(Sequence[str], target_cols))
    if not target_cols_list:
        if is_classification:
            target_cols_list = []
            y_list: List[Tuple[str, pd.Series]] = []
            groups = None
            if group_col:
                if group_col not in df.columns:
                    raise DataError(
                        f"Coluna de grupos {group_col} não encontrada{source_str}"
                    )
                groups = df[group_col].astype("category")
            return target_cols_list, y_list, groups
        raise DataError(
            "Nenhuma coluna alvo numérica encontrada. Informe 'target_cols' ou ajuste 'target_type'."
        )
    y_list: List[Tuple[str, pd.Series]] = []
    for col in target_cols_list:
        if col not in df.columns:
            raise DataError(f"Coluna alvo {col} não encontrada{source_str}")
        series = df[col]
        if is_classification:
            y_list.append((col, series.astype("category")))
        else:
            try:
                numeric = pd.to_numeric(series, errors="raise")
            except (ValueError, TypeError) as exc:
                raise DataError(
                    f"Coluna alvo {col} deve conter valores numéricos para regressão"
                ) from exc
            if not pd.api.types.is_float_dtype(numeric.dtype):
                numeric = numeric.astype(float)
            y_list.append((col, numeric))
    groups = None
    if group_col:
        if group_col not in df.columns:
            raise DataError(
                f"Coluna de grupos {group_col} não encontrada{source_str}"
            )
        groups = df[group_col].astype("category")
    expected_len = len(df)
    if any(len(y) != expected_len for _, y in y_list) or (
        groups is not None and len(groups) != expected_len
    ):
        raise DataError("X e Y com n_amostras diferente")
    return target_cols_list, y_list, groups


def read_xlsx_xy(
    path: Path,
    x_sheet: str = "X",
    y_sheet: str = "Y",
    target_cols: Optional[Sequence[str] | str] = None,
    group_col: Optional[str] = None,
    *,
    target_type: TargetType = "classification",
) -> Tuple[pd.DataFrame, List[Tuple[str, pd.Series]], Optional[pd.Series]]:
    try:
        X = pd.read_excel(path, sheet_name=x_sheet)
        Ydf = pd.read_excel(path, sheet_name=y_sheet)
    except (FileNotFoundError, PermissionError) as e:
        raise DataError(str(e)) from e
    except ValueError as e:
        raise DataError(str(e)) from e
    except ImportError as e:
        if "openpyxl" in str(e).lower():
            raise DataError(_MISSING_OPENPYXL_MSG) from e
        raise
    Ydf.attrs["_source_path"] = str(path)
    Ydf.attrs["_default_all_target_cols"] = True
    _, out, groups = _extract_targets_and_groups(
        Ydf, target_cols, group_col, target_type=target_type
    )
    X = X.select_dtypes(include=[np.number])
    if len(X) != len(Ydf):
        raise DataError("X e Y com n_amostras diferente")
    return X, out, groups


def read_tabular(
    path: Path,
    target_cols: Optional[Sequence[str] | str] = None,
    group_col: Optional[str] = None,
    target_type: TargetType = "classification",
    *,
    encoding: Optional[str] = None,
    sep: str = ",",
    decimal: str = ".",
    sheet_name: str | int | None = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, pd.Series]], Optional[pd.Series]]:
    if not path.is_file() or path.suffix.lower() not in {".csv", ".xls", ".xlsx"}:
        raise DataError("Caminho é um diretório ou possui formato não suportado")
    try:
        if path.suffix.lower() == ".csv":
            read_kwargs = {"sep": sep, "decimal": decimal}
            if encoding:
                read_kwargs["encoding"] = encoding
            df = pd.read_csv(path, **read_kwargs)
        else:
            read_kwargs = {"decimal": decimal}
            if sheet_name is not None:
                read_kwargs["sheet_name"] = sheet_name
            df = pd.read_excel(path, **read_kwargs)
    except (FileNotFoundError, PermissionError) as e:
        raise DataError(str(e)) from e
    except ImportError as e:
        if "openpyxl" in str(e).lower():
            raise DataError(_MISSING_OPENPYXL_MSG) from e
        raise
    df.attrs["_source_path"] = str(path)
    target_cols_list, y_list, groups = _extract_targets_and_groups(
        df, target_cols, group_col, target_type=target_type
    )
    drop_cols = list(target_cols_list)
    if group_col:
        drop_cols.append(group_col)
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    return X, y_list, groups


def export_xy_xlsx(
    path: Path, X: pd.DataFrame, y: pd.Series, x_sheet: str = "X", y_sheet: str = "Y"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            X.to_excel(w, index=False, sheet_name=x_sheet)
            y.to_frame(name=y.name or "target").to_excel(
                w, index=False, sheet_name=y_sheet
            )
    except ImportError as e:  # pragma: no cover - requires missing optional dependency
        if "openpyxl" in str(e).lower():
            raise DataError(_MISSING_OPENPYXL_MSG) from e
        raise


def prepare_inputs(
    data_cfg: Dict[str, Any], *, use_cache: bool = True
) -> List[Dict[str, Any]]:
    """Load and clean spectral datasets specified in configuration.

    Column names are normalised to floats (nanometers) and filtered by the
    provided spectral range. Infinite values are replaced by ``NaN`` and rows
    containing missing values are dropped in a leakage-free manner.
    """
    files = _iter_input_files(data_cfg["input_path"])
    if not files:
        raise DataError("Nenhum arquivo encontrado para o padrão informado")
    invalid = [p for p in files if not p.is_file()]
    files = [p for p in files if p.is_file()]
    if invalid:
        invalid_str = ", ".join(str(p) for p in invalid)
        raise DataError(f"Caminho(s) inválido(s): {invalid_str}")
    sp = data_cfg.get("spectral_range", None)
    sp = (float(sp[0]), float(sp[1])) if sp else None
    out_dir = Path(data_cfg.get("out_dir", RESULTS_DIR))
    cache_dir = out_dir / "cache"
    drop_nans = bool(data_cfg.get("drop_nans", True))
    tgt = data_cfg.get("target_cols")
    if tgt is None and "target_col" in data_cfg:
        tgt = data_cfg["target_col"]
    grp = data_cfg.get("group_col")
    target_type = str(data_cfg.get("target_type", "classification")).lower()
    if target_type not in {"classification", "regression"}:
        raise DataError(f"Tipo de alvo inválido: {data_cfg.get('target_type')}")
    out = []
    for f in files:
        if f.suffix.lower() in [".xlsx", ".xls"] and (
            "x_sheet" in data_cfg or "y_sheet" in data_cfg
        ):
            X, y_list, groups = read_xlsx_xy(
                f,
                data_cfg.get("x_sheet", "X"),
                data_cfg.get("y_sheet", "Y"),
                tgt,
                grp,
                target_type=target_type,
            )
        else:
            X, y_list, groups = read_tabular(
                f,
                tgt,
                grp,
                target_type=target_type,
                encoding=data_cfg.get("encoding"),
                sep=data_cfg.get("sep", ","),
                decimal=data_cfg.get("decimal", "."),
                sheet_name=data_cfg.get("sheet_name"),
            )
        fingerprint: Optional[Dict[str, Any]] = None
        cache_hit: Optional[Tuple[pd.DataFrame, pd.Index, int]] = None
        cache_path = cache_dir / f"{f.stem}.joblib"
        if use_cache:
            fingerprint = _dataset_fingerprint(f, sp, drop_nans)
            cache_hit = _load_cached_features(cache_path, fingerprint)

        if cache_hit is None:
            X = _filter_range(X, sp)
            if X.empty:
                raise DataError("Intervalo espectral fora da faixa disponível")
            X = X.replace([np.inf, -np.inf], np.nan)
            n_nans = int(X.isna().any(axis=1).sum())
            if drop_nans:
                X = X.dropna(axis=0)
            keep_idx = X.index
            if groups is not None:
                groups = groups.loc[keep_idx]
            X_reset = X.reset_index(drop=True)
            n_nans_out = n_nans
            if use_cache and fingerprint is not None:
                _store_cached_features(cache_path, fingerprint, X_reset, keep_idx, n_nans)
        else:
            X_reset, keep_idx, n_nans_out = cache_hit
            if groups is not None:
                groups = groups.loc[keep_idx]

        groups_reset = groups.reset_index(drop=True) if groups is not None else None

        for name, y in y_list:
            y = y.loc[keep_idx]
            ds = {
                "file": f.as_posix(),
                "target": name,
                "X": X_reset,
                "Y": y.reset_index(drop=True),
                "n_nans": n_nans_out,
                "target_type": target_type,
                "task_type": target_type,
            }
            if groups_reset is not None:
                ds["groups"] = groups_reset
            out.append(ds)
    return out


def summarize_dataset(ds_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Summarize dataset information.

    Parameters
    ----------
    ds_list:
        Output of :func:`prepare_inputs`.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``target``, ``class``, ``count``, ``spectral_min``,
        ``spectral_max``, ``n_nans``, ``preprocess_steps`` and ``result``
        summarising each dataset.
    """

    rows: List[Dict[str, Any]] = []
    for ds in ds_list:
        X: pd.DataFrame = ds["X"]
        y: pd.Series = ds["Y"]
        spectral_min = float(min(X.columns)) if not X.empty else float("nan")
        spectral_max = float(max(X.columns)) if not X.empty else float("nan")
        n_nans = int(ds.get("n_nans", 0))
        preprocess_steps = ds.get("preprocess_steps")
        result = ds.get("result")

        steps_value = preprocess_steps
        if steps_value is None or (
            isinstance(steps_value, float) and np.isnan(steps_value)
        ):
            steps_empty = True
        elif isinstance(steps_value, str):
            steps_empty = steps_value.strip() == ""
        else:
            try:
                steps_empty = len(steps_value) == 0  # type: ignore[arg-type]
            except TypeError:
                steps_empty = False

        if steps_empty:
            result = "RAW"
        if pd.api.types.is_categorical_dtype(y):
            counts = y.value_counts()
            for cls, cnt in counts.items():
                rows.append(
                    {
                        "target": ds["target"],
                        "class": cls,
                        "count": int(cnt),
                        "spectral_min": spectral_min,
                        "spectral_max": spectral_max,
                        "n_nans": n_nans,
                        "preprocess_steps": preprocess_steps,
                        "result": result,
                    }
                )
        else:
            rows.append(
                {
                    "target": ds["target"],
                    "class": float("nan"),
                    "count": int(len(y)),
                    "spectral_min": spectral_min,
                    "spectral_max": spectral_max,
                    "n_nans": n_nans,
                    "preprocess_steps": preprocess_steps,
                    "result": result,
                }
            )
    return pd.DataFrame(rows)
