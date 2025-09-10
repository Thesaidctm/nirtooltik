from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from joblib import dump

from .utils.errors import DataError
from .utils.misc import RESULTS_DIR


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


def read_xlsx_xy(
    path: Path,
    x_sheet: str = "X",
    y_sheet: str = "Y",
    target_cols: Optional[Sequence[str] | str] = None,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, pd.Series]], Optional[pd.Series]]:
    try:
        X = pd.read_excel(path, sheet_name=x_sheet)
        Ydf = pd.read_excel(path, sheet_name=y_sheet)
    except (FileNotFoundError, PermissionError) as e:
        raise DataError(str(e)) from e
    except ValueError as e:
        raise DataError(str(e)) from e
    if target_cols in [None, "all"]:
        target_cols_list = [c for c in Ydf.columns if c != group_col]
    elif isinstance(target_cols, str):
        target_cols_list = [target_cols]
    else:
        target_cols_list = list(cast(Sequence[str], target_cols))
    out: List[Tuple[str, pd.Series]] = []
    for col in target_cols_list:
        if col not in Ydf.columns:
            raise DataError(f"Coluna alvo {col} não encontrada em {path}")
        out.append((col, Ydf[col].astype("category")))
    groups = None
    if group_col:
        if group_col not in Ydf.columns:
            raise DataError(f"Coluna de grupos {group_col} não encontrada em {path}")
        groups = Ydf[group_col].astype("category")
    X = X.select_dtypes(include=[np.number])
    if any(len(X) != len(y) for _, y in out) or (
        groups is not None and len(X) != len(groups)
    ):
        raise DataError("X e Y com n_amostras diferente")
    return X, out, groups


def read_tabular(
    path: Path,
    target_cols: Optional[Sequence[str] | str] = None,
    group_col: Optional[str] = None,
    *,
    encoding: Optional[str] = None,
    sep: str = ",",
    decimal: str = ".",
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
            df = pd.read_excel(path, decimal=decimal)
    except (FileNotFoundError, PermissionError) as e:
        raise DataError(str(e)) from e
    if target_cols in [None, "all"]:
        target_cols_list = [
            c
            for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and c != group_col
        ]
    elif isinstance(target_cols, str):
        target_cols_list = [target_cols]
    else:
        target_cols_list = list(cast(Sequence[str], target_cols))
    y_list: List[Tuple[str, pd.Series]] = []
    for col in target_cols_list:
        if col not in df.columns:
            raise DataError(f"Coluna alvo {col} não encontrada em {path}")
        y_list.append((col, df[col].astype("category")))
    groups = None
    if group_col:
        if group_col not in df.columns:
            raise DataError(f"Coluna de grupos {group_col} não encontrada em {path}")
        groups = df[group_col].astype("category")
    drop_cols = list(target_cols_list)
    if group_col:
        drop_cols.append(group_col)
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    if any(len(X) != len(y) for _, y in y_list) or (
        groups is not None and len(X) != len(groups)
    ):
        raise DataError("X e Y com n_amostras diferente")
    return X, y_list, groups


def export_xy_xlsx(
    path: Path, X: pd.DataFrame, y: pd.Series, x_sheet: str = "X", y_sheet: str = "Y"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        X.to_excel(w, index=False, sheet_name=x_sheet)
        y.to_frame(name=y.name or "target").to_excel(w, index=False, sheet_name=y_sheet)


def prepare_inputs(
    data_cfg: Dict[str, Any], *, use_cache: bool = True
) -> List[Dict[str, Any]]:
    """Load and clean spectral datasets specified in configuration.

    Column names are normalised to floats (nanometers) and filtered by the
    provided spectral range. Infinite values are replaced by ``NaN`` and rows
    containing missing values are dropped in a leakage-free manner.
    """
    files = [Path(p) for p in glob(data_cfg["input_path"])]
    files = [p for p in files if not p.name.startswith("~$")]
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
    tgt = data_cfg.get("target_cols")
    if tgt is None and "target_col" in data_cfg:
        tgt = data_cfg["target_col"]
    grp = data_cfg.get("group_col")
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
            )
        else:
            X, y_list, groups = read_tabular(
                f,
                tgt,
                grp,
                encoding=data_cfg.get("encoding"),
                sep=data_cfg.get("sep", ","),
                decimal=data_cfg.get("decimal", "."),
            )
        X = _filter_range(X, sp)
        if X.empty:
            raise DataError("Intervalo espectral fora da faixa disponível")
        X = X.replace([np.inf, -np.inf], np.nan)
        n_nans = int(X.isna().any(axis=1).sum())
        X = X.dropna(axis=0)
        keep_idx = X.index
        if groups is not None:
            groups = groups.loc[keep_idx]
        X_reset = X.reset_index(drop=True)
        groups_reset = groups.reset_index(drop=True) if groups is not None else None

        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            fingerprint = {"mtime": f.stat().st_mtime, "spectral_range": sp}
            dump(
                {
                    "fingerprint": fingerprint,
                    "X": X_reset,
                    "keep_idx": keep_idx.tolist(),
                    "n_nans": n_nans,
                },
                cache_dir / f"{f.stem}.joblib",
            )

        for name, y in y_list:
            y = y.loc[keep_idx]
            ds = {
                "file": f.as_posix(),
                "target": name,
                "X": X_reset,
                "Y": y.reset_index(drop=True),
                "n_nans": n_nans,
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
        ``spectral_max`` and ``n_nans`` summarising each dataset.
    """

    rows: List[Dict[str, Any]] = []
    for ds in ds_list:
        X: pd.DataFrame = ds["X"]
        y: pd.Series = ds["Y"]
        spectral_min = float(min(X.columns)) if not X.empty else float("nan")
        spectral_max = float(max(X.columns)) if not X.empty else float("nan")
        n_nans = int(ds.get("n_nans", 0))
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
                }
            )
    return pd.DataFrame(rows)
