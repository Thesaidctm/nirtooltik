
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np, pandas as pd
from glob import glob

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
):
    X = pd.read_excel(path, sheet_name=x_sheet)
    Ydf = pd.read_excel(path, sheet_name=y_sheet)
    if target_cols in [None, "all"]:
        target_cols = list(Ydf.columns)
    elif isinstance(target_cols, str):
        target_cols = [target_cols]
    out: List[Tuple[str, pd.Series]] = []
    for col in target_cols:
        if col not in Ydf.columns:
            raise ValueError(f"Coluna alvo {col} não encontrada em {path}")
        out.append((col, Ydf[col].astype(str)))
    X = X.select_dtypes(include=[np.number])
    if any(len(X) != len(y) for _, y in out):
        raise ValueError("X e Y com n_amostras diferente")
    return X, out

def read_tabular(path: Path, target_cols: Optional[Sequence[str] | str] = None):
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    if target_cols in [None, "all"]:
        target_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    elif isinstance(target_cols, str):
        target_cols = [target_cols]
    y_list: List[Tuple[str, pd.Series]] = []
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna alvo {col} não encontrada em {path}")
        y_list.append((col, df[col].astype(str)))
    X = df.drop(columns=target_cols).select_dtypes(include=[np.number])
    if any(len(X) != len(y) for _, y in y_list):
        raise ValueError("X e Y com n_amostras diferente")
    return X, y_list

def export_xy_xlsx(path: Path, X: pd.DataFrame, y: pd.Series, x_sheet="X", y_sheet="Y"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        X.to_excel(w, index=False, sheet_name=x_sheet)
        y.to_frame(name=y.name or "target").to_excel(w, index=False, sheet_name=y_sheet)

def prepare_inputs(data_cfg: Dict):
    """Load and clean spectral datasets specified in configuration.

    Column names are normalised to floats (nanometers) and filtered by the
    provided spectral range. Infinite values are replaced by ``NaN`` and rows
    containing missing values are dropped in a leakage-free manner.
    """
    files = [Path(p) for p in glob(data_cfg["input_path"])]
    sp = data_cfg.get("spectral_range", None)
    sp = (float(sp[0]), float(sp[1])) if sp else None
    tgt = data_cfg.get("target_cols")
    if tgt is None and "target_col" in data_cfg:
        tgt = data_cfg["target_col"]
    out = []
    for f in files:
        if f.suffix.lower() in [".xlsx", ".xls"] and ("x_sheet" in data_cfg or "y_sheet" in data_cfg):
            X, y_list = read_xlsx_xy(
                f,
                data_cfg.get("x_sheet", "X"),
                data_cfg.get("y_sheet", "Y"),
                tgt,
            )
        else:
            X, y_list = read_tabular(f, tgt)
        X = _filter_range(X, sp)
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        for name, y in y_list:
            y = y.loc[X.index]
            out.append(
                {
                    "file": f.as_posix(),
                    "target": name,
                    "X": X.reset_index(drop=True),
                    "Y": y.reset_index(drop=True),
                }
            )
    return out
