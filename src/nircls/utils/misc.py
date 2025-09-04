
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import random, os, numpy as np
def ensure_dirs(*paths: Path):
    for p in paths: p.mkdir(parents=True, exist_ok=True)
def now_ts(): return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
def make_run_id(prefix:str): return f"{prefix}_{now_ts()}"
def set_seed(seed:int=42): random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
