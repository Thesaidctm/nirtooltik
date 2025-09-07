#!/usr/bin/env python
"""DVC stage to prepare input data."""
import sys
from pathlib import Path
import yaml
from nircls.io import prepare_inputs, export_xy_xlsx

def main(cfg_path: str, out_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))[0]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_xy_xlsx(out, ds["X"], ds["Y"])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: dvc_prepare.py <config> <out_file>")
    main(sys.argv[1], sys.argv[2])
