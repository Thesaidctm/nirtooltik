#!/usr/bin/env python
"""DVC stage to train model."""
import sys
from pathlib import Path
import yaml
from nircls.pipeline import train_and_save

def main(cfg_path: str, out_dir: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_and_save(cfg, out)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: dvc_train.py <config> <out_dir>")
    main(sys.argv[1], sys.argv[2])
