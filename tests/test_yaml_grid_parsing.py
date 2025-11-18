import numpy as np
from sklearn.model_selection import GridSearchCV
import yaml
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.processing.pipeline import build_pipeline_from_config  # noqa: E402


def test_yaml_grid_parsing_creates_valid_grid():
    yaml_cfg = """
models:
  - name: MLP
    grid:
      hidden_layer_sizes:
        - [5]
        - [10]
      alpha: [0.0001]
      max_iter: [50]
      early_stopping: [True]
      validation_fraction: [0.2]
"""
    cfg = yaml.safe_load(yaml_cfg)
    X = np.random.RandomState(0).randn(20, 10)
    y = (np.arange(20) % 2).astype(int)
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "MLP"
    assert grid["clf__hidden_layer_sizes"] == [(5,), (10,)]
    gs = GridSearchCV(pipe, grid, cv=2, n_jobs=1)
    gs.fit(X, y)
    assert gs.best_params_["clf__hidden_layer_sizes"] in [(5,), (10,)]


def test_yaml_grid_parsing_reconstructs_split_tuple_strings():
    yaml_cfg = """
models:
  - name: MLP
    grid:
      hidden_layer_sizes:
        - (100,
        - )
        - (200,
        - )
      alpha: [0.0001]
      max_iter: [50]
      early_stopping: [True]
      validation_fraction: [0.2]
"""
    cfg = yaml.safe_load(yaml_cfg)
    X = np.random.RandomState(0).randn(20, 10)
    y = (np.arange(20) % 2).astype(int)
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "MLP"
    assert grid["clf__hidden_layer_sizes"] == [(100,), (200,)]
    gs = GridSearchCV(pipe, grid, cv=2, n_jobs=1)
    gs.fit(X, y)
    assert gs.best_params_["clf__hidden_layer_sizes"] in [(100,), (200,)]
