"""Test configuration for ensuring package imports."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure that the ``src`` directory is available on the import path so that
# tests can import the project package without requiring an editable
# installation. This mirrors the behaviour of the packaging metadata while
# keeping the test environment lightweight.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.is_dir():
    sys.path.insert(0, str(SRC_PATH))
