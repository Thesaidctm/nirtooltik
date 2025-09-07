#!/usr/bin/env python3
"""Gera um pacote zip contendo o código-fonte e configurações do nircls."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import zipfile

IGNORE_PATTERNS = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")

def main() -> None:
    project_root = Path(__file__).resolve().parent

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Copia diretórios relevantes
        shutil.copytree(project_root / "src" / "nircls", tmp_path / "src" / "nircls", ignore=IGNORE_PATTERNS)
        shutil.copytree(project_root / "configs", tmp_path / "configs", ignore=IGNORE_PATTERNS)

        # Copia arquivos essenciais
        for fname in ["README.md", "pyproject.toml"]:
            src = project_root / fname
            if src.exists():
                shutil.copy(src, tmp_path / fname)

        # Gera requirements.txt com dependências atuais
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        (tmp_path / "requirements.txt").write_text(result.stdout)

        # Compacta tudo em nircls_bundle.zip
        bundle_path = project_root / "nircls_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in tmp_path.rglob("*"):
                zf.write(file, file.relative_to(tmp_path))

        print(f"Bundle criado em {bundle_path}")


if __name__ == "__main__":
    main()
