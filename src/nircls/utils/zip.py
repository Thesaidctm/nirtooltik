from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def zip_dir(src: str | Path) -> BytesIO:
    """Create a ZIP archive from ``src`` directory.

    Parameters
    ----------
    src: str | Path
        Directory to compress.

    Returns
    -------
    BytesIO
        In-memory buffer containing the ZIP archive.
    """
    src_path = Path(src)
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as zf:
        for file in src_path.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(src_path))
    buffer.seek(0)
    return buffer
