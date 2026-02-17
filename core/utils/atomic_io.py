from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """Write text atomically using a temporary file + os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile("w", encoding=encoding, dir=target.parent, delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, target)
    return target


def atomic_write_json(path: str | Path, payload: Any, *, ensure_ascii: bool = False, indent: int = 2) -> Path:
    """Serialize payload as JSON and write atomically."""
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent),
        encoding="utf-8",
    )
