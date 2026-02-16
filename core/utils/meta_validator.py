from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable

REQUIRED_KEYS = (
    "page_num",
    "page_image",
    "page_seed",
    "page_prompt",
    "style_reference",
    "text_mask",
)


def _missing_keys(meta: Dict[str, Any], required: Iterable[str]) -> list[str]:
    return [key for key in required if key not in meta or meta[key] in (None, "")]


def load_and_validate_metadata(meta_path: str | Path) -> Dict[str, Any]:
    """Load metadata JSON and enforce Pass1->Pass2 contract.

    If ALLOW_NO_STYLE=1, style_reference may be missing/empty.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as fp:
        meta = json.load(fp)

    allow_no_style = os.getenv("ALLOW_NO_STYLE", "0") == "1"
    required = REQUIRED_KEYS if not allow_no_style else tuple(
        key for key in REQUIRED_KEYS if key != "style_reference"
    )

    missing = _missing_keys(meta, required)
    if missing:
        raise ValueError(
            "Invalid metadata contract. Missing required keys: "
            + ", ".join(missing)
        )

    return meta
