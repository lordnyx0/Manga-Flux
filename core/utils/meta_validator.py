from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.analysis.metadata_contract import Pass1Metadata


def load_and_validate_metadata(meta_path: str | Path) -> Dict[str, Any]:
    """Load metadata JSON and enforce Pass1->Pass2 contract via Pydantic."""
    return Pass1Metadata.load(meta_path).model_dump()
