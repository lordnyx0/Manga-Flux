from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.analysis.metadata_contract import Pass1Metadata


def write_pass1_metadata(
    output_dir: str | Path,
    page_num: int,
    page_image: str | Path,
    page_seed: int,
    page_prompt: str,
    style_reference: str | Path,
    text_mask: str | Path,
) -> Path:
    """Write metadata/page_{NNN}.meta.json in the Pass1->Pass2 contract."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = Pass1Metadata(
        page_num=int(page_num),
        page_image=str(page_image),
        page_seed=int(page_seed),
        page_prompt=str(page_prompt),
        style_reference=str(style_reference),
        text_mask=str(text_mask),
    )

    file_path = output_dir / f"page_{int(page_num):03d}.meta.json"
    return metadata.dump_json(file_path)


def deterministic_seed(chapter_id: str, page_num: int) -> int:
    """Stable seed helper for Pass1 metadata export."""
    import hashlib

    raw = f"{chapter_id}:{int(page_num)}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest(), 16) % (2**31)


def write_pass1_runmeta(
    metadata_path: str | Path,
    mode: str,
    fallback_reason: str = "",
    dependencies: dict | None = None,
    duration_ms: int = 0,
) -> Path:
    """Write pass1 execution metadata alongside page metadata file."""
    metadata_path = Path(metadata_path)
    runmeta_path = metadata_path.with_suffix(".pass1.runmeta.json")
    payload = {
        "metadata_file": str(metadata_path),
        "mode": mode,
        "fallback_reason": fallback_reason,
        "dependencies": dependencies or {},
        "duration_ms": int(duration_ms),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }
    runmeta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return runmeta_path
