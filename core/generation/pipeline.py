from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.generation.interfaces import ColorizationEngine
from core.utils.meta_validator import load_and_validate_metadata


def _path_from_meta(raw_path: str | Path) -> Path:
    """Normalize metadata paths across OS styles (e.g., Windows '\\' on Linux)."""
    normalized = str(raw_path).replace("\\", "/")
    return Path(normalized)


class Pass2Generator:
    def __init__(self, engine: ColorizationEngine):
        self.engine = engine

    def process_page(
        self,
        meta_path: str,
        output_dir: str,
        strength: float = 1.0,
        seed_override: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        t0 = time.perf_counter()
        meta = load_and_validate_metadata(meta_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        page_num = int(meta["page_num"])
        seed = int(seed_override) if seed_override is not None else int(meta["page_seed"])

        out_image_path = out_dir / f"page_{page_num:03d}_colorized.png"
        out_meta_path = out_dir / f"page_{page_num:03d}_colorized.runmeta.json"

        meta_path_obj = Path(meta_path)
        pass1_runmeta = meta_path_obj.with_suffix(".pass1.runmeta.json")

        runmeta: dict[str, Any] = {
            "meta_source": str(meta_path),
            "engine": self.engine.__class__.__name__,
            "seed": seed,
            "strength": float(strength),
            "status": "success",
            "page_num": page_num,
            "input_image": str(meta["page_image"]),
            "style_reference": str(meta["style_reference"]),
            "text_mask": str(meta["text_mask"]),
            "pass1_runmeta": str(pass1_runmeta),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "options": options or {},
        }

        try:
            result = self.engine.generate(
                image=_path_from_meta(meta["page_image"]),
                style_reference=_path_from_meta(meta["style_reference"]),
                mask=_path_from_meta(meta["text_mask"]),
                prompt=str(meta["page_prompt"]),
                seed=seed,
                strength=float(strength),
                options=options,
            )

            if isinstance(result, (str, Path)):
                shutil.copy2(result, out_image_path)
            else:
                result.save(out_image_path)

            runmeta["output_image"] = str(out_image_path)
        except Exception as exc:
            runmeta["status"] = "failed"
            runmeta["error"] = str(exc)
            runmeta["duration_ms"] = int((time.perf_counter() - t0) * 1000)
            out_meta_path.write_text(
                json.dumps(runmeta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise

        runmeta["duration_ms"] = int((time.perf_counter() - t0) * 1000)
        out_meta_path.write_text(
            json.dumps(runmeta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return str(out_image_path)
