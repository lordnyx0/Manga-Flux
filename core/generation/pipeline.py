from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.generation.interfaces import ColorizationEngine
from core.pipeline_state_store import PipelineStateStore
from core.utils.atomic_io import atomic_write_json
from core.utils.meta_validator import load_and_validate_metadata


def _path_from_meta(raw_path: str | Path) -> Path:
    """Normalize metadata paths across OS styles (e.g., Windows '\\' on Linux)."""
    normalized = str(raw_path).replace("\\", "/")
    return Path(normalized)


class Pass2Generator:
    def __init__(self, engine: ColorizationEngine, state_db_path: str | None = None):
        self.engine = engine
        self.state_db_path = state_db_path

    def process_page(
        self,
        meta_path: str,
        output_dir: str,
        strength: float = 1.0,
        seed_override: int | None = None,
        options: dict[str, Any] | None = None,
        debug_dump_json: bool = False,
    ) -> str:
        meta = load_and_validate_metadata(meta_path)
        return self._process_meta(
            meta=meta,
            meta_source=str(meta_path),
            output_dir=output_dir,
            strength=strength,
            seed_override=seed_override,
            options=options,
            debug_dump_json=debug_dump_json,
        )

    def process_page_from_state(
        self,
        chapter_id: str,
        page_num: int,
        output_dir: str,
        strength: float = 1.0,
        seed_override: int | None = None,
        options: dict[str, Any] | None = None,
        debug_dump_json: bool = False,
    ) -> str:
        if not self.state_db_path:
            raise ValueError("state_db_path is required for process_page_from_state")

        row = PipelineStateStore(self.state_db_path).get(chapter_id=chapter_id, page_num=page_num, stage="pass1")
        if not row:
            raise FileNotFoundError(f"Pass1 state not found for chapter={chapter_id} page={page_num}")

        pass1_metadata = (row.get("metadata") or {}).get("pass1_metadata")
        if not isinstance(pass1_metadata, dict):
            raise ValueError("Invalid pass1_metadata payload in pipeline state store")

        runtime_options = dict(options or {})
        runtime_options.setdefault("chapter_id", chapter_id)

        return self._process_meta(
            meta=pass1_metadata,
            meta_source=f"sqlite://{chapter_id}/{int(page_num):03d}/pass1",
            output_dir=output_dir,
            strength=strength,
            seed_override=seed_override,
            options=runtime_options,
            debug_dump_json=debug_dump_json,
        )

    def _process_meta(
        self,
        meta: dict[str, Any],
        meta_source: str,
        output_dir: str,
        strength: float,
        seed_override: int | None,
        options: dict[str, Any] | None,
        debug_dump_json: bool,
    ) -> str:
        t0 = time.perf_counter()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        page_num = int(meta["page_num"])
        seed = int(seed_override) if seed_override is not None else int(meta["page_seed"])

        out_image_path = out_dir / f"page_{page_num:03d}_colorized.png"
        out_meta_path = out_dir / f"page_{page_num:03d}_colorized.runmeta.json"

        runmeta: dict[str, Any] = {
            "meta_source": meta_source,
            "engine": self.engine.__class__.__name__,
            "seed": seed,
            "strength": float(strength),
            "status": "success",
            "page_num": page_num,
            "input_image": str(meta["page_image"]),
            "style_reference": str(meta["style_reference"]),
            "text_mask": str(meta["text_mask"]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "options": options or {},
        }

        chapter_id = str((options or {}).get("chapter_id", "default"))

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
            if debug_dump_json:
                atomic_write_json(out_meta_path, runmeta, ensure_ascii=False, indent=2)
            if self.state_db_path:
                PipelineStateStore(self.state_db_path).upsert(
                    chapter_id=chapter_id,
                    page_num=page_num,
                    stage="pass2",
                    status="failed",
                    metadata=runmeta,
                )
            raise

        runmeta["duration_ms"] = int((time.perf_counter() - t0) * 1000)
        if debug_dump_json:
            atomic_write_json(out_meta_path, runmeta, ensure_ascii=False, indent=2)

        if self.state_db_path:
            PipelineStateStore(self.state_db_path).upsert(
                chapter_id=chapter_id,
                page_num=page_num,
                stage="pass2",
                status=str(runmeta.get("status", "success")),
                metadata=runmeta,
            )

        return str(out_image_path)
