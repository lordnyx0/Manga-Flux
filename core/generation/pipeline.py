from __future__ import annotations

import json
import shutil
from pathlib import Path

from core.generation.interfaces import ColorizationEngine
from core.utils.meta_validator import load_and_validate_metadata


class Pass2Generator:
    def __init__(self, engine: ColorizationEngine):
        self.engine = engine

    def process_page(self, meta_path: str, output_dir: str) -> str:
        meta = load_and_validate_metadata(meta_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        page_num = int(meta["page_num"])
        seed = int(meta["page_seed"])

        out_image_path = out_dir / f"page_{page_num:03d}_colorized.png"
        out_meta_path = out_dir / f"page_{page_num:03d}_colorized.runmeta.json"

        result = self.engine.generate(
            image=Path(meta["page_image"]),
            style_reference=Path(meta["style_reference"]),
            mask=Path(meta["text_mask"]),
            prompt=str(meta["page_prompt"]),
            seed=seed,
            strength=1.0,
            options=None,
        )

        if isinstance(result, (str, Path)):
            shutil.copy2(result, out_image_path)
        else:
            result.save(out_image_path)

        meta_path_obj = Path(meta_path)
        pass1_runmeta = meta_path_obj.with_suffix(".pass1.runmeta.json")

        runmeta = {
            "meta_source": str(meta_path),
            "engine": self.engine.__class__.__name__,
            "seed": seed,
            "strength": 1.0,
            "status": "success",
            "page_num": page_num,
            "input_image": str(meta["page_image"]),
            "style_reference": str(meta["style_reference"]),
            "text_mask": str(meta["text_mask"]),
            "pass1_runmeta": str(pass1_runmeta),
        }
        out_meta_path.write_text(
            json.dumps(runmeta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return str(out_image_path)
