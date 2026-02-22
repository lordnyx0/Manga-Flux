from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from core.utils.atomic_io import atomic_write_text

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path for minimal runtimes
    PYDANTIC_AVAILABLE = False


if PYDANTIC_AVAILABLE:

    class Pass1Metadata(BaseModel):
        """Strict Pass1 -> Pass2 metadata contract."""

        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

        page_num: int = Field(ge=0)
        page_image: str = Field(min_length=1)
        page_seed: int
        page_prompt: str = Field(min_length=1)
        style_reference: str = Field(min_length=1)
        text_mask: str = Field(min_length=1)
        detections: list[dict] = Field(default_factory=list)

        @classmethod
        def load(cls, meta_path: str | Path) -> "Pass1Metadata":
            path = Path(meta_path)
            if not path.exists():
                raise FileNotFoundError(f"Metadata file not found: {path}")

            payload = json.loads(path.read_text(encoding="utf-8"))
            if os.getenv("ALLOW_NO_STYLE", "0") == "1" and not payload.get("style_reference"):
                payload["style_reference"] = "__STYLE_OPTIONAL__"

            try:
                model = cls.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"Invalid metadata contract in {path}: {exc}") from exc

            if model.style_reference == "__STYLE_OPTIONAL__":
                model.style_reference = ""
            return model

        def dump_json(self, file_path: str | Path) -> Path:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, self.model_dump_json(indent=2), encoding="utf-8")
            return path

else:

    @dataclass
    class Pass1Metadata:
        page_num: int
        page_image: str
        page_seed: int
        page_prompt: str
        style_reference: str
        text_mask: str
        detections: list[dict] = None

        def __post_init__(self):
            if self.detections is None:
                self.detections = []

        @classmethod
        def load(cls, meta_path: str | Path) -> "Pass1Metadata":
            path = Path(meta_path)
            if not path.exists():
                raise FileNotFoundError(f"Metadata file not found: {path}")

            payload = json.loads(path.read_text(encoding="utf-8"))
            if os.getenv("ALLOW_NO_STYLE", "0") == "1" and not payload.get("style_reference"):
                payload["style_reference"] = ""

            required = ["page_num", "page_image", "page_seed", "page_prompt", "style_reference", "text_mask"]
            missing = [k for k in required if k not in payload or payload[k] in (None, "")]
            if missing:
                raise ValueError(f"Invalid metadata contract in {path}: missing {', '.join(missing)}")

            return cls(
                page_num=int(payload["page_num"]),
                page_image=str(payload["page_image"]),
                page_seed=int(payload["page_seed"]),
                page_prompt=str(payload["page_prompt"]),
                style_reference=str(payload["style_reference"]),
                text_mask=str(payload["text_mask"]),
                detections=payload.get("detections", []),
            )

        def model_dump(self) -> dict:
            return asdict(self)

        def dump_json(self, file_path: str | Path) -> Path:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps(self.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
            return path
