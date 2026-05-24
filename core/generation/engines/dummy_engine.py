from __future__ import annotations

from pathlib import Path

from core.generation.interfaces import ColorizationEngine


class DummyEngine(ColorizationEngine):
    """Minimal test engine: returns source image path unchanged."""

    def generate(
        self,
        payload: dict,
        seed: int,
        strength: float = 1.0,
        options: dict = None,
    ) -> tuple[Path, dict]:
        base_image_path = payload.get("base_image_path")
        return Path(base_image_path), {"duration_ms": 0, "status": "success"}

    def unload(self) -> None:
        return None
