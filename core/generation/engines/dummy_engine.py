from __future__ import annotations

from pathlib import Path

from core.generation.interfaces import ColorizationEngine


class DummyEngine(ColorizationEngine):
    """Minimal test engine: returns source image path unchanged."""

    def generate(
        self,
        image: Path,
        style_reference: Path,
        mask: Path,
        prompt: str,
        seed: int,
        strength: float = 1.0,
        options=None,
    ) -> Path:
        del style_reference, mask, prompt, seed, strength, options
        return Path(image)

    def unload(self) -> None:
        return None
