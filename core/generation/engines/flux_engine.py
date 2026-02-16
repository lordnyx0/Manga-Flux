from __future__ import annotations

import os
from pathlib import Path

from core.generation.interfaces import ColorizationEngine


class FluxEngine(ColorizationEngine):
    """Flux skeleton (mock img2img).

    In minimal environments (without Pillow), this skeleton returns the
    original image path after validating metadata contract constraints.
    """

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
        del mask, prompt, seed, strength, options

        allow_no_style = os.getenv("ALLOW_NO_STYLE", "0") == "1"
        if not allow_no_style:
            if not style_reference or not Path(style_reference).exists():
                raise ValueError("style_reference is mandatory and must exist")

        if not Path(image).exists():
            raise FileNotFoundError(f"Input image not found: {image}")

        return Path(image)

    def unload(self) -> None:
        return None
