from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class ColorizationEngine(ABC):
    """Contract for any Pass2 colorization backend."""

    @abstractmethod
    def generate(
        self,
        image: Path,
        style_reference: Path,
        mask: Path,
        prompt: str,
        seed: int,
        strength: float = 1.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Generate a colorized image from Pass1 metadata inputs."""

    @abstractmethod
    def unload(self) -> None:
        """Free resources held by the engine."""
