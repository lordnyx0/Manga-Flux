from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ObjectDetector(Protocol):
    """Minimal detector contract consumed by Pass1 pipeline logic."""

    def detect(self, image: np.ndarray) -> list[Any]:
        """Return raw detections for the provided RGB image."""

    def group_body_face_pairs(
        self,
        image: np.ndarray,
        detections: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Group detections into semantic character pairs (body/face)."""
