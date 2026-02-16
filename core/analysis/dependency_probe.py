from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict


@dataclass
class DependencyProbeResult:
    availability: Dict[str, bool]

    @property
    def all_required_for_ported_pass1(self) -> bool:
        required = ("torch", "numpy", "PIL", "cv2")
        return all(self.availability.get(dep, False) for dep in required)

    def missing_required(self) -> list[str]:
        required = ("torch", "numpy", "PIL", "cv2")
        return [dep for dep in required if not self.availability.get(dep, False)]


def _is_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def probe_pass1_dependencies() -> DependencyProbeResult:
    """Probe availability of core and optional Pass1 dependencies."""
    modules = {
        "torch": "torch",
        "numpy": "numpy",
        "PIL": "PIL",
        "cv2": "cv2",
        "ultralytics": "ultralytics",
        "transformers": "transformers",
        "insightface": "insightface",
        "sklearn": "sklearn",
        "skimage": "skimage",
    }
    availability = {name: _is_importable(mod) for name, mod in modules.items()}
    return DependencyProbeResult(availability=availability)
