from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from core.analysis.dependency_probe import probe_pass1_dependencies
from core.analysis.pass1_contract import (
    deterministic_seed,
    write_pass1_metadata,
    write_pass1_runmeta,
)

logger = logging.getLogger("Pass1Pipeline")

DEFAULT_MASK_TEMPLATE = Path("outputs/test_run/masks/page_001_text.png")


@dataclass
class Pass1RunReport:
    metadata_path: Path
    mask_path: Path
    mode: str  # "ported_pass1" | "template_fallback" | "empty_fallback"
    fallback_reason: str = ""
    dependencies: Dict[str, bool] = field(default_factory=dict)
    duration_ms: int = 0
    runmeta_path: Path | None = None


def _save_mask_array(mask, output_mask_path: Path) -> bool:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return False

    if mask is None:
        return False

    try:
        arr = np.asarray(mask).astype("uint8")
    except Exception:
        return False

    try:
        from PIL import Image  # type: ignore

        Image.fromarray(arr).save(output_mask_path)
        return True
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        cv2.imwrite(str(output_mask_path), arr)
        return True
    except Exception:
        return False


def _generate_mask_with_ported_pass1(page_image: str, output_mask: str, page_num: int) -> tuple[bool, str]:
    try:
        from core.pass1_analyzer import Pass1Analyzer
    except Exception as exc:
        reason = f"Pass1Analyzer unavailable ({exc})"
        logger.warning(reason)
        return False, reason

    output_mask_path = Path(output_mask)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        analyzer = Pass1Analyzer()
        result = analyzer.analyze_page(page_image, page_num=page_num)
        text_mask = result.get("text_mask")
        if _save_mask_array(text_mask, output_mask_path):
            logger.info("Pass1 analyzer generated text mask: %s", output_mask_path)
            return True, ""

        reason = "Pass1 analyzer ran, but mask could not be serialized"
        logger.warning(reason)
        return False, reason
    except Exception as exc:
        reason = f"Pass1 analyzer execution failed ({exc})"
        logger.warning(reason)
        return False, reason


def generate_text_mask(page_image: str, output_mask: str, page_num: int) -> tuple[Path, str, str, Dict[str, bool]]:
    output_mask_path = Path(output_mask)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    probe = probe_pass1_dependencies()
    deps = probe.availability

    ok, reason = _generate_mask_with_ported_pass1(
        page_image=page_image,
        output_mask=output_mask,
        page_num=page_num,
    )
    if ok:
        return output_mask_path, "ported_pass1", "", deps

    if DEFAULT_MASK_TEMPLATE.exists():
        shutil.copy2(DEFAULT_MASK_TEMPLATE, output_mask_path)
        logger.info("Using fallback template mask: %s", output_mask_path)
        return output_mask_path, "template_fallback", reason, deps

    output_mask_path.write_bytes(b"")
    reason = reason or "template mask not found"
    logger.warning("No template mask found; writing empty placeholder: %s", output_mask_path)
    return output_mask_path, "empty_fallback", reason, deps


def run_pass1_with_report(
    page_image: str,
    style_reference: str,
    output_mask: str,
    output_metadata_dir: str,
    page_num: int,
    page_prompt: str,
    chapter_id: str = "default",
) -> Pass1RunReport:
    t0 = time.perf_counter()

    mask_file, mode, fallback_reason, deps = generate_text_mask(
        page_image=page_image,
        output_mask=output_mask,
        page_num=page_num,
    )
    seed = deterministic_seed(chapter_id=chapter_id, page_num=page_num)

    metadata_file = write_pass1_metadata(
        output_dir=output_metadata_dir,
        page_num=page_num,
        page_image=page_image,
        page_seed=seed,
        page_prompt=page_prompt,
        style_reference=style_reference,
        text_mask=str(mask_file),
    )

    duration_ms = int((time.perf_counter() - t0) * 1000)

    runmeta_file = write_pass1_runmeta(
        metadata_path=metadata_file,
        mode=mode,
        fallback_reason=fallback_reason,
        dependencies=deps,
        duration_ms=duration_ms,
    )

    return Pass1RunReport(
        metadata_path=metadata_file,
        mask_path=mask_file,
        mode=mode,
        fallback_reason=fallback_reason,
        dependencies=deps,
        duration_ms=duration_ms,
        runmeta_path=runmeta_file,
    )


def run_pass1(
    page_image: str,
    style_reference: str,
    output_mask: str,
    output_metadata_dir: str,
    page_num: int,
    page_prompt: str,
    chapter_id: str = "default",
) -> Path:
    return run_pass1_with_report(
        page_image=page_image,
        style_reference=style_reference,
        output_mask=output_mask,
        output_metadata_dir=output_metadata_dir,
        page_num=page_num,
        page_prompt=page_prompt,
        chapter_id=chapter_id,
    ).metadata_path
