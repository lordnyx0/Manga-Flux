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
from core.pipeline_state_store import PipelineStateStore

logger = logging.getLogger("Pass1Pipeline")

DEFAULT_MASK_TEMPLATE = Path("outputs/test_run/masks/page_001_text.png")


@dataclass
class Pass1RunReport:
    metadata_path: Path | None
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


def _generate_mask_with_ported_pass1(page_image: str, output_mask: str, page_num: int) -> tuple[bool, str, list, list | None]:
    """Retorna (success, reason, detections, vlm_character_registry)."""
    try:
        from core.pass1_analyzer import Pass1Analyzer
    except Exception as exc:
        reason = f"Pass1Analyzer unavailable ({exc})"
        logger.warning(reason)
        return False, reason, [], None

    output_mask_path = Path(output_mask)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        analyzer = Pass1Analyzer()
        result = analyzer.analyze_page(page_image, page_num=page_num)
        text_mask = result.get("text_mask")
        lineart = result.get("lineart")
        detections = result.get("detections", [])
        # Captura o registry VLM produzido durante a análise (apenas page_num == -1 o gera,
        # mas retornamos sempre para uniformidade da interface)
        vlm_registry = getattr(analyzer, "_vlm_global_characters", None)
        
        # Salva a máscara de texto
        mask_saved = _save_mask_array(text_mask, output_mask_path)
        
        # Salva a lineart extraída se disponível
        if lineart is not None:
            try:
                # Substitui a pasta 'masks' por 'linearts' e o sufixo '_text.png' por '_lineart.png'
                output_lineart_path = Path(str(output_mask_path).replace("masks", "linearts").replace("_text.png", "_lineart.png"))
                output_lineart_path.parent.mkdir(parents=True, exist_ok=True)
                if _save_mask_array(lineart, output_lineart_path):
                    logger.info("Pass1 analyzer generated and saved lineart: %s", output_lineart_path)
            except Exception as lineart_exc:
                logger.warning("Failed to save extracted lineart: %s", lineart_exc)

        if mask_saved:
            logger.info("Pass1 analyzer generated text mask: %s", output_mask_path)
            return True, "", detections, vlm_registry

        reason = "Pass1 analyzer ran, but mask could not be serialized"
        logger.warning(reason)
        return False, reason, detections, vlm_registry
    except Exception as exc:
        reason = f"Pass1 analyzer execution failed ({exc})"
        logger.warning(reason)
        return False, reason, [], None


def generate_text_mask(page_image: str, output_mask: str, page_num: int) -> tuple[Path, str, str, Dict[str, bool], list, list | None]:
    """Retorna (mask_path, mode, reason, deps, detections, vlm_character_registry)."""
    output_mask_path = Path(output_mask)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    probe = probe_pass1_dependencies()
    deps = probe.availability

    ok, reason, detections, vlm_registry = _generate_mask_with_ported_pass1(
        page_image=page_image,
        output_mask=output_mask,
        page_num=page_num,
    )
    if ok:
        return output_mask_path, "ported_pass1", "", deps, detections, vlm_registry

    if DEFAULT_MASK_TEMPLATE.exists():
        shutil.copy2(DEFAULT_MASK_TEMPLATE, output_mask_path)
        logger.info("Using fallback template mask: %s", output_mask_path)
        return output_mask_path, "template_fallback", reason, deps, detections, vlm_registry

    try:
        from PIL import Image
        img = Image.new("L", (1024, 1024), 0)
        img.save(output_mask_path)
        reason = reason or "template mask not found (created blank)"
        logger.warning("No template mask found; writing blank PIL image: %s", output_mask_path)
    except Exception as exc:
        logger.error("Failed to write blanket mask fallback: %s", exc)
        reason = reason or "template mask not found"

    return output_mask_path, "empty_fallback", reason, deps, detections, vlm_registry


def run_pass1_with_report(
    page_image: str,
    style_reference: str,
    output_mask: str,
    output_metadata_dir: str,
    page_num: int,
    page_prompt: str,
    chapter_id: str = "default",
    state_db_path: str | None = None,
    debug_dump_json: bool = False,
) -> Pass1RunReport:
    t0 = time.perf_counter()

    mask_file, mode, fallback_reason, deps, detections, vlm_registry = generate_text_mask(
        page_image=page_image,
        output_mask=output_mask,
        page_num=page_num,
    )
    seed = deterministic_seed(chapter_id=chapter_id, page_num=page_num)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    metadata_payload = {
        "page_num": int(page_num),
        "page_image": str(page_image),
        "page_seed": int(seed),
        "page_prompt": str(page_prompt),
        "style_reference": str(style_reference),
        "text_mask": str(mask_file),
        "detections": detections,
        # Registry VLM da capa — propagado para o Pass2 via SQLite/meta.json
        # Permite ao Pass2Orchestrator ativar o Caminho 1 (prompt modular Gemma)
        # mesmo quando o Pass1 e Pass2 rodam em momentos distintos.
        "vlm_character_registry": vlm_registry,
    }
    pass1_runmeta_payload = {
        "mode": mode,
        "fallback_reason": fallback_reason,
        "dependencies": deps,
        "duration_ms": duration_ms,
        "status": "success",
    }

    metadata_file = write_pass1_metadata(
        output_dir=output_metadata_dir,
        page_num=page_num,
        page_image=page_image,
        page_seed=seed,
        page_prompt=page_prompt,
        style_reference=style_reference,
        text_mask=str(mask_file),
        detections=detections,
    )
    runmeta_file = write_pass1_runmeta(
        metadata_path=metadata_file,
        mode=mode,
        fallback_reason=fallback_reason,
        dependencies=deps,
        duration_ms=duration_ms,
    )

    if state_db_path:
        PipelineStateStore(state_db_path).upsert(
            chapter_id=chapter_id,
            page_num=page_num,
            stage="pass1",
            status="success",
            metadata={
                "pass1_metadata": metadata_payload,
                "pass1_runmeta": pass1_runmeta_payload,
                "metadata_path": str(metadata_file) if metadata_file else "",
                "runmeta_path": str(runmeta_file) if runmeta_file else "",
                "mask_path": str(mask_file),
            },
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
    state_db_path: str | None = None,
    debug_dump_json: bool = False,
) -> Path | None:
    return run_pass1_with_report(
        page_image=page_image,
        style_reference=style_reference,
        output_mask=output_mask,
        output_metadata_dir=output_metadata_dir,
        page_num=page_num,
        page_prompt=page_prompt,
        chapter_id=chapter_id,
        state_db_path=state_db_path,
        debug_dump_json=debug_dump_json,
    ).metadata_path
