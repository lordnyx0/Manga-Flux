from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.generation.interfaces import ColorizationEngine
from core.pipeline_state_store import PipelineStateStore
from core.utils.atomic_io import atomic_write_json
from core.utils.meta_validator import load_and_validate_metadata


def _path_from_meta(raw_path: str | Path) -> Path:
    """Normalize metadata paths across OS styles (e.g., Windows '\\' on Linux)."""
    normalized = str(raw_path).replace("\\", "/")
    return Path(normalized)


@dataclasses.dataclass
class Pass2PreparedPayload:
    """
    Resultado da fase de pré-computação do Pass2.

    Encapsula o payload completo (incluindo o prompt gerado pelo Gemma) para que
    a engine de geração (Flux/ComfyUI) possa ser invocada separadamente.

    Em modo batch, chame prepare_page_from_state() para TODAS as páginas primeiro,
    depois descarregue o Gemma (LLAMACppServerManager.stop_server()) uma única vez,
    e por fim chame execute_prepared() para cada payload. Isso evita ciclos
    repetidos de carga/descarga de VRAM e garante que a GPU esteja 100% disponível
    para o KSampler durante a geração.
    """
    payload: dict[str, Any]
    seed: int
    strength: float
    out_image_path: Path
    out_meta_path: Path
    meta: dict[str, Any]
    page_num: int
    chapter_id: str
    meta_source: str
    options: dict[str, Any]
    runmeta_base: dict[str, Any]
    faiss_service: Any   # referência ao FaissService do orchestrator para validação colorimétrica
    t0: float            # timestamp de início para cálculo de duration_ms


class Pass2Generator:
    def __init__(self, engine: ColorizationEngine, state_db_path: str | None = None):
        self.engine = engine
        self.state_db_path = state_db_path

    # ── API pública: single-page ───────────────────────────────────────────────

    def process_page(
        self,
        meta_path: str,
        output_dir: str,
        strength: float = 0.85,
        seed_override: int | None = None,
        options: dict[str, Any] | None = None,
        debug_dump_json: bool = False,
    ) -> str:
        meta = load_and_validate_metadata(meta_path)
        return self._process_meta(
            meta=meta,
            meta_source=str(meta_path),
            output_dir=output_dir,
            strength=strength,
            seed_override=seed_override,
            options=options,
            debug_dump_json=debug_dump_json,
        )

    def process_page_from_state(
        self,
        chapter_id: str,
        page_num: int,
        output_dir: str,
        strength: float = 0.85,
        seed_override: int | None = None,
        options: dict[str, Any] | None = None,
        debug_dump_json: bool = False,
    ) -> str:
        if not self.state_db_path:
            raise ValueError("state_db_path is required for process_page_from_state")

        row = PipelineStateStore(self.state_db_path).get(chapter_id=chapter_id, page_num=page_num, stage="pass1")
        if not row:
            raise FileNotFoundError(f"Pass1 state not found for chapter={chapter_id} page={page_num}")

        pass1_metadata = (row.get("metadata") or {}).get("pass1_metadata")
        if not isinstance(pass1_metadata, dict):
            raise ValueError("Invalid pass1_metadata payload in pipeline state store")

        runtime_options = dict(options or {})
        runtime_options.setdefault("chapter_id", chapter_id)

        return self._process_meta(
            meta=pass1_metadata,
            meta_source=f"sqlite://{chapter_id}/{int(page_num):03d}/pass1",
            output_dir=output_dir,
            strength=strength,
            seed_override=seed_override,
            options=runtime_options,
            debug_dump_json=debug_dump_json,
        )

    # ── API pública: batch (pre-compute + execute separados) ───────────────────

    def prepare_page_from_state(
        self,
        chapter_id: str,
        page_num: int,
        output_dir: str,
        strength: float = 0.85,
        seed_override: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Pass2PreparedPayload:
        """
        Fase de pré-computação (Fase 2 do batch):
        carrega o metadata do SQLite, aciona o Pass2Orchestrator (que chama o Gemma
        internamente) e devolve o payload completo SEM acionar a engine de geração.

        Uso recomendado em batch:
          1. Chame esta função para TODAS as páginas.
          2. Descarregue o Gemma: LLAMACppServerManager.stop_server()
          3. Chame execute_prepared() para cada Pass2PreparedPayload retornado.
        """
        if not self.state_db_path:
            raise ValueError("state_db_path is required for prepare_page_from_state")

        row = PipelineStateStore(self.state_db_path).get(chapter_id=chapter_id, page_num=page_num, stage="pass1")
        if not row:
            raise FileNotFoundError(f"Pass1 state not found for chapter={chapter_id} page={page_num}")

        pass1_metadata = (row.get("metadata") or {}).get("pass1_metadata")
        if not isinstance(pass1_metadata, dict):
            raise ValueError("Invalid pass1_metadata payload in pipeline state store")

        runtime_options = dict(options or {})
        runtime_options.setdefault("chapter_id", chapter_id)

        return self._prepare_payload(
            meta=pass1_metadata,
            meta_source=f"sqlite://{chapter_id}/{int(page_num):03d}/pass1",
            output_dir=output_dir,
            strength=strength,
            seed_override=seed_override,
            options=runtime_options,
        )

    def execute_prepared(
        self,
        prepared: Pass2PreparedPayload,
        debug_dump_json: bool = False,
    ) -> str:
        """
        Fase de geração (Fase 3 do batch):
        recebe o payload pré-computado e aciona a engine de geração (ComfyUI/Flux).

        IMPORTANTE: em uso batch, o Gemma JÁ DEVE TER SIDO descarregado antes desta
        chamada (pelo run_two_pass_batch_local.py). Esta função NÃO chama stop_server()
        — a responsabilidade é do caller no nível batch.

        Em uso single-page, prefira process_page() ou process_page_from_state(),
        que gerenciam o ciclo de vida do VLM automaticamente.
        """
        return self._execute_payload(prepared, stop_vlm=False, debug_dump_json=debug_dump_json)

    # ── Implementação interna ──────────────────────────────────────────────────

    def _process_meta(
        self,
        meta: dict[str, Any],
        meta_source: str,
        output_dir: str,
        strength: float,
        seed_override: int | None,
        options: dict[str, Any] | None,
        debug_dump_json: bool,
    ) -> str:
        """
        Pipeline completo single-page: prepara payload (chama Gemma), descarrega o
        VLM se VLM_UNLOAD_SCENARIO == 1, depois aciona a engine de geração.
        """
        prepared = self._prepare_payload(
            meta=meta,
            meta_source=meta_source,
            output_dir=output_dir,
            strength=strength,
            seed_override=seed_override,
            options=options,
        )
        return self._execute_payload(prepared, stop_vlm=True, debug_dump_json=debug_dump_json)

    def _prepare_payload(
        self,
        meta: dict[str, Any],
        meta_source: str,
        output_dir: str,
        strength: float,
        seed_override: int | None,
        options: dict[str, Any] | None,
    ) -> Pass2PreparedPayload:
        """
        Aciona o Pass2Orchestrator (que por sua vez chama o Gemma se disponível)
        e devolve um Pass2PreparedPayload pronto para ser executado pela engine.
        Não faz nenhum I/O de imagem nem chama stop_server().
        """
        t0 = time.perf_counter()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        page_num = int(meta["page_num"])
        seed = int(seed_override) if seed_override is not None else int(meta["page_seed"])
        chapter_id = str((options or {}).get("chapter_id", "default"))

        out_image_path = out_dir / f"page_{page_num:03d}_colorized.png"
        out_meta_path  = out_dir / f"page_{page_num:03d}_colorized.runmeta.json"

        runmeta_base: dict[str, Any] = {
            "meta_source":     meta_source,
            "engine":          self.engine.__class__.__name__,
            "seed":            seed,
            "strength":        float(strength),
            "status":          "success",
            "page_num":        page_num,
            "input_image":     str(meta["page_image"]),
            "style_reference": str(meta["style_reference"]),
            "text_mask":       str(meta["text_mask"]),
            "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
            "options":         options or {},
        }

        from core.generation.orchestrator import Pass2Orchestrator

        # Quando roda via SQLite o Pass2Orchestrator não precisa carregar JSON —
        # o metadata já é injetado em orchestrator.metadata logo abaixo.
        is_sqlite_mode = meta_source.startswith("sqlite")
        effective_meta_path = "" if is_sqlite_mode else meta_source

        orchestrator = Pass2Orchestrator(
            meta_json_path=effective_meta_path,
            masks_dir=str(out_dir),
            style_ref_path=str(_path_from_meta(meta["style_reference"])),
        )
        orchestrator.metadata = meta
        orchestrator.runtime_options = options or {}

        # Redundância de segurança: injeta o registry serializado pelo Pass1 via
        # SQLite se o orchestrator não o capturou ao analisar a style_ref (ex.: VLM
        # estava offline neste momento, mas estava ativo durante o Pass1).
        if orchestrator.vlm_character_registry is None:
            registry_from_meta = meta.get("vlm_character_registry")
            if registry_from_meta:
                orchestrator.vlm_character_registry = registry_from_meta
                print(
                    f"[Pass2Generator] Registry VLM restaurado do metadata "
                    f"({len(registry_from_meta)} personagem(ns)). Caminho 1 ativado."
                )

        payload = orchestrator.prepare_generation_payload()
        faiss_service = getattr(orchestrator, "faiss_service", None)

        return Pass2PreparedPayload(
            payload=payload,
            seed=seed,
            strength=float(strength),
            out_image_path=out_image_path,
            out_meta_path=out_meta_path,
            meta=meta,
            page_num=page_num,
            chapter_id=chapter_id,
            meta_source=meta_source,
            options=options or {},
            runmeta_base=runmeta_base,
            faiss_service=faiss_service,
            t0=t0,
        )

    def _execute_payload(
        self,
        prepared: Pass2PreparedPayload,
        stop_vlm: bool,
        debug_dump_json: bool,
    ) -> str:
        """
        Aciona a engine de geração com o payload pré-computado.

        stop_vlm=True  → usado pelo single-page (_process_meta): para o servidor
                          Gemma antes de chamar o KSampler para liberar 100% de VRAM.
        stop_vlm=False → usado pelo batch (execute_prepared): o caller já descarregou
                          o Gemma uma única vez antes desta chamada.
        """
        runmeta = dict(prepared.runmeta_base)

        try:
            # ── Descarrega o Gemma (single-page) ───────────────────────────────
            if stop_vlm:
                from config.settings import VLM_PROVIDER, VLM_UNLOAD_SCENARIO
                if VLM_PROVIDER == "llama-cpp" and VLM_UNLOAD_SCENARIO == 1:
                    try:
                        from core.identity.llama_server_manager import LLAMACppServerManager
                        LLAMACppServerManager.stop_server()
                        print("[Pass2Generator] Gemma descarregado da VRAM. Gerando imagem...")
                    except Exception as e:
                        print(f"[Pass2Generator] Aviso: Falha ao desalocar VLM da GPU: {e}")

            # ── Geração via engine ──────────────────────────────────────────────
            result, run_stats = self.engine.generate(
                payload=prepared.payload,
                seed=prepared.seed,
                strength=prepared.strength,
                options=prepared.options,
            )
            runmeta.update(run_stats)

            from PIL import Image

            result_img = (
                Image.open(result).convert("RGB")
                if isinstance(result, (str, Path))
                else result.convert("RGB")
            )

            # ── Text Preservation Mask ──────────────────────────────────────────
            mask_img      = prepared.payload.get("text_preservation_mask")
            base_img_path = prepared.payload.get("base_image_path")

            if mask_img is not None and base_img_path:
                try:
                    original_img = Image.open(base_img_path).convert("RGB")
                    if original_img.size != result_img.size:
                        original_img = original_img.resize(result_img.size, Image.Resampling.LANCZOS)
                    if mask_img.size != result_img.size:
                        mask_img = mask_img.resize(result_img.size, Image.Resampling.LANCZOS)
                    result_img = Image.composite(original_img, result_img, mask_img)
                    runmeta["text_preservation_applied"] = True
                except Exception as e:
                    runmeta["text_preservation_applied"] = False
                    runmeta["text_preservation_error"] = str(e)

            result_img.save(prepared.out_image_path)
            runmeta["output_image"] = str(prepared.out_image_path)

            # ── Validation Guards (Fases C & C.5) ──────────────────────────────
            try:
                from core.validation.structure_guard import StructureGuard
                from core.validation.color_guard import ColorGuard

                struct_guard  = StructureGuard()
                struct_result = struct_guard.validate_page(
                    orig_path=str(_path_from_meta(prepared.meta["page_image"])),
                    color_path=str(prepared.out_image_path),
                    output_dir=str(prepared.out_image_path.parent),
                )
                runmeta["validation_structure"] = struct_result
                print(
                    f"[Pass2Generator] Validação Estrutural "
                    f"(Dice Score: {struct_result.get('dice_score', 0.0):.3f}) "
                    f"-> Verdict: {struct_result.get('verdict')}"
                )

                color_guard  = ColorGuard()
                color_result = color_guard.validate_page_colors(
                    colorized_image_path=str(prepared.out_image_path),
                    page_metadata=prepared.meta,
                    faiss_service_instance=prepared.faiss_service,
                )
                runmeta["validation_color"] = color_result
                print(
                    f"[Pass2Generator] Validação Cromática "
                    f"(Personagens Validados: {color_result.get('validated_characters_count', 0)}) "
                    f"-> Verdict: {color_result.get('verdict')}"
                )
            except Exception as val_exc:
                print(f"[Pass2Generator] Aviso: Falha ao executar guards de validação: {val_exc}")

        except Exception as exc:
            runmeta["status"]      = "failed"
            runmeta["error"]       = str(exc)
            runmeta["duration_ms"] = int((time.perf_counter() - prepared.t0) * 1000)
            atomic_write_json(prepared.out_meta_path, runmeta, ensure_ascii=False, indent=2)
            if self.state_db_path:
                PipelineStateStore(self.state_db_path).upsert(
                    chapter_id=prepared.chapter_id,
                    page_num=prepared.page_num,
                    stage="pass2",
                    status="failed",
                    metadata=runmeta,
                )
            raise

        runmeta["duration_ms"] = int((time.perf_counter() - prepared.t0) * 1000)
        atomic_write_json(prepared.out_meta_path, runmeta, ensure_ascii=False, indent=2)

        if self.state_db_path:
            PipelineStateStore(self.state_db_path).upsert(
                chapter_id=prepared.chapter_id,
                page_num=prepared.page_num,
                stage="pass2",
                status=str(runmeta.get("status", "success")),
                metadata=runmeta,
            )

        return str(prepared.out_image_path)
