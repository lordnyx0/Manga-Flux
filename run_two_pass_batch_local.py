"""
run_two_pass_batch_local.py -- Execucao batch Pass1->Pass2 (Manga-Flux)

Fluxo de 3 fases para gestao eficiente de VRAM:

  Fase 1 -- Pass1 (YOLO + SAM + identidade):
    Analisa todas as paginas P&B e grava metadata no SQLite.

  Fase 2 -- Pre-computacao de prompts (Gemma E2B):
    O Gemma processa todas as paginas de uma vez e gera todos os prompts.
    Ao final, o Gemma e descarregado da VRAM UMA UNICA VEZ.

  Fase 3 -- Geracao de imagens (Flux/ComfyUI):
    Com 100% da VRAM livre, o KSampler colore todas as paginas em sequencia.
"""
import argparse
import json
import shutil
from pathlib import Path

from core.analysis.pass1_contract import deterministic_seed
from core.analysis.pass1_pipeline import run_pass1_with_report
from core.generation.engines.dummy_engine import DummyEngine
from core.generation.engines.flux_engine import FluxEngine
<<<<<<< HEAD
from core.generation.pipeline import Pass2Generator, Pass2PreparedPayload
=======
from core.generation.pipeline import Pass2Generator
from core.correction import run_phase_c_structure_check, save_phase_c_artifacts, serialize_phase_c_report
>>>>>>> ceac8ea69dd3e6ce644d6d7e35fdbf7424fbb819

VALID_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def discover_pages(input_dir: Path) -> list[Path]:
    pages = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT]
    pages.sort()
    return pages


def parse_options(raw_options: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for raw in raw_options:
        if "=" not in raw:
            raise SystemExit(f"Opção inválida '{raw}'. Use o formato chave=valor.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Opção inválida '{raw}': chave vazia.")
        options[key] = value
    return options


def _unload_gemma() -> None:
    """Descarrega o llama-server da VRAM, se estiver rodando."""
    try:
        from config.settings import VLM_PROVIDER
        if VLM_PROVIDER == "llama-cpp":
            from core.identity.llama_server_manager import LLAMACppServerManager
            LLAMACppServerManager.stop_server()
            print("[INFO] Gemma descarregado da VRAM. GPU 100% livre para o Flux/KSampler.")
    except Exception as e:
        print(f"[WARN] Falha ao descarregar Gemma: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execucao batch Pass1->Pass2 local (Manga-Flux) -- 3 fases com VRAM otimizada"
    )
    parser.add_argument("--input-dir",        required=True,  help="Diretorio com paginas P&B")
    parser.add_argument("--style-reference",  required=True,  help="Imagem de referencia de estilo")
    parser.add_argument("--metadata-output",  default="metadata",                help="Diretorio de saida dos .meta.json")
    parser.add_argument("--state-db",         default="metadata/pipeline_state.db", help="SQLite para estado do pipeline")
    parser.add_argument("--masks-output",     default="outputs/pass1/masks",    help="Diretorio de saida das mascaras")
    parser.add_argument("--pass2-output",     default="outputs/pass2",          help="Diretorio de saida do Pass2")
    parser.add_argument("--chapter-id",       default="default",                help="ID do capitulo para seed deterministica")
    parser.add_argument("--start-page",       type=int, default=1,              help="Numero inicial da paginacao")
    parser.add_argument(
        "--prompt-template",
        default="manga page colorization page={page_num}",
        help="Template de prompt (usa {page_num} e {filename})",
    )
    parser.add_argument("--engine",            choices=["flux", "dummy"], default="flux", help="Engine do Pass2")
    parser.add_argument("--pass2-strength",    type=float, default=0.85,  help="Denoise do KSampler (0.85 preserva tracos, 0.95 mais cor)")
    parser.add_argument(
        "--pass2-seed-offset",
        type=int,
        default=0,
        help="Offset somado ao seed deterministico da pagina",
    )
    parser.add_argument(
        "--pass2-option",
        action="append",
        default=[],
        help="Opcoes extras no formato chave=valor (pode repetir)",
    )
    parser.add_argument("--debug-dump-json", action="store_true", help="Se ativo, grava .meta/.runmeta em disco")
    parser.add_argument("--phase-c-structure", action="store_true", help="Roda validação estrutural da Fase C após o Pass2")

    args = parser.parse_args()

    input_dir       = Path(args.input_dir)
    metadata_output = Path(args.metadata_output)
    masks_output    = Path(args.masks_output)
    pass2_output    = Path(args.pass2_output)
    metadata_output.mkdir(parents=True, exist_ok=True)
    masks_output.mkdir(parents=True, exist_ok=True)
    pass2_output.mkdir(parents=True, exist_ok=True)

    pages = discover_pages(input_dir)
    if not pages:
        raise SystemExit(f"Nenhuma imagem encontrada em {input_dir}")

    pass2_options = parse_options(args.pass2_option)
    pass2_options["chapter_id"] = args.chapter_id

    engine = FluxEngine() if args.engine == "flux" else DummyEngine()
    pass2  = Pass2Generator(engine, state_db_path=args.state_db)

    print(f"\n[INFO] Encontradas {len(pages)} paginas em {input_dir}")
    print(f"[INFO] Denoise (strength): {args.pass2_strength}")
    print(f"[INFO] Capitulo: {args.chapter_id} | Engine: {args.engine}\n")

    summary: list[dict] = []

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 1 — Pass1: YOLO + SAM + identidade para todas as páginas
    # ══════════════════════════════════════════════════════════════════════════
    print(f"{'='*60}")
    print(f"[Fase 1/3] Analisando {len(pages)} pagina(s) -- YOLO + SAM + identidade")
    print(f"{'='*60}")

    pass1_reports: list[tuple[int, Path, object]] = []

    for idx, page in enumerate(pages, start=args.start_page):
        mask_path = masks_output / f"page_{idx:03d}_text.png"
        prompt    = args.prompt_template.format(page_num=idx, filename=page.name)

        p1 = run_pass1_with_report(
            page_image=str(page),
            style_reference=args.style_reference,
            output_mask=str(mask_path),
            output_metadata_dir=str(metadata_output),
            page_num=idx,
            page_prompt=prompt,
            chapter_id=args.chapter_id,
            state_db_path=args.state_db,
            debug_dump_json=args.debug_dump_json,
        )
        pass1_reports.append((idx, page, p1))
        print(f"  [Pass1] page={idx:03d} mode={p1.mode}"
              + (f" reason={p1.fallback_reason}" if p1.fallback_reason else ""))

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2 — Pré-geração de prompts via Gemma (TODAS as páginas de uma vez)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Fase 2/3] Pre-gerando prompts via Gemma -- {len(pages)} pagina(s)")
    print(f"{'='*60}")

    prepared_batch: list[tuple[int, Path, object, int | None, Pass2PreparedPayload | None]] = []

    for idx, page, p1 in pass1_reports:
        seed_override = None
        if args.pass2_seed_offset != 0:
            seed_override = deterministic_seed(args.chapter_id, idx) + args.pass2_seed_offset

        try:
            prep = pass2.prepare_page_from_state(
                chapter_id=args.chapter_id,
                page_num=idx,
                output_dir=str(pass2_output),
                strength=args.pass2_strength,
                seed_override=seed_override,
                options=pass2_options,
            )
            prompt_len = len(prep.payload.get("prompt", ""))
            print(f"  [Gemma] page={idx:03d} prompt={prompt_len} chars ✓")
            prepared_batch.append((idx, page, p1, seed_override, prep))
        except Exception as e:
            print(f"  [WARN]  page={idx:03d} falha ao pré-gerar payload: {e}")
            prepared_batch.append((idx, page, p1, seed_override, None))

    # Descarrega o Gemma da VRAM UMA ÚNICA VEZ após processar todas as páginas
    print()
    _unload_gemma()

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3 — Geração de imagens (Flux/ComfyUI) — GPU 100% disponível
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Fase 3/3] Gerando {len(prepared_batch)} imagem(ns) via {args.engine.upper()}/ComfyUI")
    print(f"{'='*60}")

    for idx, page, p1, seed_override, prep in prepared_batch:
        if prep is not None:
            # Caminho normal: executa com payload pré-computado (Gemma já descarregado)
            try:
                p2_image = pass2.execute_prepared(prep, debug_dump_json=args.debug_dump_json)
            except Exception as e:
                print(f"  [ERROR] page={idx:03d} falha na geração: {e}")
                p2_image = None
        else:
            # Fallback: payload não foi pré-computado — roda pipeline completo sem Gemma
            print(f"  [WARN]  page={idx:03d} usando fallback (sem pré-computação Gemma)")
            try:
                p2_image = pass2.process_page_from_state(
                    chapter_id=args.chapter_id,
                    page_num=idx,
                    output_dir=str(pass2_output),
                    strength=args.pass2_strength,
                    seed_override=seed_override,
                    options=pass2_options,
                    debug_dump_json=args.debug_dump_json,
                )
            except Exception as e:
                print(f"  [ERROR] page={idx:03d} fallback falhou: {e}")
                p2_image = None



        phase_c_report = None
        phase_c_artifacts = None
        if args.phase_c_structure:
            phase_c_report = run_phase_c_structure_check(
                original_path=str(page),
                colorized_path=p2_image,
            )
            phase_c_artifacts = save_phase_c_artifacts(
                report=phase_c_report,
                output_dir=pass2_output,
                page_num=idx,
                overlay_base_image=p2_image,
            )

        line = (
            f"  [OK]   page={idx:03d} mode={p1.mode} p2={p2_image} "
            f"strength={args.pass2_strength}"
        )
        if seed_override is not None:
            line += f" seed_override={seed_override}"
        if p1.fallback_reason:
            line += f" reason={p1.fallback_reason}"
        if phase_c_report is not None:
            line += (
                f" phase_c_alert={phase_c_report['has_structural_alert']}"
                f" affected={phase_c_report['page_affected_ratio_pct']:.2f}%"
            )
        print(line)

        # Copiar todos os passos para uma pasta isolada de debug
        try:
            debug_dir = Path("outputs/debug_steps") / f"page_{idx:03d}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Copiar imagem BW original
            shutil.copy2(page, debug_dir / "original_bw.png")
            
            # 2. Copiar máscara de texto se existir
            if mask_path.exists():
                shutil.copy2(mask_path, debug_dir / "text_mask.png")
                
            # 3. Copiar lineart extraída se existir
            lineart_path = Path(str(mask_path).replace("masks", "linearts").replace("_text.png", "_lineart.png"))
            if lineart_path.exists():
                shutil.copy2(lineart_path, debug_dir / "extracted_lineart.png")
                
            # 4. Copiar imagem final colorizada
            if p2_image and Path(p2_image).exists():
                shutil.copy2(p2_image, debug_dir / "final_colorized.png")
                
            print(f"  [DEBUG] Todos os passos salvos em: {debug_dir}")
        except Exception as e:
            print(f"  [WARN] Falha ao copiar passos de debug: {e}")

        summary.append(
            {
                "page_num":              idx,
                "input_page":            str(page),
                "pass1_mode":            p1.mode,
                "pass1_fallback_reason": p1.fallback_reason,
                "pass1_meta":            str(p1.metadata_path),
                "pass1_runmeta":         str(p1.runmeta_path),
                "pass2_image":           p2_image,
                "pass2_strength":        args.pass2_strength,
                "pass2_seed_override":   seed_override,
                "pass2_options":         pass2_options,
                "phase_c_structure":     (serialize_phase_c_report(phase_c_report) if phase_c_report else None),
                "phase_c_artifacts":     phase_c_artifacts,
            }
        )

    summary_path = pass2_output / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] Resumo batch salvo em {summary_path}")


if __name__ == "__main__":
    main()
