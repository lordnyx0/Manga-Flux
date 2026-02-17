import argparse
import json
from pathlib import Path

from core.analysis.pass1_contract import deterministic_seed
from core.analysis.pass1_pipeline import run_pass1_with_report
from core.generation.engines.dummy_engine import DummyEngine
from core.generation.engines.flux_engine import FluxEngine
from core.generation.pipeline import Pass2Generator

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Execução batch Pass1->Pass2 local (Manga-Flux)")
    parser.add_argument("--input-dir", required=True, help="Diretório com páginas P&B")
    parser.add_argument("--style-reference", required=True, help="Imagem de referência de estilo")
    parser.add_argument("--metadata-output", default="metadata", help="Diretório de saída dos .meta.json")
    parser.add_argument("--state-db", default="metadata/pipeline_state.db", help="SQLite para estado do pipeline")
    parser.add_argument("--masks-output", default="outputs/pass1/masks", help="Diretório de saída das máscaras")
    parser.add_argument("--pass2-output", default="outputs/pass2", help="Diretório de saída do Pass2")
    parser.add_argument("--chapter-id", default="default", help="ID do capítulo para seed determinística")
    parser.add_argument("--start-page", type=int, default=1, help="Número inicial da paginação")
    parser.add_argument(
        "--prompt-template",
        default="manga page colorization page={page_num}",
        help="Template de prompt (usa {page_num} e {filename})",
    )
    parser.add_argument("--engine", choices=["flux", "dummy"], default="flux", help="Engine do Pass2")
    parser.add_argument("--pass2-strength", type=float, default=1.0, help="Força usada no Pass2")
    parser.add_argument(
        "--pass2-seed-offset",
        type=int,
        default=0,
        help="Offset somado ao seed determinístico da página",
    )
    parser.add_argument(
        "--pass2-option",
        action="append",
        default=[],
        help="Opções extras no formato chave=valor (pode repetir)",
    )
    parser.add_argument("--debug-dump-json", action="store_true", help="Se ativo, grava .meta/.runmeta em disco")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metadata_output = Path(args.metadata_output)
    masks_output = Path(args.masks_output)
    pass2_output = Path(args.pass2_output)
    metadata_output.mkdir(parents=True, exist_ok=True)
    masks_output.mkdir(parents=True, exist_ok=True)
    pass2_output.mkdir(parents=True, exist_ok=True)

    pages = discover_pages(input_dir)
    if not pages:
        raise SystemExit(f"Nenhuma imagem encontrada em {input_dir}")

    pass2_options = parse_options(args.pass2_option)
    pass2_options["chapter_id"] = args.chapter_id

    engine = FluxEngine() if args.engine == "flux" else DummyEngine()
    pass2 = Pass2Generator(engine, state_db_path=args.state_db)

    print(f"[INFO] Encontradas {len(pages)} páginas em {input_dir}")

    summary: list[dict] = []

    for idx, page in enumerate(pages, start=args.start_page):
        mask_path = masks_output / f"page_{idx:03d}_text.png"
        prompt = args.prompt_template.format(page_num=idx, filename=page.name)

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

        seed_override = None
        if args.pass2_seed_offset != 0:
            seed_override = deterministic_seed(args.chapter_id, idx) + args.pass2_seed_offset

        p2_image = pass2.process_page_from_state(
            chapter_id=args.chapter_id,
            page_num=idx,
            output_dir=str(pass2_output),
            strength=args.pass2_strength,
            seed_override=seed_override,
            options=pass2_options,
            debug_dump_json=args.debug_dump_json,
        )

        line = (
            f"[OK] page={idx:03d} mode={p1.mode} meta={p1.metadata_path} "
            f"runmeta={p1.runmeta_path} p2={p2_image} strength={args.pass2_strength}"
        )
        if seed_override is not None:
            line += f" seed_override={seed_override}"
        if p1.fallback_reason:
            line += f" reason={p1.fallback_reason}"
        print(line)

        summary.append(
            {
                "page_num": idx,
                "input_page": str(page),
                "pass1_mode": p1.mode,
                "pass1_fallback_reason": p1.fallback_reason,
                "pass1_meta": str(p1.metadata_path),
                "pass1_runmeta": str(p1.runmeta_path),
                "pass2_image": p2_image,
                "pass2_strength": args.pass2_strength,
                "pass2_seed_override": seed_override,
                "pass2_options": pass2_options,
            }
        )

    summary_path = pass2_output / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Resumo batch salvo em {summary_path}")


if __name__ == "__main__":
    main()
