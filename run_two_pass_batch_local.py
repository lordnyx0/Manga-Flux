import argparse
from pathlib import Path

from core.analysis.pass1_pipeline import run_pass1_with_report
from core.generation.engines.dummy_engine import DummyEngine
from core.generation.engines.flux_engine import FluxEngine
from core.generation.pipeline import Pass2Generator

VALID_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def discover_pages(input_dir: Path) -> list[Path]:
    pages = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT]
    pages.sort()
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Execução batch Pass1->Pass2 local (Manga-Flux)")
    parser.add_argument("--input-dir", required=True, help="Diretório com páginas P&B")
    parser.add_argument("--style-reference", required=True, help="Imagem de referência de estilo")
    parser.add_argument("--metadata-output", default="metadata", help="Diretório de saída dos .meta.json")
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

    engine = FluxEngine() if args.engine == "flux" else DummyEngine()
    pass2 = Pass2Generator(engine)

    print(f"[INFO] Encontradas {len(pages)} páginas em {input_dir}")

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
        )

        p2_image = pass2.process_page(str(p1.metadata_path), str(pass2_output))

        line = (
            f"[OK] page={idx:03d} mode={p1.mode} meta={p1.metadata_path} "
            f"runmeta={p1.runmeta_path} p2={p2_image}"
        )
        if p1.fallback_reason:
            line += f" reason={p1.fallback_reason}"
        print(line)


if __name__ == "__main__":
    main()
