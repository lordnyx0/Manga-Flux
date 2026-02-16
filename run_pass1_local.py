import argparse
import logging

from core.analysis.pass1_pipeline import run_pass1_with_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pass1Local")


def main():
    parser = argparse.ArgumentParser(description="Execução Local do Passo 1 (Manga-Flux)")
    parser.add_argument("--page-image", required=True, help="Imagem de entrada P&B da página")
    parser.add_argument("--output-mask", required=True, help="Máscara de texto de saída")
    parser.add_argument("--style-reference", required=True, help="Imagem de referência de estilo")
    parser.add_argument("--page-num", type=int, required=True, help="Número da página")
    parser.add_argument("--prompt", default="manga page colorization", help="Prompt base da página")
    parser.add_argument("--chapter-id", default="default", help="ID do capítulo para seed determinística")
    parser.add_argument("--metadata-output", default="metadata", help="Diretório de saída dos .meta.json")

    args = parser.parse_args()

    try:
        report = run_pass1_with_report(
            page_image=args.page_image,
            style_reference=args.style_reference,
            output_mask=args.output_mask,
            output_metadata_dir=args.metadata_output,
            page_num=args.page_num,
            page_prompt=args.prompt,
            chapter_id=args.chapter_id,
        )
        print(f"\n[SUCESSO] Metadata Pass1 exportada em: {report.metadata_path}")
        print(f"[INFO] Modo de execução Pass1: {report.mode}")
        if report.fallback_reason:
            print(f"[INFO] Motivo fallback: {report.fallback_reason}")
        print(f"[INFO] Dependências detectadas: {report.dependencies}")
        print(f"[INFO] Duração Pass1: {report.duration_ms} ms")
        if report.runmeta_path:
            print(f"[INFO] Runmeta Pass1: {report.runmeta_path}")
    except Exception as exc:
        logger.error("Falha no Pass1: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
