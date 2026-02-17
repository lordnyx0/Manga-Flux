import argparse
import logging

from core.generation.engines.dummy_engine import DummyEngine
from core.generation.engines.flux_engine import FluxEngine
from core.generation.pipeline import Pass2Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pass2Local")


def main():
    parser = argparse.ArgumentParser(description="Execução Local do Passo 2 (Manga-Flux)")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--meta", help="Caminho para o arquivo .meta.json (modo legado)")
    source_group.add_argument("--page-num", type=int, help="Número da página para carregar do SQLite")

    parser.add_argument("--output", default="outputs/results", help="Diretório de saída")
    parser.add_argument("--engine", choices=["flux", "dummy"], default="flux", help="Motor de colorização")
    parser.add_argument("--strength", type=float, default=1.0, help="Força de transformação do Pass2")
    parser.add_argument("--seed-override", type=int, default=None, help="Sobrescreve o seed do metadata")
    parser.add_argument("--state-db", default="metadata/pipeline_state.db", help="SQLite para estado do pipeline")
    parser.add_argument("--chapter-id", default="default", help="ID do capítulo para indexação no estado")
    parser.add_argument("--debug-dump-json", action="store_true", help="Se ativo, grava .runmeta em disco")

    args = parser.parse_args()
    engine = FluxEngine() if args.engine == "flux" else DummyEngine()
    generator = Pass2Generator(engine, state_db_path=args.state_db)

    try:
        if args.meta:
            save_path = generator.process_page(
                args.meta,
                args.output,
                strength=args.strength,
                seed_override=args.seed_override,
                options={"chapter_id": args.chapter_id},
                debug_dump_json=args.debug_dump_json,
            )
        else:
            save_path = generator.process_page_from_state(
                chapter_id=args.chapter_id,
                page_num=args.page_num,
                output_dir=args.output,
                strength=args.strength,
                seed_override=args.seed_override,
                options={"chapter_id": args.chapter_id},
                debug_dump_json=args.debug_dump_json,
            )
        print(f"\n[SUCESSO] Página colorizada salva em: {save_path}")
    except Exception as e:
        logger.error("Falha na geração: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
