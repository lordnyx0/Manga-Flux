import argparse
import os
import logging
from core.generation.pipeline import Pass2Generator
from core.generation.engines.flux_engine import FluxEngine
from core.generation.engines.dummy_engine import DummyEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pass2Local")

def main():
    parser = argparse.ArgumentParser(description="Execução Local do Passo 2 (Manga-Flux)")
    parser.add_argument("--meta", required=True, help="Caminho para o arquivo .meta.json")
    parser.add_argument("--output", default="outputs/results", help="Diretório de saída")
    parser.add_argument("--engine", choices=["flux", "dummy"], default="flux", help="Motor de colorização")
    parser.add_argument("--strength", type=float, default=1.0, help="Força de transformação do Pass2")
    parser.add_argument("--seed-override", type=int, default=None, help="Sobrescreve o seed do metadata")

    args = parser.parse_args()
    engine = FluxEngine() if args.engine == "flux" else DummyEngine()
    generator = Pass2Generator(engine)
    
    try:
        save_path = generator.process_page(
            args.meta,
            args.output,
            strength=args.strength,
            seed_override=args.seed_override,
        )
        print(f"\n[SUCESSO] Página colorizada salva em: {save_path}")
    except Exception as e:
        logger.error(f"Falha na geração: {e}")
        exit(1)

if __name__ == "__main__":
    main()
