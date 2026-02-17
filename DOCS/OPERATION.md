# Manga-Flux — Operação local

Este guia descreve o fluxo operacional mínimo para execução local do pipeline Two-Pass.

## 1) Verificar ambiente

```bash
python scripts/pass1_dependency_report.py
```

## 2) Rodar batch Pass1 -> Pass2

```bash
python run_two_pass_batch_local.py \
  --input-dir data/pages_bw \
  --style-reference data/dummy_manga_test.png \
  --metadata-output metadata \
  --masks-output outputs/pass1/masks \
  --pass2-output outputs/pass2 \
  --chapter-id chapter_001 \
  --engine dummy \
  --pass2-strength 1.0 \
  --pass2-seed-offset 0 \
  --pass2-option sampler=euler \
  --pass2-option notes=smoke_local
```

## 3) Validar contrato e artefatos

```bash
python scripts/validate_two_pass_outputs.py \
  --metadata-dir metadata \
  --pass2-dir outputs/pass2 \
  --expected-pages 3 \
  --require-batch-summary
```

## Artefatos relevantes

- Metadata do Pass1: `metadata/page_{NNN}.meta.json`
- Runmeta do Pass1: `metadata/page_{NNN}.meta.pass1.runmeta.json`
- Saída de imagem do Pass2: `outputs/pass2/page_{NNN}_colorized.png`
- Runmeta do Pass2: `outputs/pass2/page_{NNN}_colorized.runmeta.json`
- Resumo de lote: `outputs/pass2/batch_summary.json`


## 4) Subir API local (opcional)

```bash
python api/server.py --host 127.0.0.1 --port 8765
```

## 5) Companion extension (opcional)

Ver guia:

- `DOCS/API_EXTENSION.md`


## 6) Executar API batch (opcional)

```bash
curl -sS -X POST http://127.0.0.1:8765/v1/pass2/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "metadata_dir":"metadata",
    "output_dir":"outputs/api/pass2",
    "engine":"dummy",
    "strength":1.0,
    "expected_pages":3
  }'
```
