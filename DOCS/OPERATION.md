# Manga-Flux — Local Operation

This guide describes the minimum operational flow for running the Two-Pass pipeline locally.

## 1) Verify environment

```bash
python scripts/pass1_dependency_report.py
```

## 2) Run Pass1 -> Pass2 batch

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

## 3) Validate contract and artifacts

```bash
python scripts/validate_two_pass_outputs.py \
  --metadata-dir metadata \
  --pass2-dir outputs/pass2 \
  --expected-pages 3 \
  --require-batch-summary
```

## Relevant Artifacts

- Pass1 Metadata: `metadata/page_{NNN}.meta.json`
- Pass1 Runmeta: `metadata/page_{NNN}.meta.pass1.runmeta.json`
- Pass2 Image output: `outputs/pass2/page_{NNN}_colorized.png`
- Pass2 Runmeta: `outputs/pass2/page_{NNN}_colorized.runmeta.json`
- Batch summary: `outputs/pass2/batch_summary.json`


## 4) Start local API (optional)

```bash
python api/server.py --host 127.0.0.1 --port 8765
```

## 5) Companion extension (optional)

See guide:

- `DOCS/API_EXTENSION.md`


## 6) Execute API batch (optional)

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
