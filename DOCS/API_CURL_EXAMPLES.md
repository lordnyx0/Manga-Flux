# API Bootstrap — Exemplos cURL

> Base URL local padrão: `http://127.0.0.1:8080`

## Health e versão

```bash
curl -sS http://127.0.0.1:8080/healthz
curl -sS http://127.0.0.1:8080/version
```

## OpenAPI bootstrap

```bash
curl -sS http://127.0.0.1:8080/openapi.json
```

## Criar job Two-Pass (sem auth)

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/jobs/two-pass \
  -H 'Content-Type: application/json' \
  -d '{
    "chapter_id": "chapter_001",
    "style_reference": "data/style_ref.png",
    "metadata_output": "metadata",
    "pass2_output": "outputs/pass2",
    "masks_output": "outputs/pass1/masks"
  }'
```

## Criar job Two-Pass (com auth bearer habilitada)

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/jobs/two-pass \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dev-token' \
  -d '{
    "chapter_id": "chapter_001",
    "style_reference": "data/style_ref.png"
  }'
```

## Consultar job e artefatos

```bash
curl -sS http://127.0.0.1:8080/v1/jobs/<job_id>
curl -sS http://127.0.0.1:8080/v1/jobs/<job_id>/artifacts
```

## Consultar páginas e artefatos por capítulo/página

```bash
curl -sS http://127.0.0.1:8080/v1/chapters/chapter_001/pages
curl -sS http://127.0.0.1:8080/v1/chapters/chapter_001/pages/1/metadata
curl -sS http://127.0.0.1:8080/v1/chapters/chapter_001/pages/1/runmeta/pass1
curl -sS http://127.0.0.1:8080/v1/chapters/chapter_001/pages/1/runmeta/pass2
curl -sS http://127.0.0.1:8080/v1/chapters/chapter_001/pages/1/mask
curl -sS http://127.0.0.1:8080/v1/chapters/chapter_001/pages/1/colorized
```
