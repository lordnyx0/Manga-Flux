# Manga-Flux: The First Specialist Manga Colorization Engine (v1.0)

Manga-Flux é um pipeline de colorização com arquitetura **Two-Pass**:

- **Pass1**: análise, máscara de texto e contrato de metadata.
- **Pass2**: geração usando engine (Flux mock atualmente no bootstrap).

> Estado atual da restauração: Pass1/Pass2 estão operacionais em modo local com fallback, com validação de artefatos em lote.
> Nesta versão, **não há suporte a fluxo sem imagem de referência** (`style_reference` obrigatório no Pass2).

## 🌟 Key Features

- **Flux Specialist Path**: estrutura preparada para engine Flux dedicada.
- **Two-Pass Contract**: `metadata/page_{NNN}.meta.json` validado antes do Pass2.
- **Runmeta por página**:
  - Pass1: `page_{NNN}.meta.pass1.runmeta.json`
  - Pass2: `page_{NNN}_colorized.runmeta.json`
- **Validação automática de artefatos**: script para checagem de contrato e linkage Pass1→Pass2.

## 🛠️ Bootstrap local rápido

### 1) Preparar runtime completo do Pass1

```bash
bash scripts/setup_pass1_runtime.sh
```

### 2) Verificar dependências do Pass1

```bash
python scripts/pass1_dependency_report.py
```

### 3) Executar smoke integrado (3 páginas sintéticas)

```bash
bash scripts/recovery_batch_smoke.sh
```

Esse comando:

1. cria 3 páginas sintéticas a partir de `data/dummy_manga_test.png`;
2. roda Pass1 em lote;
3. roda Pass2 para cada página;
4. valida os artefatos com `scripts/validate_two_pass_outputs.py`.

### 4) Executar batch real local (Pass1->Pass2)

```bash
python run_two_pass_batch_local.py \
  --input-dir data/pages_bw \
  --style-reference data/style_ref.png \
  --metadata-output metadata \
  --masks-output outputs/pass1/masks \
  --pass2-output outputs/pass2 \
  --chapter-id chapter_001 \
  --engine flux
```



### 5) Subir API local (bootstrap da próxima etapa)

```bash
MANGA_FLUX_API_TOKEN=dev-token python run_api_local.py --host 0.0.0.0 --port 8080
```

> Se `MANGA_FLUX_API_TOKEN` for definido, `POST /v1/jobs/two-pass` exige `Authorization: Bearer <token>`.

Endpoints disponíveis no bootstrap:

- `GET /healthz`
- `GET /version`
- `GET /openapi.json`
- `POST /v1/jobs/two-pass` (com `style_reference` obrigatório)
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/artifacts`
- `GET /v1/chapters/{chapter_id}/pages`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/metadata`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/runmeta/pass1`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/runmeta/pass2`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/mask`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/colorized`

Teste de contrato HTTP bootstrap:

```bash
python scripts/test_api_bootstrap_contract.py
```

## 📚 Documentação

- Pass1 recuperação: `PASS1_RECUPERACAO_BASE_MANGA.md`
- Pass2 operacional: `DOCS/PASS2.md`
- API (especificação inicial): `DOCS/API.md`
- API cURL examples: `DOCS/API_CURL_EXAMPLES.md`
- Extensão (especificação inicial): `DOCS/EXTENSAO.md`
- Recuperação funcional mínima: `RECUPERACAO_FUNCIONAL_MINIMA.md`


### 1) Verificar dependências do Pass1

```bash
python scripts/pass1_dependency_report.py
```

### 2) Executar smoke integrado (3 páginas sintéticas)

```bash
bash scripts/recovery_batch_smoke.sh
```

Esse comando:

1. cria 3 páginas sintéticas a partir de `data/dummy_manga_test.png`;
2. roda Pass1 em lote;
3. roda Pass2 para cada página;
4. valida os artefatos com `scripts/validate_two_pass_outputs.py`.

### 3) Executar batch real local (Pass1->Pass2)

```bash
python run_two_pass_batch_local.py \
  --input-dir data/pages_bw \
  --style-reference data/style_ref.png \
  --metadata-output metadata \
  --masks-output outputs/pass1/masks \
  --pass2-output outputs/pass2 \
  --chapter-id chapter_001 \
  --engine flux
```

## 📄 Contrato Pass1→Pass2

Documentação do contrato em:

- `metadata/README.md`

Validador usado pelo Pass2:

- `core/utils/meta_validator.py`
