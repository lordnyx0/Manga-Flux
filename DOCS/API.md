# API — Especificação Inicial (Flux)

## Objetivo

Documentar a API-alvo do Manga-Flux para orquestrar:

- submissão de páginas/capítulos para Pass1 e Pass2;
- consulta de status de jobs;
- download de artefatos (`.meta.json`, máscaras, imagens colorizadas, runmeta);
- integração futura com extensão/browser client.

> Estado atual: API HTTP **iniciada em modo bootstrap** com servidor stdlib (`api/server.py`). Nesta etapa, já inclui health/version, OpenAPI básico (`/openapi.json`), criação de job, consulta de status, consulta inicial de artefatos por job, endpoints por capítulo/página, schema de erro com `trace_id` e autenticação opcional por token bearer para escrita.

---


## Marcos já concluídos (início sistemático)

- [x] **Marco 1** — Estrutura inicial da API criada em `api/` com entrypoint `run_api_local.py`.
- [x] **Marco 2** — Endpoints `GET /healthz` e `GET /version` implementados.
- [x] **Marco 3** — Endpoint `POST /v1/jobs/two-pass` iniciado com validação obrigatória de `style_reference`.
- [x] **Marco 4** — Consulta `GET /v1/jobs/{job_id}` com persistência mínima em arquivo JSON.
- [x] **Marco 5** — Endpoint `GET /v1/jobs/{job_id}/artifacts` com listagem inicial de diretórios de artefatos.
- [x] **Marco 6** — Schema de erro padronizado no bootstrap (`code`, `message`, `details`, `trace_id`).
- [x] **Marco 7** — Endpoints iniciais por capítulo/página implementados (listagem de páginas + localização de artefatos).
- [x] **Marco 8** — Autenticação bootstrap para rotas de escrita via `MANGA_FLUX_API_TOKEN` (Bearer token).
- [x] **Marco 9** — Teste de contrato HTTP bootstrap automatizado (`scripts/test_api_bootstrap_contract.py`).
- [x] **Marco 10** — Publicação de OpenAPI bootstrap (`/openapi.json`) e exemplos cURL (`DOCS/API_CURL_EXAMPLES.md`).

---

## Escopo funcional esperado

A API deve cobrir 3 blocos principais:

1. **Ingestão e execução**
   - iniciar execução Two-Pass para uma página ou lote;
   - opcionalmente executar apenas Pass1 ou apenas Pass2;
   - exigir `style_reference` para qualquer execução de Pass2 (sem suporte no-reference nesta versão).

2. **Observabilidade e rastreabilidade**
   - consultar status por job (`queued/running/completed/failed`);
   - expor runmeta do Pass1 e Pass2;
   - expor logs resumidos por etapa.

3. **Entrega de artefatos**
   - listar artefatos por capítulo/página;
   - download direto de imagens, máscaras e metadados.

---

## Proposta de endpoints (v1)

### Saúde e versão

- `GET /healthz`
- `GET /version`
- `GET /openapi.json`

### Jobs Two-Pass

- `POST /v1/jobs/two-pass`
  - cria um job de processamento (single ou batch)
- `GET /v1/jobs/{job_id}`
  - retorna status, progresso e resumo
- `GET /v1/jobs/{job_id}/artifacts`
  - lista artefatos gerados

### Execução por etapa

- `POST /v1/jobs/pass1`
- `POST /v1/jobs/pass2`

### Artefatos por capítulo/página

- `GET /v1/chapters/{chapter_id}/pages`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/metadata`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/runmeta/pass1`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/runmeta/pass2`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/mask`
- `GET /v1/chapters/{chapter_id}/pages/{page_num}/colorized`

---

## Contratos mínimos de payload

### `POST /v1/jobs/two-pass` (request)

`style_reference` deve ser informado e apontar para uma imagem válida.

```json
{
  "chapter_id": "chapter_001",
  "input_mode": "batch",
  "input_dir": "data/pages_bw",
  "style_reference": "data/style_ref.png",
  "engine": "flux",
  "options": {
    "strength": 0.75,
    "seed_mode": "metadata"
  }
}
```

### Resposta de criação de job

```json
{
  "job_id": "job_20260101_120000",
  "status": "queued",
  "created_at": "2026-01-01T12:00:00Z"
}
```

### `GET /v1/jobs/{job_id}` (response)

```json
{
  "job_id": "job_20260101_120000",
  "status": "running",
  "progress": {
    "total_pages": 12,
    "pass1_done": 7,
    "pass2_done": 5
  },
  "last_update": "2026-01-01T12:03:40Z"
}
```

---

## Autenticação (bootstrap)

Quando `MANGA_FLUX_API_TOKEN` estiver definido no ambiente, as rotas de escrita exigem:

- header `Authorization: Bearer <token>`

No estado atual, isso se aplica a:

- `POST /v1/jobs/two-pass`

Se a variável não estiver definida, o bootstrap mantém modo aberto para desenvolvimento local.

---

## Mapeamento com implementação local atual

Mesmo sem API HTTP, o repositório já possui os blocos que serão reaproveitados:

- execução batch integrada: `run_two_pass_batch_local.py`;
- execução local Pass2: `run_pass2_local.py`;
- validação de artefatos: `scripts/validate_two_pass_outputs.py`;
- pipeline de geração e runmeta Pass2: `core/generation/pipeline.py`.

A camada API deve orquestrar esses componentes e encapsular paths/CLI em contratos HTTP estáveis.

Exemplos de chamada cURL em: `DOCS/API_CURL_EXAMPLES.md`.

---

## Checklist de pendências (API)

- [x] Definir stack bootstrap da API e estrutura inicial de pastas (`api/` + `run_api_local.py`).
- [x] Implementar `GET /healthz` e `GET /version`.
- [ ] Evoluir `POST /v1/jobs/two-pass` para fila real de execução (atual: registro em store).
- [x] Validar obrigatoriedade de `style_reference` no contrato HTTP de entrada.
- [x] Implementar status de job com persistência mínima (arquivo JSON; SQLite pendente).
- [ ] Migrar bootstrap HTTP para framework oficial (FastAPI/Flask) com OpenAPI nativa.
- [x] Implementar endpoints bootstrap de artefatos por capítulo/página (com `exists` e `path`).
- [x] Padronizar schema de erro (`code`, `message`, `details`, `trace_id`) no bootstrap.
- [x] Adicionar autenticação (token API) para rotas de escrita no bootstrap (`MANGA_FLUX_API_TOKEN`).
- [x] Adicionar testes de contrato HTTP bootstrap (`scripts/test_api_bootstrap_contract.py`).
- [x] Publicar OpenAPI e exemplos cURL (bootstrap).
