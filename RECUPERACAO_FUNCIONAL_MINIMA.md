# Plano de Recuperação Funcional Mínima (Fase A)

Objetivo da fase: restaurar execução ponta-a-ponta do Pass1+Pass2 com contrato estável e engine mock.

## Checklist

- [x] Criar contrato formal de metadata (`metadata/README.md`).
- [x] Implementar validador de contrato (`core/utils/meta_validator.py`).
- [x] Criar interface abstrata de engine (`core/generation/interfaces.py`).
- [x] Implementar `FluxEngine` skeleton mock (`core/generation/engines/flux_engine.py`).
- [x] Implementar engine de teste (`core/generation/engines/dummy_engine.py`).
- [x] Implementar pipeline Pass2 mínimo (`core/generation/pipeline.py`).
- [x] Iniciar recuperação do Pass1 como código interno em `core/analysis/pass1_pipeline.py`.
- [x] Adicionar CLI `run_pass1_local.py` para gerar máscara+metadata no próprio projeto.
- [x] Portar bloco inicial do núcleo de detecção/máscara do `/manga` para `core/analysis`.
- [x] Garantir acesso local ao código-base `/manga` via snapshot (`scripts/bootstrap_manga_source.sh`).
- [x] Completar porte do núcleo de detecção/máscara do `/manga` para execução plena sem fallback (dependências OK no report).
- [x] Adicionar smoke test manual de contrato + pipeline (Pass1 e Pass2).
- [x] Validar execução em lote de 3 páginas em smoke local (script `scripts/recovery_batch_smoke.sh`).
- [x] Integrar Pass1->Pass2 em script único de lote (`run_two_pass_batch_local.py`).
- [x] Validar artefatos e contrato em lote com `scripts/validate_two_pass_outputs.py`.
- [x] Validar execução em lote de 3 páginas com metadata real de Pass1 (dataset real, mode=ported_pass1).
- [x] Hardening inicial do Pass2: runmeta com `duration_ms`/`timestamp_utc`/`options` e CLI com `--strength` + `--seed-override`.
- [x] Expandir batch integrado com controles de Pass2 (`--pass2-strength`, `--pass2-seed-offset`, `--pass2-option`) e `batch_summary.json`.
- [x] Publicar guia operacional (`DOCS/OPERATION.md`) com fluxo executável local.
- [x] Reforçar validador de lote com descoberta de páginas, checagens de consistência (`page_num`, `output_image`) e validação opcional de `batch_summary.json`.
- [x] Iniciar trilha de API local para Pass2 com endpoint `/health` e execução `/v1/pass2/run`.
- [x] Iniciar companion extension (Chrome MV3) para configuração de URL e health-check da API.
- [x] Expandir API local com endpoint batch (`POST /v1/pass2/batch`).
- [x] Expandir companion extension com formulário para executar `POST /v1/pass2/run`.
- [x] Adicionar autenticação opcional (token) na API local.
- [x] Expandir companion extension com formulário para `POST /v1/pass2/batch`.
- [x] Adicionar histórico local na extension (últimos eventos).
- [x] Implementar pipeline de capítulo por URLs (`POST /v1/pipeline/run_chapter`) com saída em `output/<manga_id>/chapters/<chapter_id>/...`.
- [x] Captura de imagens da aba atual na extension para envio ao pipeline.
- [x] Refinar UX da extension para consumidor final (tema claro/escuro, miniaturas e remoção individual).
- [x] Garantir persistência de estado na extension para uso após minimizar/fechar popup.
- [ ] Integrar FAISS no fluxo de capítulo (indexação + busca semântica).

## Critério de saída da Fase A

- Pass1 gera `metadata/page_{NNN}.meta.json` válido.
- Pass2 executa com metadado válido e gera:
  - `page_{NNN}_colorized.png`
  - `page_{NNN}_colorized.runmeta.json`

Status atual: **atingido para 3 páginas reais**, com mode=ported_pass1 (sem fallback).

## Comandos de validação local (smoke)

```bash
bash scripts/pass1_smoke.sh
python run_pass1_batch_local.py --help
python run_two_pass_batch_local.py --help
python scripts/pass1_dependency_report.py
bash scripts/bootstrap_manga_source.sh
python run_pass2_local.py \
  --meta metadata/page_001.meta.json \
  --output outputs/recovery_smoke \
  --engine flux
python scripts/validate_two_pass_outputs.py --metadata-dir metadata --pass2-dir outputs/smoke/pass2 --expected-pages 3 \
  --require-batch-summary
```


- Observabilidade atual: runmeta do Pass1 inclui `mode`, `fallback_reason`, `dependencies`, `duration_ms` e `timestamp_utc`.
