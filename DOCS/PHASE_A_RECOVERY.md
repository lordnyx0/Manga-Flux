# Histórico de Recuperação (Fase A)

Este documento consolida o histórico da Fase A do projeto Manga-Flux, que teve como objetivo restaurar a execução ponta-a-ponta do Pass1+Pass2 com um contrato estável e um motor de simulação (mock engine). Além disso, registra a recuperação e adaptação do Pass1 legado a partir da base `/manga`.

## 1. Recuperação do Pass1 baseada em `/manga`

A recuperação do Pass1 foi feita por **adaptação interna** para o Manga-Flux, usando `/manga` historicamente como fonte de código.

### Diretriz Aplicada
- `/manga` serviu como **base de código** para a migração inicial.
- Não existe ligação em tempo de runtime entre repositórios.
- O Pass1 vive no próprio Manga-Flux (em `core/analysis`).

### Arquivos Portados
Alguns dos arquivos portados da base histórica para o projeto:
- `core/pass1_analyzer.py`
- `core/analysis/mask_processor.py`, `segmentation.py`, `z_buffer.py`, `pass1_pipeline.py`
- `core/detection/yolo_detector.py`, `nms_custom.py`
- Utilitários e constantes.

As dependências garantem que o orquestrador execute de forma autônoma sem a base legada em runtime. O comando histórico para verificar o estado das dependências era `python scripts/pass1_dependency_report.py`.

## 2. Plano de Recuperação Funcional Mínima (Fase A)

O objetivo desta fase foi alcançado: restaurar a execução do pipeline (Pass1 -> Pass2) mock, garantindo o contrato (JSON metadata) entre eles e os testes locais, para a viabilidade da reconstrução das interfaces sistêmicas.

### Checklist (Concluído)
- [x] Remover estratégia de adapter para repositório legado em runtime.
- [x] Criar contrato formal de metadata (`metadata/README.md`) e validador (`core/utils/meta_validator.py`).
- [x] Centralizar Pass1 no módulo interno e garantir exportação de metadados.
- [x] Implementar interfaces e mocks (`FluxEngine` skeleton, `DummyEngine`).
- [x] Integrar Pass1->Pass2 em script de lote (`run_two_pass_batch_local.py`) e testar smoke scripts (`scripts/pass1_smoke.sh`, `scripts/recovery_batch_smoke.sh`).
- [x] Validar 3 páginas reais com máscara + metadados válidos (mode=ported_pass1, sem fallback).
- [x] Hardening e adição de observabilidade (runmeta JSON com `duration_ms`, `timestamp_utc`, controle de seeds e steps).
- [x] Reforçar validador de lote de Pass2 (`batch_summary.json`).
- [x] Documentar operação (`DOCS/OPERATION.md`).
- [x] Começar infraestrutura da API (`api/server.py`) com autenticação, `/health`, `/v1/pass2/run`, `/v1/pass2/batch` e `/v1/pipeline/run_chapter`.
- [x] Criar e melhorar Companion Extension (Chrome MV3) para acionar API e realizar testes locais e de capturas de tela.

### Funcionalidade Pendente (Transferida para Fase B/C)
- [ ] Integrar FAISS no fluxo de capítulo (indexação + busca semântica).

## 3. Critério de Saída (Status da Fase A)
- O Pass1 gera arquivos `metadata/page_{NNN}.meta.json` válidos.
- O Pass2 consome validamente a metadata e produz a imagem mockada + um arquivo de relatório `runmeta.json`.
- Status: **Atingido** com a pipeline unificada em scripts e API.
