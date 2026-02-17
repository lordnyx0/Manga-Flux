# Análise de Viabilidade — Migração para Manga-Flux (Pass1 preservado + Pass2 reescrito)

## 1) Resumo executivo

**Status atual do repositório:** **não operacional** para o fluxo descrito no plano. O repo está em estado de esqueleto incompleto, com forte indício de remoção/perda de código crítico (especialmente `core/`, `tests/` e scripts). O entrypoint principal (`run_pass2_local.py`) referencia módulos inexistentes (`core.generation.*`) e falha imediatamente com `ModuleNotFoundError`.

**Conclusão de viabilidade:** a migração **é viável**, mas **não no estado atual sem recuperação/reconstrução de base**. A abordagem recomendada é tratar o projeto como “bootstrap + reconstrução dirigida por contrato” em 2 fases:
1. **Fase de recuperação funcional mínima** (Pass1→Pass2 contract + interface + FluxEngine mock + pipeline rodando local).
2. **Fase de produção** (integração Flux real, QA visual automatizado + humano, hardening, limpeza de legado).

**Estimativa realista a partir deste estado:**
- **MVP funcional (mock + contrato + execução):** 2–4 dias úteis.
- **Flux real + QA + hardening:** +3–7 dias úteis (dependendo de acesso ao modelo/infra GPU).

---

## 2) Metodologia usada nesta análise

Foi feita inspeção estrutural do repositório e validação básica de execução:

- Inventário de arquivos e diretórios versionados.
- Verificação de branches/tags e histórico recente.
- Inspeção dos arquivos-chave presentes (`README`, entrypoint, configurações, workflow).
- Teste direto de execução do entrypoint para validar integridade mínima.

---

## 3) Diagnóstico objetivo do estado atual

## 3.1 Estrutura encontrada (alto impacto)

- O repositório contém poucos arquivos de código-fonte Python (praticamente `run_pass2_local.py` e `config/settings.py`) e um volume muito grande de artefatos `.pt` em `data/embeddings/`.
- Não há diretório `core/` versionado, embora ele seja importado no entrypoint principal.
- Também não há `tests/` no estado atual, apesar de o workflow de CI depender fortemente desses caminhos.

## 3.2 Entrypoint quebrado

- `run_pass2_local.py` importa:
  - `core.generation.pipeline.Pass2Generator`
  - `core.generation.engines.flux_engine.FluxEngine`
  - `core.generation.engines.dummy_engine.DummyEngine`
- Como `core/` não existe no repositório atual, o script falha antes mesmo de parsear argumentos.

## 3.3 CI inconsistente com o conteúdo do repo

- Workflow `.github/workflows/test.yml` executa `pytest tests/high`, `tests/medium`, `tests/low` e lint em `core/` e `utils/`.
- Esses diretórios não estão presentes no snapshot atual do repositório.
- Resultado: a CI definida não representa o estado real do código e provavelmente quebraria em ambiente limpo.

## 3.4 Evidências de “pass1/pass2” apenas em artefato de saída

- Existe metadata de teste já gerada em `outputs/test_run/metadata/page_001.meta.json` com chaves alinhadas ao contrato planejado (`page_num`, `page_image`, `page_seed`, `page_prompt`, `style_reference`, `text_mask`).
- Porém não existe implementação rastreável no repo atual para gerar isso de forma reproduzível via pipeline completo.

---

## 4) Comparação com o plano inicial (item a item)

Escala de status:
- ✅ **Concluído**
- 🟡 **Parcial / indício**
- ❌ **Não implementado / indisponível no repo**

### 0 — Preparação de branches (`main`, `dev`, feature branches)
- **Status:** ❌
- **Achado:** branch local atual é `work`; não foram identificadas tags de release (`v0.1-flux-skeleton`, `v0.2-flux-integrated`) nem convenção de branches do plano.
- **Impacto:** reduz rastreabilidade e disciplina de integração.

### 1 — Contrato Pass1→Pass2 (`metadata/` + validador)
- **Status:** 🟡
- **Achado:** há exemplo de metadata com chaves corretas em `outputs/test_run/metadata/...`.
- **Lacuna crítica:** não há `metadata/README.md` contratual nem `core/utils/meta_validator.py` presente/operacional no repo.

### 2 — Interface `ColorizationEngine`
- **Status:** ❌
- **Achado:** arquivo `core/generation/interfaces.py` não encontrado.

### 3 — `FluxEngine` skeleton (mock)
- **Status:** ❌
- **Achado:** entrypoint referencia `core/generation/engines/flux_engine.py`, mas arquivo não está no repositório.

### 4 — SD Adapter opcional
- **Status:** ❌
- **Achado:** não identificado `engines/sd_adapter.py`.

### 5 — Integração real do Flux (img2img full-frame com style ref)
- **Status:** ❌
- **Achado:** existe apenas configuração YAML com parâmetros gerais; não há implementação de engine no repo atual.

### 6 — QA visual automático + processo humano
- **Status:** ❌
- **Achado:** não existem `tests/visual/run_batch.sh`, `tests/visual/eval.py` e fluxo QA descrito.

### 7 — Hardening (seed determinística, logs per-page, fallback OOM)
- **Status:** 🟡
- **Achado:** seed determinística operacional no contrato Pass1, runmeta do Pass2 com `duration_ms`/`timestamp_utc`/`options`, resumo por lote (`batch_summary.json`) e validação de consistência reforçada; fallback específico para OOM ainda pendente.

### 8 — Limpeza de legado (arquivar SD/tile RGB fora do caminho crítico)
- **Status:** ❌ (não verificável)
- **Achado:** não há base suficiente no repositório atual para confirmar presença/remoção estruturada de legado.

### 9 — Documentação operacional (`README` + `DOCS/OPERATION.md`)
- **Status:** 🟡
- **Achado:** README segue ativo e já referencia operação; `DOCS/OPERATION.md` foi adicionado com fluxo local executável, mas ainda faltam cenários avançados de produção/GPU.

### 10 — Preparar para Qwen (stub + adapter spec)
- **Status:** ❌
- **Achado:** inexistente no snapshot atual.

---

## 5) Principais riscos (e por que o pass1 “degradou”)

1. **Perda de código-fonte crítico no versionamento**
   - Indício forte: imports para módulos ausentes + CI apontando para estruturas inexistentes.
2. **Repo poluído por artefatos de dados e pobre em código executável**
   - Grande volume de `data/embeddings/*.pt` sem contrapartida de pipeline modular disponível.
3. **Quebra de confiança operacional**
   - README promete capacidades não comprováveis via execução imediata.
4. **Ausência de contrato formal em arquivo canônico**
   - Há metadado exemplo, mas sem validador acoplado no caminho principal.

---

## 6) Viabilidade técnica (objetiva)

**É viável?** Sim.

**Condições para viabilizar rapidamente:**
- Tratar o estado atual como **base incompleta**, não como produto quase-pronto.
- Reconstituir primeiro o **esqueleto mínimo do plano** (Pass1 contract + engine interface + engine mock + pipeline runner).
- Só então plugar Flux real e validar qualidade.

**Dependências externas críticas:**
- Acesso ao modelo Flux Klein 9B (ou endpoint equivalente).
- Ambiente GPU com VRAM suficiente para teste (ideal >=12GB com estratégia de offload).
- Conjunto mínimo de páginas e style refs para QA visual.

---

## 7) Plano de recuperação recomendado (priorizado)

## Fase A — Recuperação funcional mínima (prioridade máxima)

1. **Restaurar árvore base de código**
   - Criar/recuperar: `core/analysis`, `core/generation`, `core/utils`, `scripts`, `tests`.
2. **Implementar contrato Pass1→Pass2 formal**
   - `metadata/README.md` + `core/utils/meta_validator.py`.
3. **Criar interface estável de engine**
   - `core/generation/interfaces.py` (`ColorizationEngine`).
4. **Implementar FluxEngine mock**
   - valida style ref + preserva texto por máscara + I/O consistente.
5. **Reconectar entrypoint**
   - `run_pass2_local.py` funcional com `--meta`, `--output`, `--engine`.

**Gate de saída da Fase A:** comando local roda fim-a-fim com dummy/mock e gera imagem + runmeta.

## Fase B — Integração de produção

6. **Integrar Flux real no engine**
   - img2img full-frame + style ref obrigatória + seed/strength/sampler configuráveis.
7. **QA automatizado + humano**
   - batch visual, métricas (SSIM/LPIPS opcional), CSV de aprovação.
8. **Hardening e observabilidade**
   - seed determinística, logs per-page, fallback OOM.
9. **Higienização de legado**
   - arquivar código antigo e remover caminhos críticos instáveis.
10. **Documentação operacional de verdade**
   - README executável + `DOCS/OPERATION.md`.

---

## 8) Recomendação sobre governança de branches

Para alinhar com seu plano original e evitar nova regressão:

- Reestabelecer imediatamente:
  - `main` estável
  - `dev` integração
  - feature branches curtas por etapa
- Exigir PR pequeno por milestone (contrato, interface, mock, integração real, QA).
- Reativar tags semânticas de progresso (`v0.1-flux-skeleton`, `v0.2-flux-integrated`, etc.).

---

## 9) Parecer final

O projeto **não está pronto** no estado atual e apresenta sinais claros de “apagamento” de partes centrais da arquitetura planejada. Ainda assim, a migração é plenamente **recuperável e viável** se você reintroduzir disciplina de contrato, modularidade por interface e pipeline incremental (mock → real → QA).

Em termos práticos: **não recomendo tentar “consertar por remendo” o estado atual**. Recomendo executar a recuperação por fases acima e tratar cada fase como critério de aceite formal.


---

## 10) Atualização de status (pós-recuperação Fase A)

**Data:** 2026-02-16

A Fase A de recuperação foi **concluída com sucesso**:

- ✅ Árvore base de código restaurada (`core/`, `scripts/`, `config/`)
- ✅ Contrato Pass1→Pass2 implementado (`core/analysis/pass1_contract.py`, `core/utils/meta_validator.py`)
- ✅ Interface estável de engine (`core/generation/interfaces.py`)
- ✅ FluxEngine mock + DummyEngine implementados (`core/generation/engines/`)
- ✅ Entrypoints funcionais:
  - `run_pass1_local.py` (Pass1 standalone)
  - `run_two_pass_batch_local.py` (Pass1→Pass2 integrado)
- ✅ Dependências do Pass1 resolvidas (torch, numpy, PIL, cv2, YOLO, SAM)
- ✅ Execução em lote de 3 páginas reais com `mode=ported_pass1` (sem fallback)
- ✅ Validação contratual passando (`scripts/validate_two_pass_outputs.py`)

**Comandos de validação:**
```bash
# Verificar dependências
python scripts/pass1_dependency_report.py

# Executar lote Pass1→Pass2
python run_two_pass_batch_local.py \
  --input-dir data/pages_bw \
  --style-reference data/dummy_manga_test.png \
  --metadata-output metadata \
  --masks-output outputs/pass1/masks \
  --pass2-output outputs/pass2 \
  --chapter-id test_chapter \
  --engine dummy

# Validar artefatos
python scripts/validate_two_pass_outputs.py \
  --metadata-dir metadata \
  --pass2-dir outputs/pass2 \
  --expected-pages 3
```

**Próximos passos (Fase B):**
- Integrar Flux real no engine
- QA automatizado + processo humano
- Hardening e observabilidade completa


## 11) Atualização incremental (Fase B parcial)

**Data:** 2026-02-17

Avanços incrementais implementados:

- ✅ Pass2 com observabilidade reforçada em runmeta (`duration_ms`, `timestamp_utc`, `options`, `output_image`)
- ✅ CLI local do Pass2 com controles explícitos de geração (`--strength`, `--seed-override`)
- ✅ Batch integrado com parâmetros de Pass2 (`--pass2-strength`, `--pass2-seed-offset`, `--pass2-option`)
- ✅ Geração de resumo por lote (`outputs/pass2/batch_summary.json`)
- ✅ Guia de operação local publicado (`DOCS/OPERATION.md`)
- ✅ Validador de artefatos mais robusto (descoberta dinâmica de páginas, consistência de `output_image` e checagem opcional de `batch_summary.json`)

Pendências para completar Fase B:

- Integrar engine Flux real (inferência de produção)
- Implementar fallback OOM dedicado e telemetria de memória
- Institucionalizar QA visual automatizado + humano


## 12) Atualização incremental (API + extensão)

**Data:** 2026-02-17

Avanços desta iteração:

- ✅ API local mínima implementada (`api/server.py`) com `/health` e `/v1/pass2/run`
- ✅ Companion extension MV3 iniciada (`extension/manga-flux-extension`) para health-check
- ✅ Documentação dedicada adicionada (`DOCS/API_EXTENSION.md`)

Pendências seguintes:

- [x] autenticação local opcional (token)
- [x] endpoint batch na API (`POST /v1/pass2/batch`)
- [x] extensão com formulário para acionar `/v1/pass2/run`
- [x] extensão com formulário para acionar `/v1/pass2/batch`
- [x] histórico local de execuções na extensão
- [x] pipeline de capítulo via API a partir de URLs de páginas
- [x] captura de imagens da aba atual na extensão
- [x] tema claro/escuro e UX de miniaturas com remoção individual
- [x] persistência de estado da extensão para uso após minimizar/fechar popup
- [ ] integração FAISS no fluxo online (index/search)
