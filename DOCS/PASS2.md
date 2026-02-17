# PASS2 — Documentação Operacional (Flux)

## Objetivo

O Pass2 consome metadados gerados pelo Pass1 (`.meta.json`) e produz:

- imagem final colorizada por página (`page_{NNN}_colorized.png`)
- runmeta de execução (`page_{NNN}_colorized.runmeta.json`)

No estado atual, o caminho principal está em:

- `core/generation/pipeline.py`
- `core/generation/interfaces.py`
- `core/generation/engines/flux_engine.py`

---

## Contrato de entrada

O Pass2 exige arquivo JSON com as chaves:

- `page_num`
- `page_image`
- `page_seed`
- `page_prompt`
- `style_reference`
- `text_mask`

Validação feita por `core/utils/meta_validator.py`.

> **Importante:** nesta versão, `style_reference` é obrigatório. Não há suporte a execução Pass2 sem imagem de referência.

---

## Execução local

### Single page

```bash
python run_pass2_local.py \
  --meta metadata/page_001.meta.json \
  --output outputs/pass2 \
  --engine flux
```

### Batch integrado (Pass1->Pass2)

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

---

## Artefatos de saída

Para cada página `N`:

- `outputs/.../page_{NNN}_colorized.png`
- `outputs/.../page_{NNN}_colorized.runmeta.json`

Campos atuais de `runmeta` do Pass2:

- `meta_source`
- `engine`
- `seed`
- `strength`
- `status`
- `page_num`
- `input_image`
- `style_reference`
- `text_mask`
- `pass1_runmeta`

---

## Checklist de pendências (Pass2)

- [x] Contrato de entrada validado antes da execução.
- [x] Pipeline mínimo Pass2 funcional.
- [x] Runmeta de rastreabilidade por página.
- [x] Integração com fluxo batch local Pass1->Pass2.
- [x] Fluxo com referência explícita (sem modo no-reference nesta versão).
- [ ] Substituir Flux mock por integração real Flux img2img full-frame.
- [ ] Expor parâmetros de geração avançados (sampler, strength por CLI/config).
- [ ] Implementar logging de performance (latência por etapa e uso de memória).
- [ ] Adicionar suite de regressão visual (dataset expected vs generated).
- [ ] Definir e documentar estratégia de fallback/OOM no Pass2 real.
