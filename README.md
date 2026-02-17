# Manga-Flux: The First Specialist Manga Colorization Engine (v1.0)

Manga-Flux é um pipeline de colorização com arquitetura **Two-Pass**:

- **Pass1**: análise, máscara de texto e contrato de metadata.
- **Pass2**: geração usando engine (Flux mock atualmente no bootstrap).

> Estado atual da restauração: Pass1/Pass2 estão operacionais em modo local com fallback, com validação de artefatos em lote.

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

## 📄 Contrato Pass1→Pass2

Documentação do contrato em:

- `metadata/README.md`

Validador usado pelo Pass2:

- `core/utils/meta_validator.py`
