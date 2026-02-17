# Recuperação do Pass1 baseada em `/manga`

A recuperação do Pass1 está sendo feita por **adaptação interna** para o Manga-Flux, usando `/manga` como fonte de código.

## Diretriz aplicada

- `/manga` é **base de código** para migração.
- Não existe ligação em runtime entre repositórios.
- O Pass1 vive no próprio Manga-Flux (`core/analysis`).

## Progresso desta etapa

Arquivos do Pass1 portados da base histórica para dentro do projeto:

- `core/pass1_analyzer.py`
- `core/analysis/mask_processor.py`
- `core/analysis/segmentation.py`
- `core/analysis/z_buffer.py`
- `core/detection/yolo_detector.py`
- `core/detection/nms_custom.py`
- `core/constants.py`
- `core/utils/image_ops.py`
- `core/logging/setup.py`

Além disso, o orquestrador interno de Pass1 foi mantido em `core/analysis/pass1_pipeline.py`, com fallback quando dependências pesadas não estão disponíveis no ambiente.

## Checklist

- [x] Remover estratégia de adapter para repositório legado em runtime.
- [x] Centralizar Pass1 em módulo interno (`core/analysis/pass1_pipeline.py`).
- [x] Portar primeiro bloco de módulos do Pass1 da base `/manga` para dentro do Manga-Flux.
- [x] Garantir export de metadata compatível com Pass2.
- [x] Disponibilizar smoke script local (`scripts/pass1_smoke.sh`).
- [x] Portar blocos restantes de detecção/segmentação necessários para execução completa sem fallback.
- [x] Validar lote smoke de 3 páginas com metadata válida (pipeline Pass1->Pass2).
- [x] Validar 3 páginas reais com máscara + metadata válidos (mode=ported_pass1).
- [x] Integrar Pass1->Pass2 em script único de lote (`run_two_pass_batch_local.py`).

## Comando de smoke

```bash
bash scripts/pass1_smoke.sh
```


## Comando de lote smoke (3 páginas)

```bash
bash scripts/recovery_batch_smoke.sh
```


## Execução batch local do Pass1

```bash
python run_pass1_batch_local.py \
  --input-dir data/pages_bw \
  --style-reference data/style_ref.png \
  --metadata-output metadata \
  --masks-output outputs/pass1/masks \
  --chapter-id chapter_001
```

## Acesso local ao código-base `/manga`

Quando o clone do GitHub estiver bloqueado, gere uma cópia local em `/workspace/manga` a partir do histórico deste repo:

```bash
bash scripts/bootstrap_manga_source.sh
```

Isso mantém a regra de migração: usar `/manga` como base de código, com implementação final interna no Manga-Flux.


## Relatório de dependências do Pass1

```bash
python scripts/pass1_dependency_report.py
```


## Execução batch integrada (Pass1->Pass2)

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


## Validação de artefatos Pass1->Pass2

```bash
python scripts/validate_two_pass_outputs.py --metadata-dir metadata --pass2-dir outputs/smoke/pass2 --expected-pages 3
```


- Observabilidade atual: runmeta do Pass1 inclui `mode`, `fallback_reason`, `dependencies`, `duration_ms` e `timestamp_utc`.
