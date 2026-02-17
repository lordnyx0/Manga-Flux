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
- [x] Completar porte do núcleo de detecção/máscara do `/manga` para execução plena sem fallback em ambiente preparado (`ported_pass1` validado em lote local).
- [x] Adicionar smoke test manual de contrato + pipeline (Pass1 e Pass2).
- [x] Validar execução em lote de 3 páginas em smoke local (script `scripts/recovery_batch_smoke.sh`).
- [x] Integrar Pass1->Pass2 em script único de lote (`run_two_pass_batch_local.py`).
- [x] Validar artefatos e contrato em lote com `scripts/validate_two_pass_outputs.py`.
- [ ] Validar execução em lote de 3 páginas com metadata real de Pass1 (dataset real externo ainda não incluído no repositório).

## Critério de saída da Fase A

- Pass1 gera `metadata/page_{NNN}.meta.json` válido.
- Pass2 executa com metadado válido e gera:
  - `page_{NNN}_colorized.png`
  - `page_{NNN}_colorized.runmeta.json`

Status atual: **atingido para smoke de 1 página**, com fallback quando dependências do Pass1 completo não estão disponíveis no ambiente.

## Comandos de validação local (smoke)

```bash
bash scripts/pass1_smoke.sh
python run_pass1_batch_local.py --help
python run_two_pass_batch_local.py --help
python scripts/setup_pass1_runtime.sh
python scripts/pass1_dependency_report.py
bash scripts/bootstrap_manga_source.sh
python run_pass2_local.py \
  --meta metadata/page_001.meta.json \
  --output outputs/recovery_smoke \
  --engine flux
python scripts/validate_two_pass_outputs.py --metadata-dir metadata --pass2-dir outputs/smoke/pass2 --expected-pages 3
```


- Observabilidade atual: runmeta do Pass1 inclui `mode`, `fallback_reason`, `dependencies`, `duration_ms` e `timestamp_utc`.
