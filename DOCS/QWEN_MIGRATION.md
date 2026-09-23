# Migração Qwen-Image-2.1-Uncensored-GGUF (experimental)

Flux continua o default (`--engine flux`). Qwen é opt-in (`--engine qwen`).
Registro da sessão de 2026-09-21 (RTX 3060 12GB + 32GB RAM).

## 1. Backend ComfyUI (feito)

- ComfyUI portable atualizado `v0.34.0 (12d5279)` → `master b0f4b7b` (traz
  `TextEncodeQwenImage21` + `QwenImage21Cache`; `ReferenceLatent` segue
  presente, Flux preservado). Rollback: `git checkout --detach 12d5279`.
- Custom node `city96/ComfyUI-GGUF` → `leejet/ComfyUI-GGUF` (commit
  `f912d5e`, com suporte Qwen-Image 2.1). Backup em
  `C:\Users\Nyx\ComfyUI\backups\ComfyUI-GGUF-city96`. Atenção: diretórios
  backup **dentro** de `custom_nodes/` são carregados e sombreiam o fork
  novo (`Unknown model architecture!`) — manter fora.
- Modelos em `models/`: `diffusion_models/qwen-image-2.1-Q4_K_M.gguf`
  (4.3GB), `text_encoders/qwen3vl_8b_int8_convrot.safetensors` (8.7GB, fica
  na RAM), `vae/qwen_image_2.1_vae_bf16.safetensors` (644MB). Nunca `Q8_0`
  (shape mismatch, bug do autor).
- Dependência nova do master no Python embarcado: `comfy_aimdo==0.5.5`
  (a 0.4.15 não tem `comfy_aimdo.storage`).
- Script: `scripts/setup_qwen_comfy_backend.ps1` (`-CheckOnly` para auditar).

## 2. Engine Qwen + smoke test (medido, artefatos em `outputs/`)

- `core/generation/engines/qwen_engine.py` — Image-Edit via
  `TextEncodeQwenImage21` (`image_1`=P&B, `image_2`=style, `image_3..N`=crops,
  até 10), `cfg=1.0`, `denoise=1.0`, `SaveImage="19"`, `QwenImage21Cache`.
  Sem `ReferenceLatent`/LoRA. `strength` é ignorado (avisa). `ref_crops_dir`
  injeta PNGs manuais (teste A/B).
- `qwen_api_workflow.json` (referência), `configs/qwen.yaml`,
  `tests/test_qwen_payload.py` (6 testes CPU-only, passando).
- CLIs + API aceitam `--engine qwen` (`run_pass2_local.py`,
  `run_two_pass_batch_local.py`, `api/server.py`). Resolvido o conflito de
  merge nos imports do batch (mantidas as duas metades).
- `orchestrator.prepare_qwen_edit_payload()` determinístico (sem Gemma por
  página) + `pipeline.py` roteando por engine (`use_gemma_prompts=1` volta
  ao modular legado).
- **Smoke real** (página screenshot 699×935 → 896×1184):
  `qwen_smoke_10steps.png` 89.6s frio (com carga); `qwen_full_25steps.png`
  ~100s quente (~2.5–4s/step — bem abaixo dos 5–9min estimados).
  Traço e texto intactos; **céu azul inventado** nos fundos vazios
  (horror-vacui) — caso para prompt (`preserve empty white backgrounds`),
  não para mais steps.
- **StructureGuard invalidado para Qwen:** self-test (original vs original)
  dá `dice=0.496, CRITICAL_FAILURE, 0 painéis` — a extração assimétrica de
  bordas + threshold 0.75 foram calibrados pro Flux+ReferenceLatent.
  Recalibrar (params simétricos, threshold menor, fix em `extract_panels`)
  antes de usar como juiz do A/B.

## 3. Extensão + API (fixes do dia)

- `popup.js`: fallback CORS — `img.nx-toons.xyz` não envia ACAO e o fetch
  in-tab morre com `TypeError: Failed to fetch`. Páginas com falha vão em
  `page_urls` p/ download server-side (sem CORS); se 1 falhar, o capítulo
  todo vai como URL (o servidor concatena urls antes de uploads — misturar
  embaralharia a ordem). Payload agora leva `page_referer` +
  `page_cookie_header` (coletava e nunca enviava).
- Servidor roda da **raiz** (`python api\server.py`); `output_root` vazio
  salva relativo ao CWD (`api/manga_default/...`). Preencher com caminho
  absoluto na extensão.
- Capítulo teste: 25 págs baixadas em
  `api/manga_default/chapters/chapter_001/inputs/` (curadoria: capa colorida
  + 2 coloridas + 6 P&B).

## 4. Consistência de personagens (achados)

- Classificador colorida-vs-P&B por saturação: 9/9.
- **Capa invisível ao YOLO:** 3 personagens enormes, 0 detecções em qualquer
  threshold/cor/crop-full. Só tile central acha (`body 0.25/face 0.59`).
  RetinaFace (InsightFace) também 0. Índice FAISS da capa = vazio.
- **Matriz CLIP corpo** (ref=page_006): recall 100%, discriminação nula
  (pág 011 inteira → mesmo ref, banda 0.6–0.8, threshold 0.35 passa tudo).
- **Só-rosto:** ArcFace 0% (RetinaFace não vê anime — híbrido opera como
  `clip_only` silencioso); CLIP-em-rosto melhor distribuído mas margens
  0.01–0.11, sem grau de decisão.
- Conclusão: detecção ok, re-identificação fraca → decisão foi pro VLM.

## 5. Resolvedor de elenco chapter-wide (novo, `core/identity/cast_resolver.py`)

- Set-of-marks numerados + downscale 768px, chamada única (teto 12 págs),
  `dramatis_personae.json` cacheável, VLM nunca escreve prompt (só JSON).
- Validação de cobertura exata (cada marca 1 vez; página sem marca = zero;
  retry com o erro realimentado), `top_why_freq` anti-colapso, raw + trail
  salvos (`.attemptN.raw.txt` / `.think.txt`).
- **Gemma 4 E2B 32k (`GEMMA_CTX`, `-c 32768`, +2.8GB VRAM):**
  sem âncora = 5 IDs, pág-3 1/4; com âncora = colapso em 2 IDs
  (`top_why_freq` 0.68); **ablação B de prompt** (temp 0.4, anti-colapso,
  calibração, 2 evidências) = 6 IDs, `top_why_freq` 0.28, pág-3 ~2/4 com
  continuidade temporal citada. Maioria prompt, minoria modelo.
- **Qwen3.5-4B UD-Q5_K_XL + mmproj-F16** (porta 1235, `VLM_MODEL=qwen3.5`,
  ~5.1GB VRAM): thinking de ~72k chars estourou 32k (`finish=length`, JSON
  vazio). Trail salvo mostra elenco plausível (cavaleira/mãe/protagonista/
  pai/tio) com loop de ruminação no fim. Fix pronto: `REASONING_BUDGET=8192`
  (`--reasoning-budget`), parser tolera `<think>`, `max_tokens=32768`.
  **Pendente:** restart com budget + comparativo idêntico + ablação de
  nº de páginas.
- Troubleshooting desta sessão arquivado nos logs mentais acima; rollback
  Flux/Qwen: seção 1 e `VLM_MODEL=gemma`.

## 6. E2E do capítulo (8 págs, `api/manga_default/chapters/chapter_001/`)

- `marks.json` persistido no resolver; `prepare_cast_payload()` monta
  prompt posicional+contrastivo a partir do dramatis Qwen3.5
  (capa `page_002` como `image_2`); `run_chapter_e2e` coloriu 8/8 a
  ~3.7min/pág (`cast_colorized/page_00X_qwen.png`).
- **Colorização seletiva (emergente, não intencional):** só figuras com ID
  ganharam cor; fundos/figurantes ficaram P&B — o template falava do elenco
  mas nunca mandava colorir tudo. Fix: instrução em dois níveis
  (elenco exato + *"every person, animal, object, background and sky gets
  full color"*). Flashback preservado em P&B (convenção correta).
- **Cabelo cinza no noturno (pág-5):** erro do Qwen-Image em geração
  (dramatis e marcas corretos, tudo P3) — screentone de chuva/noite venceu
  a identidade. Fix: cláusula de iluminação no template
  (*shading never changes identity colors*).
- **Pupila roxa (pág-6):** subespecificação nossa — nada na cadeia dizia a
  cor dos olhos. Criado slot `eyes` no ledger; P3 = `#26223E` medido.
- **Bugs do ledger (corrigidos):** bboxes sem reescala p/ saída do Qwen
  (tamanhos variam! → `scale_bbox` + `input_size` obrigatório) e overwrite
  com Nones em escrita concorrente (merge sem-destruir).
- Ledger P3 confirmado c/ hexes reais
  (`hair #4E3D38, eyes #26223E, skin #CFB6AB, clothes #432222`); P4
  provisionado. A/B das págs 4–5 com template novo + hexes em `*_v2.png`.
- `AGENTS.md` no repo + instrução global de background no
  `~/.config/opencode` (jobs longos só via WMI-detached).

## 7. Escopo por painel (anti-bleed) — validado em `page_004`

- v2 (página inteira, todos os IDs): cobertura total mas **sangramento** —
  a cavaleira P2 saiu com rosto/cabelo do Regulus (2 âncoras P3 dominaram).
- v3 (1 geração por painel, 1 ID cada): bleed zerado, mas micro-painel do
  olho (2095×507) **derivou** para dois bustos inventados.
- v4 (painel + margem de contexto 12%, recorte interno recomposto):
  melhor das três — cobertura total, sem bleed, sem deriva.
- **Regras:** 1 ID por geração sempre que separável por frame; micro-painel
  (<15% da página ou <600px) nunca gera solo, vai com margem; 1 crop-âncora
  por ID por página; P2 ancorada via `anchor_P2_knight.png` + hexes no ledger.

## 8. Próximos passos

1. Recalibrar StructureGuard; prompt anti-céu-azul.
2. Fiar Fase 2 do batch no dramatis + ledger + escopo por painel
   (substituir prompt modular por página).
3. Ablação nº de páginas se o thinking voltar a enrolar.

## 7. Rollback

Default segue `flux` / `VLM_MODEL=gemma`. Qwen só com `--engine qwen` /
`"engine": "qwen"` / `VLM_MODEL=qwen3.5`.
