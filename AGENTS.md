# Manga-Flux — agent ops

## Pipeline
Two-pass manga colorization. Pass1 = YOLO Manga109 + SAM + embeddings
(`core/`). Pass2 = ComfyUI via API. **Qwen-Image-2.1 is the default engine**
(`--engine qwen`); `flux` is a legacy fallback. Never commit secrets.

## Ports (all local)
- ComfyUI 8188 · VLM Gemma 1234 · VLM Qwen3.5 1235 · Manga-Flux API 8765
- `VLM_MODEL=gemma|qwen3.5` selects the VLM (`config/settings.py`).

## VRAM discipline (RTX 3060 12GB)
ComfyUI (Qwen) + VLM servers do NOT fit together. Stop one before starting
the other (`Stop-Process` by `CommandLine` filter). Text encoder lives in
RAM; run ComfyUI with `--lowvram`.

## Background jobs (Windows rule — see global instructions)
Long jobs ONLY via WMI-detached `cmd /c` with full stdio redirect; monitor
with short log/artifact polls; completion = artifact exists, never log silence.
Never `Start-Process`/`Start-Job`/`&` for background work.

## Key paths
- Engines: `core/generation/engines/{qwen_engine,flux_engine}.py`
- Cast resolver: `core/identity/cast_resolver.py` (+ `marks.json`,
  `dramatis_personae.json` per chapter) · registry: `core/identity/character_registry.py`
- ComfyUI portable: `C:\Users\Nyx\ComfyUI\ComfyUI_windows_portable`
- CPU-only tests: `python -m pytest tests/test_qwen_payload.py tests/test_cast_resolver.py -q`
- Full backend map + session log: `DOCS/QWEN_MIGRATION.md`
