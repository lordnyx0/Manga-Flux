# Local API and Extension (Manga-Flux)

> Note: the API was already under development. This document consolidates the current state and operational endpoints.

## Checklist (API + extension)

### API

- [x] `GET /health` endpoint implemented
- [x] `POST /v1/pass2/run` endpoint implemented
- [x] `POST /v1/pass2/batch` endpoint implemented
- [x] `POST /v1/pipeline/run_chapter` endpoint implemented (ingestion via URLs **and local image upload**)
- [x] Payload validation (`engine`, `strength`, `seed_override`, `options`)
- [x] Cross-platform metadata compatibility (paths `\` / `/`)
- [x] Optional local authentication (token via `MANGA_FLUX_API_TOKEN` + `X-API-Token` header)

### Extension

- [x] MV3 Popup created
- [x] Configuration persistence in `chrome.storage.local`
- [x] Health-check for `GET /health`
- [x] Form for `POST /v1/pass2/run`
- [x] Form for `POST /v1/pass2/batch`
- [x] Display of local execution history (last 20 events)
- [x] Current tab image capture to send to pipeline
- [x] Local PC image upload to send to pipeline
- [x] Light/dark theme (with auto mode)
- [x] Thumbnail view of captured images
- [x] Individual thumbnail removal
- [x] State persistence for use even with minimized/closed popup

## Local API

Start server:

```bash
python api/server.py --host 127.0.0.1 --port 8765
```

With token authentication (optional):

```bash
MANGA_FLUX_API_TOKEN=your_token python api/server.py --host 127.0.0.1 --port 8765
```

### Endpoints

- `GET /health`
  - basic service status
  - informs if token is enabled (`token_required`)

- `POST /v1/pass2/run`
  - executes Pass2 for a single page

- `POST /v1/pass2/batch`
  - executes Pass2 for all `page_*.meta.json` in `metadata_dir`

- `POST /v1/pipeline/run_chapter`
  - End-to-end Pass1+Pass2 flow from image URLs or local upload (base64)
  - supports style reference by URL **or upload (base64)**
  - saves to `output/<manga_id>/chapters/<chapter_id>/...`

Example body for `run_chapter`:

```json
{
  "manga_id": "my_manga",
  "chapter_id": "chapter_001",
  "style_reference_url": "https://.../style.png",
  "style_reference_base64": "<optional base64 sent by extension>",
  "style_reference_filename": "style.png",
  "page_urls": [
    "https://.../page1.png"
  ],
  "page_uploads": [
    {"filename": "page2.png", "content_base64": "<local image base64>"}
  ],
  "engine": "dummy",
  "strength": 1.0
}
```

## Chrome Extension (Companion)

Extension folder:

- `extension/manga-flux-extension`

Load in Chrome:

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select `extension/manga-flux-extension`

The popup allows you to:

- save API URL/token
- run health-check (`GET /health`)
- run single-page Pass2 (`POST /v1/pass2/run`)
- run batch Pass2 (`POST /v1/pass2/batch`)
- run chapter pipeline (`POST /v1/pipeline/run_chapter`)
- view local history of recent executions
- switch between light/dark/auto theme
- view thumbnails of captured/local uploaded images and remove individually
- keep configuration/state after minimizing/closing popup

## FAISS Analysis

- See analysis and adaptation plan: `DOCS/FAISS_ADAPTACAO_MANGA_FLUX.md`

## Consolidated status in recovery plan

The API/extension items above were also reflected in the main recovery checklist:

- `RECUPERACAO_FUNCIONAL_MINIMA.md`
- `VIABILIDADE.md`
