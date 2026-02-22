# Analysis and adaptation of the `/manga` (FAISS) flow for the current Manga-Flux

## Context

In the `/manga` project, FAISS is usually used to retrieve reference embeddings (identity/character/palette) in continuity flows between pages/chapters.

In the current state of Manga-Flux:

- the Two-Pass pipeline is already functional (Pass1 + Pass2)
- an embeddings directory already exists (`data/embeddings`)
- the local API now supports page ingestion from the browser and full chapter execution

## Adaptation implemented in this stage

- New endpoint: `POST /v1/pipeline/run_chapter`
  - receives `manga_id`, `chapter_id`, `style_reference_url`, `page_urls[]`
  - downloads web images
  - runs Pass1 + Pass2 per page
  - saves to: `output/<manga_id>/chapters/<chapter_id>/...`
- Extension updated to:
  - capture image URLs from the current tab
  - send batch via API for chapter processing

## How this approaches the `/manga` flow

- **Chapter semantics**: structured output by manga/chapter
- **Web-first ingestion**: extension collects images from the browser for the pipeline
- **Vector retrieval foundation**: artifacts per chapter are organized for future indexing/retrieval

## Next FAISS adaptation (objective proposal)

1. Create `core/identity/faiss_service.py` module with numpy fallback when FAISS is unavailable.
2. Generate embedding per page/character at the end of Pass1 and persist in the chapter.
3. Expose endpoints:
   - `POST /v1/faiss/index_chapter`
   - `POST /v1/faiss/search`
4. Integrate search result into Pass2 `options` to reinforce visual consistency between pages.

## Risk and mitigation

- **Risk**: FAISS unavailable in the environment.
- **Mitigation**: memory fallback with cosine distance (numpy), maintaining API contract.
