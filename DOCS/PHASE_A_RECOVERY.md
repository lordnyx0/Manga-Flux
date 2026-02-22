# Recovery History (Phase A)

This document consolidates the history of Phase A of the Manga-Flux project, which aimed to restore end-to-end execution of Pass1+Pass2 with a stable contract and a simulation engine (mock engine). Furthermore, it logs the recovery and adaptation of the legacy Pass1 from the `/manga` base.

## 1. Pass1 Recovery based on `/manga`

Pass1 recovery was done by **internal adaptation** to Manga-Flux, using `/manga` historically as the source code base.

### Applied Guideline
- `/manga` served as the **codebase** for the initial migration.
- There is no runtime link between repositories.
- Pass1 lives within Manga-Flux itself (in `core/analysis`).

### Ported Files
Some of the files ported from the historical base to the project:
- `core/pass1_analyzer.py`
- `core/analysis/mask_processor.py`, `segmentation.py`, `z_buffer.py`, `pass1_pipeline.py`
- `core/detection/yolo_detector.py`, `nms_custom.py`
- Utilities and constants.

Dependencies ensure the orchestrator runs autonomously without the legacy base at runtime. The historical command to check dependency state was `python scripts/pass1_dependency_report.py`.

## 2. Minimal Functional Recovery Plan (Phase A)

The goal of this phase was achieved: restore the pipeline execution (Pass1 -> Pass2) mock, ensuring the contract (JSON metadata) between them and local tests, for the feasibility of systemic interface reconstruction.

### Checklist (Completed)
- [x] Remove adapter strategy for legacy repository at runtime.
- [x] Create formal metadata contract (`metadata/README.md`) and validator (`core/utils/meta_validator.py`).
- [x] Centralize Pass1 in the internal module and ensure metadata export.
- [x] Implement interfaces and mocks (`FluxEngine` skeleton, `DummyEngine`).
- [x] Integrate Pass1->Pass2 in a batch script (`run_two_pass_batch_local.py`) and test smoke scripts (`scripts/pass1_smoke.sh`, `scripts/recovery_batch_smoke.sh`).
- [x] Validate 3 real pages with mask + valid metadata (mode=ported_pass1, no fallback).
- [x] Hardening and observability addition (runmeta JSON with `duration_ms`, `timestamp_utc`, seeds and steps control).
- [x] Reinforce Pass2 batch validator (`batch_summary.json`).
- [x] Document operation (`DOCS/OPERATION.md`).
- [x] Start API infrastructure (`api/server.py`) with authentication, `/health`, `/v1/pass2/run`, `/v1/pass2/batch` and `/v1/pipeline/run_chapter`.
- [x] Create and improve Companion Extension (Chrome MV3) to trigger API and perform local and screenshot tests.

### Pending Functionality (Moved to Phase B/C)
- [ ] Integrate FAISS in the chapter flow (indexing + semantic search).

## 3. Exit Criteria (Phase A Status)
- Pass1 generates valid `metadata/page_{NNN}.meta.json` files.
- Pass2 validly consumes the metadata and produces the mocked image + a `runmeta.json` report file.
- Status: **Achieved** with the unified pipeline in scripts and API.
