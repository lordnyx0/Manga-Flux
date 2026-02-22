# Feasibility Analysis — Migration to Manga-Flux (Pass1 preserved + Pass2 rewritten)

## 1) Executive Summary

**Current repository status:** **not operational** for the flow described in the plan. The repo is in an incomplete skeleton state, with a strong indication of critical code removal/loss (especially `core/`, `tests/` and scripts). The main entrypoint (`run_pass2_local.py`) references nonexistent modules (`core.generation.*`) and fails immediately with `ModuleNotFoundError`.

**Feasibility conclusion:** migration **is feasible**, but **not in the current state without base recovery/reconstruction**. The recommended approach is to treat the project as "bootstrap + contract-driven reconstruction" in 2 phases:
1. **Minimum functional recovery phase** (Pass1→Pass2 contract + interface + FluxEngine mock + local running pipeline).
2. **Production phase** (real Flux integration, automated + human visual QA, hardening, legacy cleanup).

**Realistic estimate from this state:**
- **Functional MVP (mock + contract + execution):** 2–4 business days.
- **Real Flux + QA + hardening:** +3–7 business days (depending on model/GPU infrastructure access).

---

## 2) Methodology used in this analysis

Structural repository inspection and basic execution validation were performed:

- Inventory of versioned files and directories.
- Verification of branches/tags and recent history.
- Inspection of key present files (`README`, entrypoint, configurations, workflow).
- Direct execution test of the entrypoint to validate minimal integrity.

---

## 3) Objective diagnosis of current state

## 3.1 Found structure (high impact)

- The repository contains few Python source code files (practically `run_pass2_local.py` and `config/settings.py`) and a very large volume of `.pt` artifacts in `data/embeddings/`.
- There is no versioned `core/` directory, although it is imported in the main entrypoint.
- There is also no `tests/` in the current state, despite the CI workflow relying heavily on these paths.

## 3.2 Broken entrypoint

- `run_pass2_local.py` imports:
  - `core.generation.pipeline.Pass2Generator`
  - `core.generation.engines.flux_engine.FluxEngine`
  - `core.generation.engines.dummy_engine.DummyEngine`
- Since `core/` does not exist in the current repository, the script fails even before parsing arguments.

## 3.3 CI inconsistent with repo content

- `.github/workflows/test.yml` workflow executes `pytest tests/high`, `tests/medium`, `tests/low` and linting in `core/` and `utils/`.
- These directories are not present in the current snapshot of the repository.
- Result: the defined CI does not represent the real state of the code and would probably break in a clean environment.

## 3.4 Evidence of "pass1/pass2" only in output artifact

- There is test metadata already generated in `outputs/test_run/metadata/page_001.meta.json` with keys aligned to the planned contract (`page_num`, `page_image`, `page_seed`, `page_prompt`, `style_reference`, `text_mask`).
- However, there is no traceable implementation in the current repo to generate this reproducibly via a complete pipeline.

---

## 4) Comparison with the initial plan (item by item)

Status scale:
- ✅ **Completed**
- 🟡 **Partial / indication**
- ❌ **Not implemented / unavailable in repo**

### 0 — Preparation of branches (`main`, `dev`, feature branches)
- **Status:** ❌
- **Finding:** current local branch is `work`; no release tags (`v0.1-flux-skeleton`, `v0.2-flux-integrated`) or branch convention from the plan were identified.
- **Impact:** reduces traceability and integration discipline.

### 1 — Pass1→Pass2 Contract (`metadata/` + validator)
- **Status:** 🟡
- **Finding:** there is an example metadata with correct keys in `outputs/test_run/metadata/...`.
- **Critical gap:** there is no contractual `metadata/README.md` or `core/utils/meta_validator.py` present/operational in the repo.

### 2 — `ColorizationEngine` Interface
- **Status:** ❌
- **Finding:** `core/generation/interfaces.py` file not found.

### 3 — `FluxEngine` skeleton (mock)
- **Status:** ❌
- **Finding:** entrypoint references `core/generation/engines/flux_engine.py`, but file is not in the repository.

### 4 — Optional SD Adapter
- **Status:** ❌
- **Finding:** `engines/sd_adapter.py` not identified.

### 5 — Real Flux Integration (full-frame img2img with style ref)
- **Status:** ❌
- **Finding:** there is only a YAML configuration with general parameters; no engine implementation is in the current repo.

### 6 — Automated + human visual QA
- **Status:** ❌
- **Finding:** `tests/visual/run_batch.sh`, `tests/visual/eval.py` and described QA flow do not exist.

### 7 — Hardening (deterministic seed, per-page logs, OOM fallback)
- **Status:** 🟡
- **Finding:** operational deterministic seed in Pass1 contract, Pass2 runmeta with `duration_ms`/`timestamp_utc`/`options`, batch summary (`batch_summary.json`) and reinforced consistency validation; specific OOM fallback still pending.

### 8 — Legacy cleanup (archive SD/RGB tile outside critical path)
- **Status:** ❌ (unverifiable)
- **Finding:** not enough basis in the current repository to confirm structured presence/removal of legacy.

### 9 — Operational documentation (`README` + `DOCS/OPERATION.md`)
- **Status:** 🟡
- **Finding:** README remains active and already references operation; `DOCS/OPERATION.md` was added with runnable local flow, but advanced production/GPU scenarios are still missing.

### 10 — Prepare for Qwen (stub + adapter spec)
- **Status:** ❌
- **Finding:** nonexistent in current snapshot.

---

## 5) Main risks (and why pass1 "degraded")

1. **Loss of critical source code in versioning**
   - Strong indication: imports for missing modules + CI pointing to nonexistent structures.
2. **Repo polluted by data artifacts and poor in executable code**
   - Large volume of `data/embeddings/*.pt` without counterpart of modular available pipeline.
3. **Breakage of operational trust**
   - README promises capabilities unverifiable via immediate execution.
4. **Absence of formal contract in canonical file**
   - Metadata example exists, but without coupled validator in the main path.

---

## 6) Technical feasibility (objective)

**Is it feasible?** Yes.

**Conditions to quickly make it feasible:**
- Treat the current state as **incomplete base**, not an almost-ready product.
- First reconstruct the **minimal plan skeleton** (Pass1 contract + engine interface + engine mock + pipeline runner).
- Only then plug real Flux and validate quality.

**Critical external dependencies:**
- Access to the Flux Klein 9B model (or equivalent endpoint).
- GPU environment with sufficient VRAM for testing (ideal >=12GB with offload strategy).
- Minimum set of pages and style refs for visual QA.

---

## 7) Recommended recovery plan (prioritized)

## Phase A — Minimal functional recovery (highest priority)

1. **Restore base code tree**
   - Create/recover: `core/analysis`, `core/generation`, `core/utils`, `scripts`, `tests`.
2. **Implement formal Pass1→Pass2 contract**
   - `metadata/README.md` + `core/utils/meta_validator.py`.
3. **Create stable engine interface**
   - `core/generation/interfaces.py` (`ColorizationEngine`).
4. **Implement FluxEngine mock**
   - validates style ref + preserves text by mask + consistent I/O.
5. **Reconnect entrypoint**
   - functional `run_pass2_local.py` with `--meta`, `--output`, `--engine`.

**Phase A exit gate:** local command runs end-to-end with dummy/mock and outputs image + runmeta.

## Phase B — Production integration

6. **Integrate real Flux into engine**
   - full-frame img2img + mandatory style ref + configurable seed/strength/sampler.
7. **Automated + human QA**
   - visual batch, metrics (optional SSIM/LPIPS), approval CSV.
8. **Hardening and observability**
   - deterministic seed, per-page logs, OOM fallback.
9. **Legacy sanitization**
   - archive old code and remove unstable critical paths.
10. **Real operational documentation**
   - Executable README + `DOCS/OPERATION.md`.

---

## 8) Recommendation on branch governance

To align with your original plan and avoid new regression:

- Immediately re-establish:
  - stable `main`
  - integration `dev`
  - short feature branches by stage
- Require small PR per milestone (contract, interface, mock, real integration, QA).
- Reactivate semantic progress tags (`v0.1-flux-skeleton`, `v0.2-flux-integrated`, etc.).

---

## 9) Final opinion

The project is **not ready** in its current state and displays clear signs of "erasure" of central parts of the planned architecture. Still, migration is fully **recoverable and feasible** if you reintroduce contract discipline, modularity by interface, and incremental pipeline (mock → real → QA).

In practical terms: **I do not recommend trying to "patch fix" the current state**. I recommend executing the phased recovery above and treating each phase as a formal acceptance criteria.

---

## 10) Status Update (post-recovery Phase A)

**Date:** 2026-02-16

Phase A recovery was **successfully completed**:

- ✅ Base code tree restored (`core/`, `scripts/`, `config/`)
- ✅ Pass1→Pass2 Contract implemented (`core/analysis/pass1_contract.py`, `core/utils/meta_validator.py`)
- ✅ Stable engine interface (`core/generation/interfaces.py`)
- ✅ FluxEngine mock + DummyEngine implemented (`core/generation/engines/`)
- ✅ Functional entrypoints:
  - `run_pass1_local.py` (standalone Pass1)
  - `run_two_pass_batch_local.py` (integrated Pass1→Pass2)
- ✅ Pass1 dependencies resolved (torch, numpy, PIL, cv2, YOLO, SAM)
- ✅ Batch execution of 3 real pages with `mode=ported_pass1` (no fallback)
- ✅ Contract validation passing (`scripts/validate_two_pass_outputs.py`)

**Validation commands:**
```bash
# Verify dependencies
python scripts/pass1_dependency_report.py

# Execute Pass1→Pass2 batch
python run_two_pass_batch_local.py \
  --input-dir data/pages_bw \
  --style-reference data/dummy_manga_test.png \
  --metadata-output metadata \
  --masks-output outputs/pass1/masks \
  --pass2-output outputs/pass2 \
  --chapter-id test_chapter \
  --engine dummy

# Validate artifacts
python scripts/validate_two_pass_outputs.py \
  --metadata-dir metadata \
  --pass2-dir outputs/pass2 \
  --expected-pages 3
```

**Next steps (Phase B):**
- Integrate real Flux into engine
- Automated QA + human process
- Full hardening and observability


## 11) Incremental update (Partial Phase B)

**Date:** 2026-02-17

Incremental advancements implemented:

- ✅ Pass2 with reinforced observability in runmeta (`duration_ms`, `timestamp_utc`, `options`, `output_image`)
- ✅ Pass2 local CLI with explicit generation controls (`--strength`, `--seed-override`)
- ✅ Integrated batch with Pass2 parameters (`--pass2-strength`, `--pass2-seed-offset`, `--pass2-option`)
- ✅ Batch summary generation (`outputs/pass2/batch_summary.json`)
- ✅ Local operation guide published (`DOCS/OPERATION.md`)
- ✅ More robust artifact validator (dynamic page discovery, `output_image` consistency, and optional `batch_summary.json` check)

Pendings to complete Phase B:

- Integrate real Flux engine (production inference)
- Implement dedicated OOM fallback and memory telemetry
- Institutionalize automated + human visual QA


## 12) Incremental update (API + extension)

**Date:** 2026-02-17

Advancements of this iteration:

- ✅ Minimal local API implemented (`api/server.py`) with `/health` and `/v1/pass2/run`
- ✅ Companion extension MV3 started (`extension/manga-flux-extension`) for health-check
- ✅ Dedicated documentation added (`DOCS/API_EXTENSION.md`)

Next pendings:

- [x] optional local authentication (token)
- [x] API batch endpoint (`POST /v1/pass2/batch`)
- [x] extension with form to trigger `/v1/pass2/run`
- [x] extension with form to trigger `/v1/pass2/batch`
- [x] local execution history in the extension
- [x] chapter pipeline via API from page URLs
- [x] current tab image capture in extension
- [x] light/dark theme and thumbnail UX with individual removal
- [x] extension state persistence for use after minimizing/closing popup
- [ ] FAISS integration in online flow (index/search)
