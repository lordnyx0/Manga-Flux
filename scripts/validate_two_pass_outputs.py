#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_META_KEYS = {
    "page_num",
    "page_image",
    "page_seed",
    "page_prompt",
    "style_reference",
    "text_mask",
}

REQUIRED_PASS1_RUNMETA_KEYS = {
    "metadata_file",
    "mode",
    "fallback_reason",
    "dependencies",
    "duration_ms",
    "timestamp_utc",
    "status",
}

REQUIRED_PASS2_RUNMETA_KEYS = {
    "meta_source",
    "engine",
    "seed",
    "strength",
    "status",
    "page_num",
    "input_image",
    "style_reference",
    "text_mask",
    "pass1_runmeta",
    "timestamp_utc",
    "duration_ms",
    "options",
    "output_image",
}

REQUIRED_BATCH_SUMMARY_KEYS = {
    "page_num",
    "input_page",
    "pass1_mode",
    "pass1_fallback_reason",
    "pass1_meta",
    "pass1_runmeta",
    "pass2_image",
    "pass2_strength",
    "pass2_seed_override",
    "pass2_options",
}


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_keys(path: Path, data: dict, required: set[str]) -> None:
    missing = [k for k in sorted(required) if k not in data]
    if missing:
        raise AssertionError(f"{path}: missing keys {missing}")


def _normalized_name(raw: str | Path) -> str:
    return Path(str(raw).replace("\\", "/")).name


def _discover_page_numbers(metadata_dir: Path) -> list[int]:
    pages: list[int] = []
    for meta_file in sorted(metadata_dir.glob("page_*.meta.json")):
        stem = meta_file.name.replace("page_", "").replace(".meta.json", "")
        if stem.isdigit():
            pages.append(int(stem))
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Pass1->Pass2 batch artifacts")
    parser.add_argument("--metadata-dir", default="metadata")
    parser.add_argument("--pass2-dir", default="outputs/smoke/pass2")
    parser.add_argument("--expected-pages", type=int, default=3)
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="Valida a primeira página esperada (útil para batches com paginação deslocada)",
    )
    parser.add_argument(
        "--require-batch-summary",
        action="store_true",
        help="Falha se pass2-dir/batch_summary.json não existir ou estiver inconsistente",
    )
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    pass2_dir = Path(args.pass2_dir)

    page_numbers = _discover_page_numbers(metadata_dir)
    if not page_numbers:
        raise AssertionError(f"Nenhum metadata encontrado em {metadata_dir}")

    if args.expected_pages > 0 and len(page_numbers) != args.expected_pages:
        raise AssertionError(
            f"Quantidade de páginas inesperada: found={len(page_numbers)} expected={args.expected_pages}"
        )

    if args.start_page is not None and page_numbers[0] != args.start_page:
        raise AssertionError(f"Primeira página esperada={args.start_page}, encontrada={page_numbers[0]}")

    for page_num in page_numbers:
        p = f"page_{page_num:03d}"
        meta = metadata_dir / f"{p}.meta.json"
        pass1_runmeta = metadata_dir / f"{p}.meta.pass1.runmeta.json"
        pass2_image = pass2_dir / f"{p}_colorized.png"
        pass2_runmeta = pass2_dir / f"{p}_colorized.runmeta.json"

        for path in [meta, pass1_runmeta, pass2_image, pass2_runmeta]:
            if not path.exists():
                raise AssertionError(f"Missing expected artifact: {path}")

        meta_data = _load_json(meta)
        assert isinstance(meta_data, dict)
        _assert_keys(meta, meta_data, REQUIRED_META_KEYS)
        if int(meta_data["page_num"]) != page_num:
            raise AssertionError(f"{meta}: page_num inconsistente ({meta_data['page_num']})")

        p1_data = _load_json(pass1_runmeta)
        assert isinstance(p1_data, dict)
        _assert_keys(pass1_runmeta, p1_data, REQUIRED_PASS1_RUNMETA_KEYS)
        if p1_data["status"] != "success":
            raise AssertionError(f"{pass1_runmeta}: status is not success")
        if _normalized_name(p1_data["metadata_file"]) != meta.name:
            raise AssertionError(
                f"{pass1_runmeta}: metadata_file mismatch ({p1_data['metadata_file']})"
            )
        if not isinstance(p1_data["duration_ms"], int) or p1_data["duration_ms"] < 0:
            raise AssertionError(f"{pass1_runmeta}: invalid duration_ms={p1_data['duration_ms']}")

        p2_data = _load_json(pass2_runmeta)
        assert isinstance(p2_data, dict)
        _assert_keys(pass2_runmeta, p2_data, REQUIRED_PASS2_RUNMETA_KEYS)
        if p2_data["status"] != "success":
            raise AssertionError(f"{pass2_runmeta}: status is not success")
        if int(p2_data["page_num"]) != page_num:
            raise AssertionError(f"{pass2_runmeta}: invalid page_num={p2_data['page_num']}")
        if _normalized_name(p2_data["pass1_runmeta"]) != pass1_runmeta.name:
            raise AssertionError(
                f"{pass2_runmeta}: pass1_runmeta mismatch ({p2_data['pass1_runmeta']})"
            )
        if _normalized_name(p2_data["output_image"]) != pass2_image.name:
            raise AssertionError(
                f"{pass2_runmeta}: output_image mismatch ({p2_data['output_image']})"
            )
        if not isinstance(p2_data["duration_ms"], int) or p2_data["duration_ms"] < 0:
            raise AssertionError(f"{pass2_runmeta}: invalid duration_ms={p2_data['duration_ms']}")
        if not isinstance(p2_data["options"], dict):
            raise AssertionError(f"{pass2_runmeta}: options must be an object")

    if args.require_batch_summary:
        summary_file = pass2_dir / "batch_summary.json"
        if not summary_file.exists():
            raise AssertionError(f"Missing expected artifact: {summary_file}")

        summary = _load_json(summary_file)
        if not isinstance(summary, list):
            raise AssertionError(f"{summary_file}: must be a list")
        if len(summary) != len(page_numbers):
            raise AssertionError(
                f"{summary_file}: invalid size={len(summary)} expected={len(page_numbers)}"
            )

        summary_pages = []
        for entry in summary:
            if not isinstance(entry, dict):
                raise AssertionError(f"{summary_file}: all entries must be objects")
            _assert_keys(summary_file, entry, REQUIRED_BATCH_SUMMARY_KEYS)
            summary_pages.append(int(entry["page_num"]))

        if sorted(summary_pages) != page_numbers:
            raise AssertionError(
                f"{summary_file}: page list mismatch summary={sorted(summary_pages)} metadata={page_numbers}"
            )

    print(f"[OK] Validated {len(page_numbers)} pages (Pass1->Pass2 artifacts + contracts).")


if __name__ == "__main__":
    main()
