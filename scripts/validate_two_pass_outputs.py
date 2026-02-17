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
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_keys(path: Path, data: dict, required: set[str]) -> None:
    missing = [k for k in sorted(required) if k not in data]
    if missing:
        raise AssertionError(f"{path}: missing keys {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Pass1->Pass2 batch artifacts")
    parser.add_argument("--metadata-dir", default="metadata")
    parser.add_argument("--pass2-dir", default="outputs/smoke/pass2")
    parser.add_argument("--expected-pages", type=int, default=3)
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    pass2_dir = Path(args.pass2_dir)

    for page_num in range(1, args.expected_pages + 1):
        p = f"page_{page_num:03d}"
        meta = metadata_dir / f"{p}.meta.json"
        pass1_runmeta = metadata_dir / f"{p}.meta.pass1.runmeta.json"
        pass2_image = pass2_dir / f"{p}_colorized.png"
        pass2_runmeta = pass2_dir / f"{p}_colorized.runmeta.json"

        for path in [meta, pass1_runmeta, pass2_image, pass2_runmeta]:
            if not path.exists():
                raise AssertionError(f"Missing expected artifact: {path}")

        meta_data = _load_json(meta)
        _assert_keys(meta, meta_data, REQUIRED_META_KEYS)

        p1_data = _load_json(pass1_runmeta)
        _assert_keys(pass1_runmeta, p1_data, REQUIRED_PASS1_RUNMETA_KEYS)
        if p1_data["status"] != "success":
            raise AssertionError(f"{pass1_runmeta}: status is not success")
        if not isinstance(p1_data["duration_ms"], int) or p1_data["duration_ms"] < 0:
            raise AssertionError(f"{pass1_runmeta}: invalid duration_ms={p1_data['duration_ms']}")

        p2_data = _load_json(pass2_runmeta)
        _assert_keys(pass2_runmeta, p2_data, REQUIRED_PASS2_RUNMETA_KEYS)
        if p2_data["status"] != "success":
            raise AssertionError(f"{pass2_runmeta}: status is not success")
        if Path(p2_data["pass1_runmeta"]).name != pass1_runmeta.name:
            raise AssertionError(
                f"{pass2_runmeta}: pass1_runmeta mismatch ({p2_data['pass1_runmeta']})"
            )

    print(f"[OK] Validated {args.expected_pages} pages (Pass1->Pass2 artifacts + contracts).")


if __name__ == "__main__":
    main()
