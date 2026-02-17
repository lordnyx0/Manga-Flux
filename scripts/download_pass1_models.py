#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import requests

DEFAULT_MODEL_URL = (
    "https://huggingface.co/deepghs/manga109_yolo/resolve/main/"
    "v2023.12.07_n/model.onnx"
)


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with target.open("wb") as fp:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fp.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Pass1 bootstrap models")
    parser.add_argument("--yolo-url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--yolo-output", default="data/models/manga109_yolo.onnx")
    args = parser.parse_args()

    out = Path(args.yolo_output)
    print(f"[INFO] Downloading YOLO model from: {args.yolo_url}")
    download_file(args.yolo_url, out)
    print(f"[OK] Saved: {out}")


if __name__ == "__main__":
    main()
