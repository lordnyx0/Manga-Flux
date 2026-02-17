from __future__ import annotations

import argparse
import base64
import json
import os
import re
import traceback
import urllib.request
import urllib.parse
import urllib.error
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.analysis.pass1_pipeline import run_pass1_with_report
from core.generation.engines.dummy_engine import DummyEngine
from core.generation.engines.flux_engine import FluxEngine
from core.generation.pipeline import Pass2Generator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Token")
    handler.end_headers()
    handler.wfile.write(body)


def _load_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_len = handler.headers.get("Content-Length", "0")
    try:
        content_length = int(raw_len)
    except ValueError as exc:
        raise ValueError(f"Invalid Content-Length header: {raw_len}") from exc

    if content_length <= 0:
        return {}

    body = handler.rfile.read(content_length)
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Body must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")

    return payload


def _make_engine(engine_name: str):
    if engine_name == "flux":
        return FluxEngine()
    if engine_name == "dummy":
        return DummyEngine()
    raise ValueError("Invalid engine. Supported values: flux, dummy")


def _list_meta_files(metadata_dir: Path) -> list[Path]:
    return [p for p in sorted(metadata_dir.glob("page_*.meta.json")) if p.is_file()]


def _slugify(raw: str, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip())
    value = value.strip("_")
    return value or fallback




DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _download_with_urllib(url: str, timeout: int = 30) -> bytes:
    parsed = urllib.parse.urlparse(url)
    headers = dict(DEFAULT_HTTP_HEADERS)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 (local tool usage)
        return response.read()


def _download_with_cloudscraper(url: str, timeout: int = 30) -> bytes | None:
    try:
        import cloudscraper  # type: ignore
    except Exception:
        return None

    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
    response = scraper.get(url, headers=DEFAULT_HTTP_HEADERS, timeout=timeout)
    response.raise_for_status()
    return bytes(response.content)


def _download_with_requests(url: str, timeout: int = 30) -> bytes | None:
    try:
        import requests  # type: ignore
    except Exception:
        return None

    response = requests.get(url, headers=DEFAULT_HTTP_HEADERS, timeout=timeout)
    response.raise_for_status()
    return bytes(response.content)

def _download_to(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for attempt in range(1, 4):
        try:
            data = _download_with_urllib(url, timeout=30)
            dest.write_bytes(data)
            return
        except urllib.error.HTTPError as exc:
            errors.append(f"urllib HTTP {exc.code}")
            if exc.code not in {403, 429, 503}:
                raise
        except Exception as exc:  # pragma: no cover - network behavior
            errors.append(f"urllib {type(exc).__name__}: {exc}")

        if attempt == 1:
            try:
                data = _download_with_cloudscraper(url, timeout=30)
                if data:
                    dest.write_bytes(data)
                    return
            except Exception as exc:  # pragma: no cover - optional dependency
                errors.append(f"cloudscraper {type(exc).__name__}: {exc}")

        if attempt == 2:
            try:
                data = _download_with_requests(url, timeout=30)
                if data:
                    dest.write_bytes(data)
                    return
            except Exception as exc:  # pragma: no cover - optional dependency
                errors.append(f"requests {type(exc).__name__}: {exc}")

        time.sleep(min(1.5 * attempt, 4.0))

    raise RuntimeError(
        "Failed to download image after retries. "
        f"Last attempts: {' | '.join(errors[-4:])}"
    )


def _resolve_chapter_pages(payload: dict[str, Any], inputs_dir: Path) -> list[dict[str, str]]:
    page_urls = payload.get("page_urls", [])
    page_uploads = payload.get("page_uploads", [])
    page_referer = str(payload.get("page_referer", "")).strip()

    if page_urls is None:
        page_urls = []
    if page_uploads is None:
        page_uploads = []

    if not isinstance(page_urls, list):
        raise ValueError("'page_urls' must be a list")
    if not isinstance(page_uploads, list):
        raise ValueError("'page_uploads' must be a list")
    if not page_urls and not page_uploads:
        raise ValueError("Provide at least one page via 'page_urls' or 'page_uploads'")

    inputs_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, str]] = []

    for page_url in page_urls:
        if not isinstance(page_url, str) or not page_url.strip():
            raise ValueError("Invalid item in 'page_urls'")
        pages.append({"source": "url", "source_value": page_url.strip()})

    for upload in page_uploads:
        if not isinstance(upload, dict):
            raise ValueError("Invalid item in 'page_uploads'")
        filename = str(upload.get("filename", "page_upload.png")).strip()
        content_base64 = str(upload.get("content_base64", "")).strip()
        if not content_base64:
            raise ValueError("Each 'page_uploads' item must include 'content_base64'")

        suffix = Path(filename).suffix.lower() or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            suffix = ".png"

        clean_stem = _slugify(Path(filename).stem, "page_upload")
        try:
            raw = base64.b64decode(content_base64, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 in 'page_uploads'") from exc

        pages.append({
            "source": "upload",
            "filename": f"{clean_stem}{suffix}",
            "raw_bytes": raw,
        })

    resolved: list[dict[str, str]] = []
    for idx, page in enumerate(pages, start=1):
        source = str(page["source"])
        source_value = str(page.get("source_value", ""))
        suffix = Path(str(page.get("filename", "page.png"))).suffix.lower() or ".png"
        page_path = inputs_dir / f"page_{idx:03d}{suffix}"

        if source == "url":
            _download_to(source_value, page_path, referer=page_referer or None)
            source_label = source_value
        else:
            raw_bytes = page.get("raw_bytes")
            if not isinstance(raw_bytes, (bytes, bytearray)):
                raise ValueError("Invalid upload bytes")
            page_path.write_bytes(bytes(raw_bytes))
            source_label = str(page.get("filename", f"upload_{idx:03d}{suffix}"))

        resolved.append({
            "source": source,
            "source_label": source_label,
            "input_path": str(page_path),
        })

    return resolved


def _resolve_style_reference(payload: dict[str, Any], root_dir: Path) -> tuple[Path, str]:
    style_reference_url = str(payload.get("style_reference_url", "")).strip()
    style_reference_base64 = str(payload.get("style_reference_base64", "")).strip()
    style_reference_filename = str(payload.get("style_reference_filename", "style_reference.png")).strip()

    if style_reference_base64:
        clean_name = _slugify(Path(style_reference_filename).stem, "style_reference")
        suffix = Path(style_reference_filename).suffix.lower() or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            suffix = ".png"
        style_path = root_dir / f"{clean_name}{suffix}"
        style_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = base64.b64decode(style_reference_base64, validate=True)
        except Exception as exc:
            raise ValueError("Invalid 'style_reference_base64' payload") from exc
        style_path.write_bytes(raw)
        return style_path, "upload"

    if style_reference_url:
        style_path = root_dir / "style_reference.png"
        _download_to(style_reference_url, style_path)
        return style_path, "url"

    raise ValueError("Provide 'style_reference_url' or uploaded 'style_reference_base64'")


class MangaFluxAPIHandler(BaseHTTPRequestHandler):
    server_version = "MangaFluxAPI/0.5"

    def _send_error(self, status: int, message: str, details: str | None = None) -> None:
        payload: dict[str, Any] = {
            "status": "error",
            "message": message,
            "timestamp_utc": _utc_now(),
        }
        if details:
            payload["details"] = details
        _json_response(self, status, payload)

    def _check_token(self) -> bool:
        required_token = os.getenv("MANGA_FLUX_API_TOKEN", "").strip()
        if not required_token:
            return True
        request_token = self.headers.get("X-API-Token", "").strip()
        return request_token == required_token

    def do_OPTIONS(self) -> None:  # noqa: N802
        _json_response(self, HTTPStatus.NO_CONTENT, {"status": "ok"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "manga-flux-api",
                    "version": "0.5",
                    "token_required": bool(os.getenv("MANGA_FLUX_API_TOKEN", "").strip()),
                    "timestamp_utc": _utc_now(),
                },
            )
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_token():
            return self._send_error(HTTPStatus.UNAUTHORIZED, "Invalid or missing X-API-Token")

        if self.path == "/v1/pass2/run":
            return self._handle_pass2_run()
        if self.path == "/v1/pass2/batch":
            return self._handle_pass2_batch()
        if self.path == "/v1/pipeline/run_chapter":
            return self._handle_run_chapter()
        self._send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def _parse_common(self, payload: dict[str, Any]) -> tuple[str, float, int | None, dict[str, Any]]:
        engine_name = str(payload.get("engine", "flux")).strip().lower()
        if engine_name not in {"flux", "dummy"}:
            raise ValueError("Invalid engine. Supported values: flux, dummy")

        strength_raw = payload.get("strength", 1.0)
        try:
            strength = float(strength_raw)
        except Exception as exc:
            raise ValueError("'strength' must be numeric") from exc

        seed_override = payload.get("seed_override")
        if seed_override is not None:
            try:
                seed_override = int(seed_override)
            except Exception as exc:
                raise ValueError("'seed_override' must be integer") from exc

        options = payload.get("options", {})
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("'options' must be an object")

        return engine_name, strength, seed_override, options

    def _handle_pass2_run(self) -> None:
        try:
            payload = _load_json_body(self)
            engine_name, strength, seed_override, options = self._parse_common(payload)
        except Exception as exc:
            return self._send_error(HTTPStatus.BAD_REQUEST, "Invalid request body", str(exc))

        meta_path = str(payload.get("meta_path", "")).strip()
        output_dir = str(payload.get("output_dir", "outputs/api/pass2")).strip()
        if not meta_path:
            return self._send_error(HTTPStatus.BAD_REQUEST, "'meta_path' is required")

        try:
            engine = _make_engine(engine_name)
            generator = Pass2Generator(engine)
            image_path = generator.process_page(
                meta_path=meta_path,
                output_dir=output_dir,
                strength=strength,
                seed_override=seed_override,
                options=options,
            )

            page_name = Path(image_path).name.replace("_colorized.png", "")
            runmeta_path = str(Path(output_dir) / f"{page_name}_colorized.runmeta.json")

            return _json_response(
                self,
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "image_path": image_path,
                    "runmeta_path": runmeta_path,
                    "engine": engine_name,
                    "timestamp_utc": _utc_now(),
                },
            )
        except Exception as exc:
            return self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Pass2 execution failed",
                f"{exc}\n{traceback.format_exc(limit=1)}",
            )

    def _handle_pass2_batch(self) -> None:
        try:
            payload = _load_json_body(self)
            engine_name, strength, seed_override, options = self._parse_common(payload)
        except Exception as exc:
            return self._send_error(HTTPStatus.BAD_REQUEST, "Invalid request body", str(exc))

        metadata_dir = Path(str(payload.get("metadata_dir", "metadata"))).expanduser()
        output_dir = Path(str(payload.get("output_dir", "outputs/api/pass2"))).expanduser()

        if not metadata_dir.exists() or not metadata_dir.is_dir():
            return self._send_error(HTTPStatus.BAD_REQUEST, "'metadata_dir' must be an existing directory")

        meta_files = _list_meta_files(metadata_dir)
        if not meta_files:
            return self._send_error(HTTPStatus.BAD_REQUEST, "No page_*.meta.json found in metadata_dir")

        if payload.get("expected_pages") is not None:
            try:
                expected_pages = int(payload["expected_pages"])
            except Exception:
                return self._send_error(HTTPStatus.BAD_REQUEST, "'expected_pages' must be integer")
            if expected_pages > 0 and len(meta_files) != expected_pages:
                return self._send_error(
                    HTTPStatus.BAD_REQUEST,
                    f"expected_pages mismatch: found={len(meta_files)} expected={expected_pages}",
                )

        results = []
        try:
            engine = _make_engine(engine_name)
            generator = Pass2Generator(engine)

            for meta_file in meta_files:
                image_path = generator.process_page(
                    meta_path=str(meta_file),
                    output_dir=str(output_dir),
                    strength=strength,
                    seed_override=seed_override,
                    options=options,
                )
                page_name = Path(image_path).name.replace("_colorized.png", "")
                runmeta_path = str(output_dir / f"{page_name}_colorized.runmeta.json")
                results.append(
                    {
                        "meta_path": str(meta_file),
                        "image_path": image_path,
                        "runmeta_path": runmeta_path,
                    }
                )

            return _json_response(
                self,
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "count": len(results),
                    "engine": engine_name,
                    "results": results,
                    "timestamp_utc": _utc_now(),
                },
            )
        except Exception as exc:
            return self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Pass2 batch execution failed",
                f"{exc}\n{traceback.format_exc(limit=1)}",
            )

    def _handle_run_chapter(self) -> None:
        try:
            payload = _load_json_body(self)
            engine_name, strength, seed_override, options = self._parse_common(payload)
        except Exception as exc:
            return self._send_error(HTTPStatus.BAD_REQUEST, "Invalid request body", str(exc))

        manga_id = _slugify(str(payload.get("manga_id", "manga_default")), "manga_default")
        chapter_id = _slugify(str(payload.get("chapter_id", "chapter_001")), "chapter_001")
        debug_dump_json = bool(payload.get("debug_dump_json", False))

        root_dir = Path(str(payload.get("output_root", "output"))) / manga_id / "chapters" / chapter_id
        inputs_dir = root_dir / "inputs"
        pass1_mask_dir = root_dir / "pass1" / "masks"
        metadata_dir = root_dir / "metadata"
        pass2_dir = root_dir / "pass2"

        try:
            root_dir.mkdir(parents=True, exist_ok=True)
            state_db_path = root_dir / "pipeline_state.db"
            style_path, style_source = _resolve_style_reference(payload, root_dir)

            engine = _make_engine(engine_name)
            options = dict(options)
            options["chapter_id"] = chapter_id
            pass2 = Pass2Generator(engine, state_db_path=str(state_db_path))
            page_results = []
            chapter_pages = _resolve_chapter_pages(payload, inputs_dir)

            for i, chapter_page in enumerate(chapter_pages, start=1):
                page_path = Path(chapter_page["input_path"])

                mask_path = pass1_mask_dir / f"page_{i:03d}_text.png"
                prompt = f"manga page colorization page={i} manga={manga_id} chapter={chapter_id}"
                p1 = run_pass1_with_report(
                    page_image=str(page_path),
                    style_reference=str(style_path),
                    output_mask=str(mask_path),
                    output_metadata_dir=str(metadata_dir),
                    page_num=i,
                    page_prompt=prompt,
                    chapter_id=chapter_id,
                    state_db_path=str(state_db_path),
                    debug_dump_json=debug_dump_json,
                )

                image_path = pass2.process_page_from_state(
                    chapter_id=chapter_id,
                    page_num=i,
                    output_dir=str(pass2_dir),
                    strength=strength,
                    seed_override=seed_override,
                    options=options,
                    debug_dump_json=debug_dump_json,
                )

                page_results.append(
                    {
                        "page_num": i,
                        "source": chapter_page["source"],
                        "source_label": chapter_page["source_label"],
                        "input_path": str(page_path),
                        "metadata_path": str(p1.metadata_path),
                        "pass1_runmeta": str(p1.runmeta_path),
                        "pass2_image": image_path,
                    }
                )

            return _json_response(
                self,
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "manga_id": manga_id,
                    "chapter_id": chapter_id,
                    "output_root": str(root_dir),
                    "count": len(page_results),
                    "engine": engine_name,
                    "style_reference_source": style_source,
                    "state_db": str(state_db_path),
                    "results": page_results,
                    "timestamp_utc": _utc_now(),
                },
            )
        except Exception as exc:
            return self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Chapter pipeline execution failed",
                f"{exc}\n{traceback.format_exc(limit=1)}",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manga-Flux local API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MangaFluxAPIHandler)
    print(f"[INFO] Manga-Flux API listening on http://{args.host}:{args.port}")
    if os.getenv("MANGA_FLUX_API_TOKEN", "").strip():
        print("[INFO] API token auth enabled (header: X-API-Token)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
