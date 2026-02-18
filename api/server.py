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


# ---------------------------------------------------------------------------
# Helpers — time / JSON
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: dict[str, Any],
) -> None:
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


# ---------------------------------------------------------------------------
# Helpers — misc
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

DEFAULT_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _origin(url: str) -> str:
    """Return scheme://netloc for *url*."""
    p = urllib.parse.urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _build_image_headers(
    url: str,
    referer: str | None = None,
    cookie_header: str | None = None,
) -> dict[str, str]:
    """
    Merge DEFAULT_HTTP_HEADERS with caller-supplied overrides.

    referer  — if omitted, derived from the URL origin.
    cookie_header — raw Cookie string forwarded from the browser extension
                    (may contain cf_clearance, session tokens, etc.).
    """
    headers = dict(DEFAULT_HTTP_HEADERS)
    headers["Referer"] = referer or (_origin(url) + "/")
    if cookie_header:
        headers["Cookie"] = cookie_header
    return headers


# ---------------------------------------------------------------------------
# Session warm-up  (key fix for WP-Manga / Madara hotlink protection)
# ---------------------------------------------------------------------------
#
# Sites like imperiodabritannia.com serve manga images only when the request
# carries cookies that were set during a normal page-load on the same domain.
# The image CDN checks that the visitor already has a session (WordPress
# cookies, CF cookies, nonce tokens, etc.) before serving the file.
#
# Strategy:
#   1. Make a GET request to the *referer* page (the manga chapter URL).
#      This is a normal HTML page load — the server sets its session cookies.
#   2. Use those cookies + the referer on every subsequent image request.
#
# This is equivalent to a user opening the chapter in a browser tab and then
# the browser fetching each image with the same session — which is exactly
# what the server-side hotlink check expects.
# ---------------------------------------------------------------------------

def _warm_session(
    session,           # requests.Session
    referer: str,
    timeout: int = 20,
) -> None:
    """
    Visit *referer* so the session accumulates the cookies the site requires.
    Errors are silenced: a failed warm-up just means we try without cookies.
    """
    try:
        session.get(
            referer,
            headers={**DEFAULT_HTTP_HEADERS, "Accept": "text/html,*/*;q=0.8"},
            timeout=timeout,
            allow_redirects=True,
        )
    except Exception:
        pass


def _cookies_from_session(session) -> str:
    """Serialise session cookie jar to a raw Cookie header string."""
    return "; ".join(f"{c.name}={c.value}" for c in session.cookies)


# ---------------------------------------------------------------------------
# Per-tier download implementations
# ---------------------------------------------------------------------------

def _download_with_urllib(
    url: str,
    timeout: int = 30,
    referer: str | None = None,
    cookie_header: str | None = None,
) -> bytes:
    headers = _build_image_headers(url, referer=referer, cookie_header=cookie_header)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return resp.read()


def _download_with_requests_session(
    url: str,
    timeout: int = 30,
    referer: str | None = None,
    cookie_header: str | None = None,
) -> bytes | None:
    """
    Two-step download using a persistent requests.Session:

      Step 1 — warm up: GET the referer page so the session collects the
               cookies the site sets on a normal HTML page-load.
      Step 2 — image fetch: GET the image URL with those session cookies
               plus the Referer header pointing at the chapter page.

    This resolves 403 errors from WP-Manga / Madara hotlink protection,
    which gates image delivery on the visitor already holding valid session
    cookies from a prior page visit on the same domain.

    If the caller supplies an explicit *cookie_header* (forwarded from the
    browser extension), that takes priority over the session warm-up cookies.
    """
    try:
        import requests  # type: ignore
    except Exception:
        return None

    effective_referer = referer or (_origin(url) + "/")
    session = requests.Session()

    # If the caller did not supply cookies, warm the session by visiting
    # the chapter page so we obtain whatever cookies the site requires.
    if not cookie_header:
        _warm_session(session, effective_referer, timeout=20)
        # Merge harvested cookies into the cookie_header for urllib fallbacks.
        harvested = _cookies_from_session(session)
        effective_cookie = harvested or None
    else:
        effective_cookie = cookie_header

    headers = _build_image_headers(url, referer=effective_referer, cookie_header=effective_cookie)
    resp = session.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return bytes(resp.content)


def _download_with_cloudscraper(
    url: str,
    timeout: int = 30,
    referer: str | None = None,
    cookie_header: str | None = None,
) -> bytes | None:
    """
    Fallback using cloudscraper (handles legacy Cloudflare JS-challenge V1).
    Ineffective against modern Managed Challenge / Turnstile, but still
    worth trying as a cheap third option.
    """
    try:
        import cloudscraper  # type: ignore
    except Exception:
        return None

    effective_referer = referer or (_origin(url) + "/")
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )

    # Warm up the scraper session the same way.
    if not cookie_header:
        try:
            scraper.get(
                effective_referer,
                headers={**DEFAULT_HTTP_HEADERS, "Accept": "text/html,*/*;q=0.8"},
                timeout=20,
            )
        except Exception:
            pass

    headers = _build_image_headers(url, referer=effective_referer, cookie_header=cookie_header)
    resp = scraper.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return bytes(resp.content)


# ---------------------------------------------------------------------------
# Public download entry-point
# ---------------------------------------------------------------------------

def _download_to(
    url: str,
    dest: Path,
    referer: str | None = None,
    cookie_header: str | None = None,
) -> None:
    """
    Download *url* to *dest*.

    Tier 1  requests.Session with session warm-up  (best for hotlink-protected
            WordPress/Madara sites — visits the chapter page first to collect
            session cookies, then fetches the image with those cookies).

    Tier 2  urllib  (stdlib; used when requests is unavailable or on retry).

    Tier 3  cloudscraper  (legacy Cloudflare JS-challenge bypass).

    Parameters
    ----------
    url:
        Full image URL.
    dest:
        Destination path. Parent directories are created automatically.
    referer:
        The manga chapter page URL (e.g. the URL the extension is currently
        viewing). Used both as the Referer header and as the warm-up URL.
        If omitted, the URL origin is used — less effective for strict
        per-chapter hotlink checks.
    cookie_header:
        Raw Cookie string forwarded from the browser extension.  When provided
        the session warm-up step is skipped and these cookies are used directly,
        which is the most reliable path for Cloudflare-protected sites.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for attempt in range(1, 4):

        # --- Tier 1: requests.Session with warm-up (primary) ------------------
        try:
            data = _download_with_requests_session(
                url, timeout=30, referer=referer, cookie_header=cookie_header
            )
            if data:
                dest.write_bytes(data)
                return
        except Exception as exc:
            errors.append(f"requests-session {type(exc).__name__}: {exc}")

        # --- Tier 2: urllib (stdlib fallback) ---------------------------------
        try:
            data = _download_with_urllib(
                url, timeout=30, referer=referer, cookie_header=cookie_header
            )
            dest.write_bytes(data)
            return
        except urllib.error.HTTPError as exc:
            errors.append(f"urllib HTTP {exc.code}")
            if exc.code not in {403, 429, 503}:
                raise
        except Exception as exc:
            errors.append(f"urllib {type(exc).__name__}: {exc}")

        # --- Tier 3: cloudscraper (legacy CF, attempt 2 only) -----------------
        if attempt == 2:
            try:
                data = _download_with_cloudscraper(
                    url, timeout=30, referer=referer, cookie_header=cookie_header
                )
                if data:
                    dest.write_bytes(data)
                    return
            except Exception as exc:
                errors.append(f"cloudscraper {type(exc).__name__}: {exc}")

        time.sleep(min(1.5 * attempt, 4.0))

    raise RuntimeError(
        "Failed to download image after retries.\n"
        f"Last attempts: {' | '.join(errors[-6:])}\n\n"
        "Possible causes and fixes:\n"
        "  1. HOTLINK PROTECTION (most common for WP-Manga/Madara sites):\n"
        "     Pass the chapter page URL as 'page_referer' in your request payload.\n"
        "     Example: \"page_referer\": \"https://imperiodabritannia.com/manga/x/chapter-42/\"\n"
        "  2. CLOUDFLARE MANAGED CHALLENGE:\n"
        "     Use the browser extension to forward cookies via 'page_cookie_header'.\n"
        "     Example: \"page_cookie_header\": \"cf_clearance=abc123; ...\"\n"
        "  3. UPLOAD DIRECTLY:\n"
        "     Use 'page_uploads' with base64-encoded images — the extension can\n"
        "     read the images from the page DOM and send them directly.\n"
    )


# ---------------------------------------------------------------------------
# Payload resolution helpers
# ---------------------------------------------------------------------------

def _resolve_chapter_pages(
    payload: dict[str, Any],
    inputs_dir: Path,
) -> list[dict[str, str]]:
    """
    Resolve chapter pages from either remote URLs or base64 uploads.

    For URL-based pages, *page_referer* and *page_cookie_header* from the
    payload are forwarded to the download layer so the browser extension can
    supply session cookies (e.g. ``cf_clearance``) required by Cloudflare-
    protected manga sites.
    """
    page_urls: list = payload.get("page_urls") or []
    page_uploads: list = payload.get("page_uploads") or []
    page_referer = str(payload.get("page_referer", "")).strip() or None
    page_cookie_header = str(payload.get("page_cookie_header", "")).strip() or None

    if not isinstance(page_urls, list):
        raise ValueError("'page_urls' must be a list")
    if not isinstance(page_uploads, list):
        raise ValueError("'page_uploads' must be a list")
    if not page_urls and not page_uploads:
        raise ValueError("Provide at least one page via 'page_urls' or 'page_uploads'")

    inputs_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []

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
        suffix = Path(str(page.get("filename", "page.png"))).suffix.lower() or ".png"
        page_path = inputs_dir / f"page_{idx:03d}{suffix}"

        if source == "url":
            source_value = str(page["source_value"])
            _download_to(
                url=source_value,
                dest=page_path,
                referer=page_referer,
                cookie_header=page_cookie_header,
            )
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


def _resolve_style_reference(
    payload: dict[str, Any],
    root_dir: Path,
) -> tuple[Path, str]:
    style_reference_url = str(payload.get("style_reference_url", "")).strip()
    style_reference_base64 = str(payload.get("style_reference_base64", "")).strip()
    style_reference_filename = str(
        payload.get("style_reference_filename", "style_reference.png")
    ).strip()

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
        _download_to(url=style_reference_url, dest=style_path)
        return style_path, "url"

    raise ValueError(
        "Provide 'style_reference_url' or uploaded 'style_reference_base64'"
    )


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class MangaFluxAPIHandler(BaseHTTPRequestHandler):
    server_version = "MangaFluxAPI/0.5"

    # ------------------------------------------------------------------
    # Auth / error helpers
    # ------------------------------------------------------------------

    def _send_error(
        self,
        status: int,
        message: str,
        details: str | None = None,
    ) -> None:
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

    # ------------------------------------------------------------------
    # HTTP verbs
    # ------------------------------------------------------------------

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
            return self._send_error(
                HTTPStatus.UNAUTHORIZED, "Invalid or missing X-API-Token"
            )

        routes = {
            "/v1/pass2/run": self._handle_pass2_run,
            "/v1/pass2/batch": self._handle_pass2_batch,
            "/v1/pipeline/run_chapter": self._handle_run_chapter,
        }
        handler_fn = routes.get(self.path)
        if handler_fn:
            return handler_fn()
        self._send_error(HTTPStatus.NOT_FOUND, "Route not found")

    # ------------------------------------------------------------------
    # Common payload parsing
    # ------------------------------------------------------------------

    def _parse_common(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, float, int | None, dict[str, Any]]:
        engine_name = str(payload.get("engine", "flux")).strip().lower()
        if engine_name not in {"flux", "dummy"}:
            raise ValueError("Invalid engine. Supported values: flux, dummy")

        try:
            strength = float(payload.get("strength", 1.0))
        except Exception as exc:
            raise ValueError("'strength' must be numeric") from exc

        seed_override = payload.get("seed_override")
        if seed_override is not None:
            try:
                seed_override = int(seed_override)
            except Exception as exc:
                raise ValueError("'seed_override' must be integer") from exc

        options = payload.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("'options' must be an object")

        return engine_name, strength, seed_override, options

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

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
            return self._send_error(
                HTTPStatus.BAD_REQUEST, "'metadata_dir' must be an existing directory"
            )

        meta_files = _list_meta_files(metadata_dir)
        if not meta_files:
            return self._send_error(
                HTTPStatus.BAD_REQUEST, "No page_*.meta.json found in metadata_dir"
            )

        if payload.get("expected_pages") is not None:
            try:
                expected_pages = int(payload["expected_pages"])
            except Exception:
                return self._send_error(
                    HTTPStatus.BAD_REQUEST, "'expected_pages' must be integer"
                )
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

        root_dir = (
            Path(str(payload.get("output_root", "output")))
            / manga_id / "chapters" / chapter_id
        )
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

            chapter_pages = _resolve_chapter_pages(payload, inputs_dir)
            page_results = []

            for i, chapter_page in enumerate(chapter_pages, start=1):
                page_path = Path(chapter_page["input_path"])
                mask_path = pass1_mask_dir / f"page_{i:03d}_text.png"
                prompt = (
                    f"manga page colorization "
                    f"page={i} manga={manga_id} chapter={chapter_id}"
                )

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
