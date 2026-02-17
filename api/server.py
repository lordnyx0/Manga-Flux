"""Minimal HTTP API bootstrap for Manga-Flux (stdlib-only).

This module intentionally avoids third-party dependencies so the project can
start API integration in constrained environments.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

API_VERSION = "0.5.0-bootstrap"
PAGE_META_RE = re.compile(r"^page_(\d{3})\.meta\.json$")


def build_openapi_schema() -> dict[str, Any]:
    """Return a minimal OpenAPI schema for the bootstrap API."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Manga-Flux Bootstrap API",
            "version": API_VERSION,
            "description": "Bootstrap API for Two-Pass Manga-Flux orchestration.",
        },
        "paths": {
            "/healthz": {"get": {"summary": "Health check"}},
            "/version": {"get": {"summary": "API version"}},
            "/openapi.json": {"get": {"summary": "OpenAPI schema"}},
            "/v1/jobs/two-pass": {
                "post": {
                    "summary": "Create a two-pass job",
                    "description": "Requires style_reference. Requires bearer token when MANGA_FLUX_API_TOKEN is configured.",
                }
            },
            "/v1/jobs/{job_id}": {"get": {"summary": "Get job status"}},
            "/v1/jobs/{job_id}/artifacts": {"get": {"summary": "Get job artifacts"}},
            "/v1/chapters/{chapter_id}/pages": {"get": {"summary": "List chapter pages"}},
            "/v1/chapters/{chapter_id}/pages/{page_num}/metadata": {"get": {"summary": "Locate page metadata"}},
            "/v1/chapters/{chapter_id}/pages/{page_num}/runmeta/pass1": {"get": {"summary": "Locate Pass1 runmeta"}},
            "/v1/chapters/{chapter_id}/pages/{page_num}/runmeta/pass2": {"get": {"summary": "Locate Pass2 runmeta"}},
            "/v1/chapters/{chapter_id}/pages/{page_num}/mask": {"get": {"summary": "Locate text mask"}},
            "/v1/chapters/{chapter_id}/pages/{page_num}/colorized": {"get": {"summary": "Locate colorized image"}},
        },
    }


@dataclass
class JobStore:
    """Simple JSON-file job store for bootstrap usage."""

    path: Path
    _lock: threading.Lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"jobs": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            db = self._read()
            job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            item = {
                "job_id": job_id,
                "status": "queued",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "request": payload,
                "progress": {
                    "total_pages": 0,
                    "pass1_done": 0,
                    "pass2_done": 0,
                },
            }
            db.setdefault("jobs", {})[job_id] = item
            self._write(db)
            return item

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            db = self._read()
            return db.get("jobs", {}).get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            db = self._read()
            return list(db.get("jobs", {}).values())


class MangaFluxAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for bootstrap API endpoints."""

    job_store = JobStore(Path(os.environ.get("MANGA_FLUX_JOB_STORE", "outputs/api/jobs.json")))

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        trace_id = uuid.uuid4().hex[:12]
        self._send_json(
            status,
            {
                "code": code,
                "message": message,
                "details": details or {},
                "trace_id": trace_id,
            },
        )

    def _require_write_token(self) -> bool:
        """Validate bearer token for write operations when configured."""
        expected_token = os.environ.get("MANGA_FLUX_API_TOKEN", "").strip()
        if not expected_token:
            return True

        auth_header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not auth_header.startswith(prefix):
            self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "Missing bearer token")
            return False

        provided = auth_header[len(prefix):].strip()
        if provided != expected_token:
            self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "Invalid bearer token")
            return False

        return True

    def _read_json_body(self) -> dict[str, Any] | None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return None
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _extract_job_path(self, path: str) -> tuple[str, str] | None:
        parts = [part for part in path.split("/") if part]
        if len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
            return parts[2], "status"
        if len(parts) == 4 and parts[:2] == ["v1", "jobs"] and parts[3] == "artifacts":
            return parts[2], "artifacts"
        return None

    def _extract_chapter_path(self, path: str) -> tuple[str, str, int | None] | None:
        parts = [part for part in path.split("/") if part]
        if len(parts) == 4 and parts[0:2] == ["v1", "chapters"] and parts[3] == "pages":
            return parts[2], "pages", None

        if len(parts) >= 6 and parts[0:2] == ["v1", "chapters"] and parts[3] == "pages":
            chapter_id = parts[2]
            try:
                page_num = int(parts[4])
            except ValueError:
                return None

            tail = parts[5:]
            if tail == ["metadata"]:
                return chapter_id, "metadata", page_num
            if tail == ["runmeta", "pass1"]:
                return chapter_id, "runmeta_pass1", page_num
            if tail == ["runmeta", "pass2"]:
                return chapter_id, "runmeta_pass2", page_num
            if tail == ["mask"]:
                return chapter_id, "mask", page_num
            if tail == ["colorized"]:
                return chapter_id, "colorized", page_num
        return None

    def _find_latest_job_by_chapter(self, chapter_id: str) -> dict[str, Any] | None:
        jobs = [job for job in self.job_store.list_jobs() if job.get("request", {}).get("chapter_id") == chapter_id]
        if not jobs:
            return None
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return jobs[0]

    def _get_chapter_dirs(self, chapter_id: str) -> tuple[Path, Path, Path]:
        job = self._find_latest_job_by_chapter(chapter_id)
        if job:
            req = job.get("request", {})
            metadata_dir = Path(req.get("metadata_output", "metadata"))
            pass2_dir = Path(req.get("pass2_output", "outputs/pass2"))
            masks_dir = Path(req.get("masks_output", "outputs/pass1/masks"))
            return metadata_dir, pass2_dir, masks_dir
        return Path("metadata"), Path("outputs/pass2"), Path("outputs/pass1/masks")

    def _list_chapter_pages(self, chapter_id: str) -> list[int]:
        metadata_dir, _, _ = self._get_chapter_dirs(chapter_id)
        if not metadata_dir.exists():
            return []
        pages: list[int] = []
        for candidate in metadata_dir.iterdir():
            match = PAGE_META_RE.match(candidate.name)
            if match:
                pages.append(int(match.group(1)))
        return sorted(set(pages))

    def _build_artifacts_payload(self, job: dict[str, Any]) -> dict[str, Any]:
        request = job.get("request", {})
        chapter_id = request.get("chapter_id", "unknown_chapter")
        metadata_dir = request.get("metadata_output", "metadata")
        pass2_dir = request.get("pass2_output", "outputs/pass2")
        artifacts = [
            {
                "kind": "metadata_dir",
                "path": metadata_dir,
                "exists": Path(metadata_dir).exists(),
            },
            {
                "kind": "pass2_dir",
                "path": pass2_dir,
                "exists": Path(pass2_dir).exists(),
            },
        ]
        return {
            "job_id": job["job_id"],
            "chapter_id": chapter_id,
            "status": job["status"],
            "artifacts": artifacts,
        }

    def _build_page_artifact(self, chapter_id: str, page_num: int, artifact_type: str) -> dict[str, Any]:
        metadata_dir, pass2_dir, masks_dir = self._get_chapter_dirs(chapter_id)
        page = f"page_{page_num:03d}"

        mapping = {
            "metadata": metadata_dir / f"{page}.meta.json",
            "runmeta_pass1": metadata_dir / f"{page}.meta.pass1.runmeta.json",
            "runmeta_pass2": pass2_dir / f"{page}_colorized.runmeta.json",
            "mask": masks_dir / f"{page}_text.png",
            "colorized": pass2_dir / f"{page}_colorized.png",
        }

        file_path = mapping[artifact_type]
        return {
            "chapter_id": chapter_id,
            "page_num": page_num,
            "artifact": artifact_type,
            "path": str(file_path),
            "exists": file_path.exists(),
        }

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        if path == "/version":
            self._send_json(HTTPStatus.OK, {"version": API_VERSION})
            return

        if path == "/openapi.json":
            self._send_json(HTTPStatus.OK, build_openapi_schema())
            return

        job_target = self._extract_job_path(path)
        if job_target:
            job_id, mode = job_target
            job = self.job_store.get_job(job_id)
            if not job:
                self._error(HTTPStatus.NOT_FOUND, "job_not_found", "Requested job was not found", {"job_id": job_id})
                return

            if mode == "status":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "job_id": job["job_id"],
                        "status": job["status"],
                        "progress": job["progress"],
                        "last_update": job["created_at"],
                    },
                )
                return

            if mode == "artifacts":
                self._send_json(HTTPStatus.OK, self._build_artifacts_payload(job))
                return

        chapter_target = self._extract_chapter_path(path)
        if chapter_target:
            chapter_id, mode, page_num = chapter_target
            if mode == "pages":
                self._send_json(HTTPStatus.OK, {"chapter_id": chapter_id, "pages": self._list_chapter_pages(chapter_id)})
                return

            if page_num is not None:
                self._send_json(HTTPStatus.OK, self._build_page_artifact(chapter_id, page_num, mode))
                return

        self._error(HTTPStatus.NOT_FOUND, "not_found", "Endpoint not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/v1/jobs/two-pass":
            self._error(HTTPStatus.NOT_FOUND, "not_found", "Endpoint not found")
            return

        if not self._require_write_token():
            return

        payload = self._read_json_body()
        if payload is None:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_json", "Invalid or missing JSON body")
            return

        style_reference = payload.get("style_reference")
        if not isinstance(style_reference, str) or not style_reference.strip():
            self._error(HTTPStatus.BAD_REQUEST, "style_reference_required", "style_reference is required in this version")
            return

        item = self.job_store.create_job(payload)
        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "job_id": item["job_id"],
                "status": item["status"],
                "created_at": item["created_at"],
            },
        )


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run bootstrap API server."""
    httpd = ThreadingHTTPServer((host, port), MangaFluxAPIHandler)
    print(f"[api] Listening on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
