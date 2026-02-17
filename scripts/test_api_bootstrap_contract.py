#!/usr/bin/env python3
"""Contract smoke tests for Manga-Flux bootstrap API.

Runs a temporary local API server and validates core endpoint contracts:
- health/version
- auth behavior for write route
- job creation and retrieval
- artifacts and chapter/page endpoints payload shape
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
PORT = 8096
BASE = f"http://{HOST}:{PORT}"
TOKEN = "contract-token"


def _request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE}{path}",
        method=method,
        headers=req_headers,
        data=data,
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _wait_server() -> None:
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            status, body = _request("GET", "/healthz")
            if status == 200 and body.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("API server did not start in time")


def main() -> int:
    store_path = Path("outputs/api/contract_test_jobs.json")
    if store_path.exists():
        store_path.unlink()

    env = os.environ.copy()
    env["MANGA_FLUX_API_TOKEN"] = TOKEN
    env["MANGA_FLUX_JOB_STORE"] = str(store_path)

    proc = subprocess.Popen(
        [sys.executable, "run_api_local.py", "--host", HOST, "--port", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    try:
        _wait_server()

        status, body = _request("GET", "/version")
        _assert(status == 200, "GET /version should return 200")
        _assert("version" in body, "GET /version should include version field")

        status, body = _request("GET", "/openapi.json")
        _assert(status == 200, "GET /openapi.json should return 200")
        _assert(body.get("openapi") == "3.0.3", "OpenAPI schema should declare openapi=3.0.3")

        status, body = _request(
            "POST",
            "/v1/jobs/two-pass",
            payload={"chapter_id": "chapter_001", "style_reference": "data/style_ref.png"},
        )
        _assert(status == 401, "POST /v1/jobs/two-pass without token should return 401")
        _assert(body.get("code") == "unauthorized", "401 payload should contain code=unauthorized")
        _assert("trace_id" in body, "401 payload should include trace_id")

        status, body = _request(
            "POST",
            "/v1/jobs/two-pass",
            payload={"chapter_id": "chapter_001"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        _assert(status == 400, "POST /v1/jobs/two-pass without style_reference should return 400")
        _assert(body.get("code") == "style_reference_required", "Missing style should return style_reference_required")

        status, body = _request(
            "POST",
            "/v1/jobs/two-pass",
            payload={
                "chapter_id": "chapter_001",
                "style_reference": "data/style_ref.png",
                "metadata_output": "metadata",
                "pass2_output": "outputs/pass2",
                "masks_output": "outputs/pass1/masks",
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        _assert(status == 202, "POST /v1/jobs/two-pass should return 202 with valid token/payload")
        _assert(body.get("status") == "queued", "Created job should be queued")
        job_id = body.get("job_id")
        _assert(isinstance(job_id, str) and job_id.startswith("job_"), "job_id should be present")

        status, body = _request("GET", f"/v1/jobs/{job_id}")
        _assert(status == 200, "GET /v1/jobs/{job_id} should return 200")
        _assert(body.get("job_id") == job_id, "Job status payload should match job_id")

        status, body = _request("GET", f"/v1/jobs/{job_id}/artifacts")
        _assert(status == 200, "GET /v1/jobs/{job_id}/artifacts should return 200")
        _assert(isinstance(body.get("artifacts"), list), "Artifacts payload should contain artifacts list")

        status, body = _request("GET", "/v1/chapters/chapter_001/pages")
        _assert(status == 200, "GET chapter pages should return 200")
        _assert("pages" in body, "Chapter pages payload should include pages")

        status, body = _request("GET", "/v1/chapters/chapter_001/pages/1/metadata")
        _assert(status == 200, "GET page metadata locator should return 200")
        _assert(body.get("artifact") == "metadata", "metadata locator should mark artifact=metadata")

        print("[OK] API bootstrap contract smoke passed")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
