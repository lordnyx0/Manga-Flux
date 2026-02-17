from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PipelineStateStore:
    """SQLite-backed state store for Pass1/Pass2 pipeline execution metadata."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    chapter_id TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    PRIMARY KEY(chapter_id, page_num, stage)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pipeline_stage_status "
                "ON pipeline_state(stage, status)"
            )

    def upsert(
        self,
        chapter_id: str,
        page_num: int,
        stage: str,
        status: str,
        metadata: dict[str, Any],
    ) -> None:
        payload = json.dumps(metadata, ensure_ascii=False)
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_state(chapter_id, page_num, stage, status, metadata_json, updated_at_utc)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_id, page_num, stage) DO UPDATE SET
                    status=excluded.status,
                    metadata_json=excluded.metadata_json,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (chapter_id, int(page_num), stage, status, payload, ts),
            )

    def get(self, chapter_id: str, page_num: int, stage: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status, metadata_json, updated_at_utc
                FROM pipeline_state
                WHERE chapter_id=? AND page_num=? AND stage=?
                """,
                (chapter_id, int(page_num), stage),
            ).fetchone()
        if not row:
            return None
        status, metadata_json, updated_at_utc = row
        return {
            "chapter_id": chapter_id,
            "page_num": int(page_num),
            "stage": stage,
            "status": status,
            "metadata": json.loads(metadata_json),
            "updated_at_utc": updated_at_utc,
        }

    def list_stage(self, stage: str, status: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT chapter_id, page_num, status, metadata_json, updated_at_utc "
            "FROM pipeline_state WHERE stage=?"
        )
        params: list[Any] = [stage]
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY chapter_id, page_num"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        result: list[dict[str, Any]] = []
        for chapter_id, page_num, row_status, metadata_json, updated_at_utc in rows:
            result.append(
                {
                    "chapter_id": chapter_id,
                    "page_num": int(page_num),
                    "stage": stage,
                    "status": row_status,
                    "metadata": json.loads(metadata_json),
                    "updated_at_utc": updated_at_utc,
                }
            )
        return result
