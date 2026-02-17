from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SQLiteVectorStore:
    """Local vector store backed by a single SQLite file."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    chapter_id TEXT,
                    vector BLOB NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_chapter ON embeddings(chapter_id)")

    @staticmethod
    def _tensor_to_blob(tensor: torch.Tensor) -> bytes:
        arr = tensor.detach().cpu().numpy()
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()

    @staticmethod
    def _blob_to_tensor(blob: bytes) -> torch.Tensor:
        arr = np.load(io.BytesIO(blob), allow_pickle=False)
        return torch.from_numpy(arr)

    def upsert(self, vector_id: str, tensor: torch.Tensor, metadata_json: str, chapter_id: str = "") -> None:
        blob = self._tensor_to_blob(tensor)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO embeddings(id, chapter_id, vector, metadata_json)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    chapter_id=excluded.chapter_id,
                    vector=excluded.vector,
                    metadata_json=excluded.metadata_json
                """,
                (vector_id, chapter_id, blob, metadata_json),
            )

    def get(self, vector_id: str) -> tuple[torch.Tensor, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT vector, metadata_json FROM embeddings WHERE id=?",
                (vector_id,),
            ).fetchone()
        if not row:
            return None
        vector_blob, metadata_json = row
        return self._blob_to_tensor(vector_blob), metadata_json

    def list_ids(self, chapter_id: str = "") -> list[str]:
        with self._connect() as conn:
            if chapter_id:
                rows = conn.execute("SELECT id FROM embeddings WHERE chapter_id=?", (chapter_id,)).fetchall()
            else:
                rows = conn.execute("SELECT id FROM embeddings").fetchall()
        return [row[0] for row in rows]

    def delete(self, vector_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM embeddings WHERE id=?", (vector_id,))

    def count(self, chapter_id: str = "") -> int:
        with self._connect() as conn:
            if chapter_id:
                row = conn.execute("SELECT COUNT(*) FROM embeddings WHERE chapter_id=?", (chapter_id,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(row[0]) if row else 0
