
"""SQLite-backed one-worker queue.
Durable on local disk, NOT durable on free cloud restarts.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path

from qc_core import (
    ENGINE_VERSION,
    AnalysisError,
    QuotaError,
    inspect,
)

ROOT = Path(
    os.getenv(
        "QC_DATA_DIR",
        "runtime_data",
    )
).resolve()

UPLOADS = ROOT / "uploads"
THUMBS = ROOT / "thumbnails"
DB = ROOT / "queue.sqlite3"

ALLOWED_SUFFIXES = {
    ".mp4",
    ".mov",
    ".webm",
    ".avi",
    ".mkv",
}

MAX_BATCH_BYTES = (
    1024 * 1024 * 1024
)


def connection():
    con = sqlite3.connect(
        DB,
        timeout=25,
    )

    con.row_factory = sqlite3.Row

    con.execute(
        "PRAGMA busy_timeout=25000"
    )

    return con


class JobManager:
    def __init__(self, api_key: str):
        ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        UPLOADS.mkdir(
            parents=True,
            exist_ok=True,
        )

        THUMBS.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.api_key = api_key
        self.lock = threading.RLock()

        self.paused = threading.Event()
        self.paused.set()

        with closing(connection()) as con, con:
            con.execute(
                "PRAGMA journal_mode=WAL"
            )

            con.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              batch TEXT NOT NULL,
              filename TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              path TEXT NOT NULL,
              size INTEGER NOT NULL,
              mode TEXT NOT NULL,
              model TEXT NOT NULL,
              status TEXT NOT NULL,
              verdict TEXT,
              category TEXT,
              reason TEXT,
              details TEXT,
              duration REAL,
              elapsed REAL,
              cache_hit INTEGER DEFAULT 0,
              error TEXT,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS
            jobs_batch_idx
            ON jobs(batch, created_at);

            CREATE INDEX IF NOT EXISTS
            jobs_status_idx
            ON jobs(status, created_at);

            CREATE TABLE IF NOT EXISTS cache (
              cache_key TEXT PRIMARY KEY,
              result_json TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            """)

            con.execute(
                "UPDATE jobs "
                "SET status='QUEUED' "
                "WHERE status='RUNNING'"
            )

        self.worker = threading.Thread(
            target=self._work,
            daemon=True,
            name="qc-worker",
        )

        self.worker.start()

    def update_key(self, key: str):
        if key:
            self.api_key = key

    def batches(self):
        with self.lock, closing(connection()) as con:
            return [
                dict(r)
                for r in con.execute(
                    "SELECT batch, COUNT(*) n, "
                    "MAX(created_at) t "
                    "FROM jobs "
                    "GROUP BY batch "
                    "ORDER BY t DESC"
                )
            ]

    def jobs(self, batch: str):
        with self.lock, closing(connection()) as con:
            return [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM jobs "
                    "WHERE batch=? "
                    "ORDER BY created_at, filename",
                    (batch,),
                )
            ]

    def queue_count(self):
        with self.lock, closing(connection()) as con:
            return con.execute(
                "SELECT COUNT(*) FROM jobs "
                "WHERE status='QUEUED'"
            ).fetchone()[0]

    def submit(
        self,
        uploaded,
        batch: str,
        mode: str,
        model: str,
    ):
        original_name = Path(
            uploaded.name.replace(
                "\\", "/"
            )
        ).name[:180]

        ext = Path(
            original_name
        ).suffix.lower()

        if ext not in ALLOWED_SUFFIXES:
            raise ValueError(
                f"ไม่รองรับไฟล์ {original_name}"
            )

        if (
            not uploaded.size
            or uploaded.size > MAX_BATCH_BYTES
        ):
            raise ValueError(
                "ไฟล์ว่างหรือใหญ่เกิน 1GB"
            )

        fd, temp_name = tempfile.mkstemp(
            prefix="incoming_",
            dir=UPLOADS,
        )

        sha = hashlib.sha256()

        try:
            with os.fdopen(
                fd, "wb"
            ) as out:

                uploaded.seek(0)

                while chunk := uploaded.read(
                    1024 * 1024
                ):
                    out.write(chunk)
                    sha.update(chunk)

            hashed = sha.hexdigest()

            final = UPLOADS / (
                f"{hashed}{ext}"
            )

            if not final.exists():
                os.replace(
                    temp_name,
                    final,
                )
            else:
                os.unlink(
                    temp_name
                )

        finally:
            if os.path.exists(
                temp_name
            ):
                os.unlink(
                    temp_name
                )

        now = time.time()

        with self.lock, closing(connection()) as con, con:
            found = con.execute(
                "SELECT id FROM jobs "
                "WHERE batch=? "
                "AND sha256=? "
                "AND filename=?",
                (
                    batch,
                    hashed,
                    original_name,
                ),
            ).fetchone()

            if found:
                return False

            con.execute("""
              INSERT INTO jobs
              (
                id, batch, filename,
                sha256, path, size,
                mode, model, status,
                created_at, updated_at
              )
              VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                uuid.uuid4().hex,
                batch,
                original_name,
                hashed,
                str(final),
                uploaded.size,
                mode,
                model,
                "QUEUED",
                now,
                now,
            ))

        return True

    def new_batch(self) -> str:
        return (
            time.strftime(
                "%Y%m%d-%H%M%S"
            )
            + "-"
            + uuid.uuid4().hex[:6]
        )

    def start(self):
        self.paused.clear()

    def stop(self):
        self.paused.set()

    def retry(self, batch: str):
        with self.lock, closing(connection()) as con, con:
            con.execute(
                "UPDATE jobs "
                "SET status='QUEUED', "
                "error=NULL, updated_at=? "
                "WHERE batch=? "
                "AND status IN ('ERROR','WAITING')",
                (
                    time.time(),
                    batch,
                ),
            )

        self.paused.clear()

    def _take(self):
        with self.lock, closing(connection()) as con, con:
            row = con.execute(
                "SELECT * FROM jobs "
                "WHERE status='QUEUED' "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()

            if row is None:
                return None

            con.execute(
                "UPDATE jobs "
                "SET status='RUNNING', "
                "updated_at=? "
                "WHERE id=? "
                "AND status='QUEUED'",
                (
                    time.time(),
                    row["id"],
                ),
            )

            return dict(row)

    @staticmethod
    def _cache_key(job):
        return "|".join((
            job["sha256"],
            job["mode"],
            job["model"],
            ENGINE_VERSION,
        ))

    def _process(self, job):
        key = self._cache_key(job)

        with self.lock, closing(connection()) as con:
            old = con.execute(
                "SELECT result_json "
                "FROM cache "
                "WHERE cache_key=?",
                (key,),
            ).fetchone()

        reused = bool(old)

        if old:
            result = json.loads(
                old["result_json"]
            )

            result["elapsed"] = 0.0

        else:
            if not self.api_key:
                raise AnalysisError(
                    "ไม่พบ GEMINI_API_KEY "
                    "ใน Secrets"
                )

            result = inspect(
                Path(job["path"]),
                self.api_key,
                job["model"],
                job["mode"],
                THUMBS / (
                    f"{job['sha256']}.jpg"
                ),
            )

        with self.lock, closing(connection()) as con, con:
            if not old:
                con.execute(
                    "INSERT OR REPLACE "
                    "INTO cache("
                    "cache_key,result_json,created_at"
                    ") VALUES(?,?,?)",
                    (
                        key,
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        ),
                        time.time(),
                    ),
                )

            con.execute("""
                UPDATE jobs
                SET status='DONE',
                verdict=?,
                category=?,
                reason=?,
                details=?,
                duration=?,
                elapsed=?,
                cache_hit=?,
                error=NULL,
                updated_at=?
                WHERE id=?
            """, (
                result["decision"],
                result["video_type"],
                result["reason_th"],
                json.dumps(
                    result,
                    ensure_ascii=False,
                ),
                result["duration"],
                result["elapsed"],
                int(reused),
                time.time(),
                job["id"],
            ))

    def _work(self):
        while True:
            if self.paused.is_set():
                time.sleep(0.7)
                continue

            job = self._take()

            if job is None:
                time.sleep(0.8)
                continue

            try:
                self._process(job)

            except QuotaError as exc:
                self._mark_failure(
                    job,
                    "WAITING",
                    str(exc),
                )

                self.paused.set()

            except Exception as exc:
                self._mark_failure(
                    job,
                    "ERROR",
                    str(exc)[:900],
                )

    def _mark_failure(
        self,
        job,
        status: str,
        reason: str,
    ):
        with self.lock, closing(connection()) as con, con:
            con.execute(
                "UPDATE jobs "
                "SET status=?, "
                "error=?, "
                "updated_at=? "
                "WHERE id=?",
                (
                    status,
                    reason,
                    time.time(),
                    job["id"],
                ),
            )

    def cleanup(
        self,
        older_than_days=7,
    ):
        cutoff = (
            time.time()
            - older_than_days * 86400
        )

        with self.lock, closing(connection()) as con:
            old_paths = [
                r[0]
                for r in con.execute("""
                    SELECT path FROM jobs
                    GROUP BY path
                    HAVING MAX(
                        CASE WHEN status IN (
                            'QUEUED',
                            'RUNNING',
                            'WAITING'
                        )
                        THEN 1 ELSE 0 END
                    ) = 0
                    AND MAX(updated_at) < ?
                """, (cutoff,))
            ]

            deleted = 0

            for path in old_paths:
                p = Path(path)

                if (
                    p.is_file()
                    and p.parent == UPLOADS
                ):
                    p.unlink()
                    deleted += 1

            return deleted

    def disk_usage(self):
        return shutil.disk_usage(ROOT)
