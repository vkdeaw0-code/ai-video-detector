
"""AI Video QC — Persistent Batch Queue V3.

Uses local SQLite for batches, jobs and cached results.
Streamlit Community Cloud does not guarantee permanent storage.
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
    os.getenv("QC_DATA_DIR", "runtime_data")
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

MAX_BATCH_BYTES = 1024**3


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
        for p in (
            ROOT,
            UPLOADS,
            THUMBS,
        ):
            p.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.api_key = api_key
        self.lock = threading.RLock()
        self.paused = threading.Event()
        self.paused.set()

        with self.lock, closing(connection()) as con:
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

            CREATE INDEX IF NOT EXISTS jobs_batch_idx
              ON jobs(batch, created_at);

            CREATE INDEX IF NOT EXISTS jobs_status_idx
              ON jobs(status, created_at);

            CREATE INDEX IF NOT EXISTS jobs_hash_in_batch
              ON jobs(batch, sha256);

            CREATE TABLE IF NOT EXISTS cache (
              cache_key TEXT PRIMARY KEY,
              result_json TEXT NOT NULL,
              created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS batches (
              id TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              state TEXT NOT NULL DEFAULT 'OPEN',
              created_at REAL NOT NULL,
              closed_at REAL
            );
            """)

            # ย้ายประวัติจากเวอร์ชันเก่าเข้าระบบรอบใหม่
            con.execute("""
              INSERT OR IGNORE INTO batches(
                id, label, state, created_at
              )
              SELECT
                batch,
                'รอบเดิม • ' || SUBSTR(batch,1,15),
                'OPEN',
                MIN(created_at)
              FROM jobs
              GROUP BY batch
            """)

            # งานที่ถูกขัดจังหวะระหว่างตรวจกลับมารอคิว
            con.execute("""
              UPDATE jobs
              SET status='QUEUED'
              WHERE status='RUNNING'
            """)

            con.commit()

        self.worker = threading.Thread(
            target=self._work,
            daemon=True,
            name="qc-worker",
        )
        self.worker.start()

    def update_key(self, key: str):
        if key:
            self.api_key = key

    def create_batch(self, label: str = "") -> str:
        now = time.time()

        with self.lock, closing(connection()) as con, con:
            count = con.execute(
                "SELECT COUNT(*) FROM batches"
            ).fetchone()[0]

            batch_id = (
                time.strftime("%Y%m%d-%H%M%S")
                + "-"
                + uuid.uuid4().hex[:6]
            )

            clean = (
                label.strip()[:70]
                or f"รอบที่ {count + 1:03d}"
            )

            con.execute("""
              INSERT INTO batches(
                id, label, state, created_at
              )
              VALUES(?,?,?,?)
            """, (
                batch_id,
                clean,
                "OPEN",
                now,
            ))

        return batch_id

    def batches(self):
        with self.lock, closing(connection()) as con:
            return [
                dict(r)
                for r in con.execute("""
                  SELECT
                    b.id AS batch,
                    b.label,
                    b.state,
                    b.created_at,
                    b.closed_at,
                    COUNT(j.id) AS n,
                    COALESCE(SUM(j.size),0) AS bytes,
                    COALESCE(SUM(
                      CASE WHEN j.status='DONE'
                      THEN 1 ELSE 0 END
                    ),0) AS done,
                    COALESCE(SUM(
                      CASE WHEN j.verdict='PASS'
                      THEN 1 ELSE 0 END
                    ),0) AS passed,
                    COALESCE(SUM(
                      CASE WHEN j.verdict='FAIL'
                      THEN 1 ELSE 0 END
                    ),0) AS failed,
                    COALESCE(SUM(
                      CASE WHEN j.status
                      IN ('ERROR','WAITING')
                      THEN 1 ELSE 0 END
                    ),0) AS errors
                  FROM batches b
                  LEFT JOIN jobs j ON j.batch=b.id
                  GROUP BY b.id
                  ORDER BY b.created_at DESC
                """)
            ]

    def batch(self, batch_id: str):
        for b in self.batches():
            if b["batch"] == batch_id:
                return b
        return None

    def rename_batch(self, batch_id: str, label: str):
        clean = label.strip()[:70]

        if not clean:
            raise ValueError("กรุณาตั้งชื่อรอบ")

        with self.lock, closing(connection()) as con, con:
            con.execute(
                "UPDATE batches SET label=? WHERE id=?",
                (clean, batch_id),
            )

    def close_batch(self, batch_id: str):
        with self.lock, closing(connection()) as con, con:
            batch = con.execute(
                "SELECT state FROM batches WHERE id=?",
                (batch_id,),
            ).fetchone()

            if not batch or batch["state"] != "OPEN":
                raise ValueError(
                    "รอบนี้ถูกปิดไปแล้ว"
                )

            total, pending = con.execute("""
              SELECT
                COUNT(*),
                COALESCE(SUM(
                  CASE WHEN status!='DONE'
                  THEN 1 ELSE 0 END
                ),0)
              FROM jobs
              WHERE batch=?
            """, (batch_id,)).fetchone()

            if total == 0 or pending:
                raise ValueError(
                    "ปิดรอบได้เมื่อตรวจครบทุกคลิปแล้วเท่านั้น"
                )

            con.execute("""
              UPDATE batches
              SET state='CLOSED', closed_at=?
              WHERE id=?
            """, (
                time.time(),
                batch_id,
            ))

    def jobs(self, batch_id: str):
        with self.lock, closing(connection()) as con:
            return [
                dict(r)
                for r in con.execute("""
                  SELECT *
                  FROM jobs
                  WHERE batch=?
                  ORDER BY created_at,id
                """, (batch_id,))
            ]

    def queue_count(self):
        with self.lock, closing(connection()) as con:
            return con.execute("""
              SELECT COUNT(*)
              FROM jobs
              WHERE status='QUEUED'
            """).fetchone()[0]

    def submit(
        self,
        uploaded,
        batch_id: str,
        mode: str,
        model: str,
    ) -> bool:
        """Add file to batch. False means duplicate."""

        original = Path(
            str(uploaded.name).replace("\\", "/")
        ).name[:180]

        ext = Path(original).suffix.lower()
        size = int(uploaded.size)

        if (
            ext not in ALLOWED_SUFFIXES
            or not original
        ):
            raise ValueError(
                f"ไม่รองรับชนิดไฟล์ {original}"
            )

        if not (0 < size <= MAX_BATCH_BYTES):
            raise ValueError(
                "ไฟล์ว่างหรือใหญ่เกิน 1GB"
            )

        fd, temp_name = tempfile.mkstemp(
            prefix="incoming_",
            dir=UPLOADS,
        )

        digest = hashlib.sha256()
        written = 0

        try:
            with os.fdopen(fd, "wb") as out:
                uploaded.seek(0)

                while chunk := uploaded.read(1024 * 1024):
                    written += len(chunk)
                    out.write(chunk)
                    digest.update(chunk)

            if written != size:
                raise ValueError(
                    "ข้อมูลไฟล์อัปโหลดไม่ครบ กรุณาลองใหม่"
                )

            sha = digest.hexdigest()
            saved = UPLOADS / f"{sha}{ext}"

            with self.lock, closing(connection()) as con, con:
                con.execute("BEGIN IMMEDIATE")

                batch = con.execute(
                    "SELECT state FROM batches WHERE id=?",
                    (batch_id,),
                ).fetchone()

                if (
                    not batch
                    or batch["state"] != "OPEN"
                ):
                    raise ValueError(
                        "รอบนี้ปิดแล้ว โปรดสร้างรอบใหม่"
                    )

                existing = con.execute("""
                  SELECT 1 FROM jobs
                  WHERE batch=? AND sha256=?
                """, (
                    batch_id,
                    sha,
                )).fetchone()

                if existing:
                    return False

                used = con.execute("""
                  SELECT COALESCE(SUM(size),0)
                  FROM jobs
                  WHERE batch=?
                """, (batch_id,)).fetchone()[0]

                if used + size > MAX_BATCH_BYTES:
                    raise ValueError(
                        "ข้อมูลรวมในรอบเกิน 1GB "
                        "ให้สร้างรอบใหม่"
                    )

                if (
                    shutil.disk_usage(ROOT).free
                    < size + 200 * 1024**2
                ):
                    raise ValueError(
                        "พื้นที่ว่างบนเซิร์ฟเวอร์"
                        "เหลือน้อยเกินไป"
                    )

                if not saved.exists():
                    os.replace(
                        temp_name,
                        saved,
                    )

                con.execute("""
                  INSERT INTO jobs(
                    id,
                    batch,
                    filename,
                    sha256,
                    path,
                    size,
                    mode,
                    model,
                    status,
                    created_at,
                    updated_at
                  )
                  VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    uuid.uuid4().hex,
                    batch_id,
                    original,
                    sha,
                    str(saved),
                    size,
                    mode,
                    model,
                    "QUEUED",
                    time.time(),
                    time.time(),
                ))

            return True

        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def start(self):
        self.paused.clear()

    def stop(self):
        self.paused.set()

    def retry(self, batch_id: str):
        with self.lock, closing(connection()) as con, con:
            con.execute("""
              UPDATE jobs
              SET
                status='QUEUED',
                error=NULL,
                updated_at=?
              WHERE batch=?
              AND status IN ('ERROR','WAITING')
            """, (
                time.time(),
                batch_id,
            ))

        self.paused.clear()

    def _take(self):
        with self.lock, closing(connection()) as con, con:
            con.execute("BEGIN IMMEDIATE")

            job = con.execute("""
              SELECT *
              FROM jobs
              WHERE status='QUEUED'
              ORDER BY created_at
              LIMIT 1
            """).fetchone()

            if job is None:
                return None

            con.execute("""
              UPDATE jobs
              SET status='RUNNING', updated_at=?
              WHERE id=? AND status='QUEUED'
            """, (
                time.time(),
                job["id"],
            ))

            return dict(job)

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
            hit = con.execute("""
              SELECT result_json
              FROM cache
              WHERE cache_key=?
            """, (key,)).fetchone()

        if hit:
            result = json.loads(
                hit["result_json"]
            )
            result["elapsed"] = 0.0

        else:
            if not self.api_key:
                raise AnalysisError(
                    "ไม่พบ GEMINI_API_KEY ใน Secrets"
                )

            result = inspect(
                Path(job["path"]),
                self.api_key,
                job["model"],
                job["mode"],
                THUMBS / f"{job['sha256']}.jpg",
            )

        with self.lock, closing(connection()) as con, con:
            if not hit:
                con.execute("""
                  INSERT OR REPLACE
                  INTO cache(
                    cache_key,
                    result_json,
                    created_at
                  )
                  VALUES(?,?,?)
                """, (
                    key,
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                    time.time(),
                ))

            con.execute("""
              UPDATE jobs
              SET
                status='DONE',
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
                int(bool(hit)),
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

    def _mark_failure(self, job, status, error):
        with self.lock, closing(connection()) as con, con:
            con.execute("""
              UPDATE jobs
              SET
                status=?,
                error=?,
                updated_at=?
              WHERE id=?
            """, (
                status,
                error,
                time.time(),
                job["id"],
            ))

    def cleanup(self, older_than_days=7):
        cutoff = (
            time.time()
            - older_than_days * 86400
        )

        with self.lock, closing(connection()) as con:
            old_paths = [
                r[0]
                for r in con.execute("""
                  SELECT path
                  FROM jobs
                  GROUP BY path
                  HAVING
                    MAX(CASE
                      WHEN status IN (
                        'QUEUED',
                        'RUNNING',
                        'WAITING'
                      )
                      THEN 1 ELSE 0
                    END)=0
                  AND MAX(updated_at) < ?
                """, (cutoff,))
            ]

            removed = 0

            for file in old_paths:
                p = Path(file)

                if (
                    p.is_file()
                    and p.parent == UPLOADS
                ):
                    p.unlink()
                    removed += 1

            return removed

    def disk_usage(self):
        return shutil.disk_usage(ROOT)
