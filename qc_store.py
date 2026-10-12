
"""V3.9.1 MP4-only disk queue.

No JSON matching.
No historical verdict cache.
PASS / FAIL / REVIEW.
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
import unicodedata
import uuid

from contextlib import closing
from pathlib import Path

from qc_core import (
    AnalysisError,
    QuotaError,
    DEFAULT_MODEL,
    MODELS,
    inspect,
)


ROOT = Path(
    os.getenv(
        "QC_DATA_DIR",
        "runtime_data",
    )
).resolve()

ACTIVE = ROOT / "json_current_round"
UPLOADS = ACTIVE / "videos"
THUMBS = ACTIVE / "thumbnails"
EXPORTS = ACTIVE / "exports"
DB = ACTIVE / "work.sqlite3"

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv",
    ".avi", ".webm",
}

MAX_VIDEO = 50 * 1024**2
MAX_JSON = 150 * 1024

MAX_VIDEOS = 150
MAX_ACTIVE_BYTES = 1200 * 1024**2
MIN_DISK_FREE = 260 * 1024**2


def _name(value):
    normalized = unicodedata.normalize(
        "NFC",
        str(value or "").replace("\\", "/"),
    )

    return normalized.split("/")[-1].strip()[:190]


def _key(value):
    return _name(value).casefold()


def _db():
    con = sqlite3.connect(
        DB,
        timeout=30,
    )

    con.row_factory = sqlite3.Row

    con.execute(
        "PRAGMA busy_timeout=30000"
    )

    return con


class ActiveQueue:

    def __init__(self, api_key):
        for path in (
            ROOT,
            ACTIVE,
            UPLOADS,
            THUMBS,
            EXPORTS,
        ):
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.api_key = api_key
        self.lock = threading.RLock()

        self.paused = threading.Event()
        self.paused.set()

        self.model_override = DEFAULT_MODEL

        with self.lock, closing(_db()) as con:
            con.execute(
                "PRAGMA journal_mode=WAL"
            )

            con.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name_key TEXT UNIQUE NOT NULL,
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
                    error TEXT,
                    added_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS jobs_status_idx
                ON jobs(status,added_at);

                UPDATE jobs
                SET status='QUEUED'
                WHERE status IN (
                    'RUNNING',
                    'WAIT_JSON'
                );
            """)

            con.commit()

        self.worker = threading.Thread(
            target=self._work,
            daemon=True,
        )

        self.worker.start()

    def update_key(self, api_key):
        self.api_key = api_key

    def set_model(self, model):
        if model not in MODELS:
            raise ValueError(
                "ไม่รู้จักโมเดลนี้"
            )

        with (
            self.lock,
            closing(_db()) as con,
            con,
        ):
            self.model_override = model

            con.execute("""
                UPDATE jobs
                SET model=?,updated_at=?
                WHERE status IN (
                    'QUEUED',
                    'ERROR',
                    'WAITING'
                )
            """, (
                model,
                time.time(),
            ))

    def jobs(self):
        with self.lock, closing(_db()) as con:
            return [
                dict(row)
                for row in con.execute("""
                    SELECT *
                    FROM jobs
                    ORDER BY added_at,id
                """)
            ]

    def stats(self):
        with self.lock, closing(_db()) as con:
            count, used = con.execute("""
                SELECT
                    COUNT(*),
                    COALESCE(SUM(size),0)
                FROM jobs
            """).fetchone()

            return (
                int(count),
                int(used),
                0,
            )

    def unmatched_references(self):
        return []

    def free_disk(self):
        return shutil.disk_usage(
            ROOT
        ).free

    def add_json(
        self,
        raw,
        filename="product.json",
    ):
        raise ValueError(
            "V3.9.1 ใช้ MP4 อย่างเดียว "
            "ไม่รองรับ JSON"
        )

    def add_video(
        self,
        stream,
        filename,
        size,
        mode,
        model,
    ):
        filename = _name(filename)

        if (
            Path(filename).suffix.lower()
            not in VIDEO_EXTS
        ):
            raise ValueError(
                "รองรับไฟล์ MP4/MOV/MKV/AVI/WEBM เท่านั้น"
            )

        try:
            size = int(size)

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "ขนาดไฟล์ไม่ถูกต้อง"
            ) from exc

        if not 0 < size <= MAX_VIDEO:
            raise ValueError(
                "ไฟล์วิดีโอต้องไม่เกิน 50 MB"
            )

        if (
            self.free_disk()
            < MIN_DISK_FREE + size
        ):
            raise ValueError(
                "พื้นที่เซิร์ฟเวอร์ไม่พอ "
                "ต้องล้างรอบหรือใช้ไฟล์ขนาดเล็กลง"
            )

        model = (
            model if model in MODELS
            else DEFAULT_MODEL
        )

        fd, tmp = tempfile.mkstemp(
            dir=UPLOADS,
            prefix="incoming_",
        )

        written = 0
        digest = hashlib.sha256()

        try:
            with os.fdopen(
                fd, "wb"
            ) as output:

                while True:
                    chunk = stream.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    written += len(chunk)

                    if written > MAX_VIDEO:
                        raise ValueError(
                            "ขนาดไฟล์เกิน 50 MB"
                        )

                    output.write(chunk)
                    digest.update(chunk)

            if written != size:
                raise ValueError(
                    "ไฟล์ที่ส่งมาไม่ครบ"
                )

            with (
                self.lock,
                closing(_db()) as con,
                con,
            ):
                con.execute(
                    "BEGIN IMMEDIATE"
                )

                existing = con.execute("""
                    SELECT sha256
                    FROM jobs
                    WHERE name_key=?
                """, (
                    _key(filename),
                )).fetchone()

                if existing:
                    if (
                        existing["sha256"]
                        == digest.hexdigest()
                    ):
                        return False

                    raise ValueError(
                        "ชื่อไฟล์ซ้ำแต่เนื้อหาต่างกัน "
                        "ให้เปลี่ยนชื่อ MP4"
                    )

                count, used = con.execute("""
                    SELECT
                        COUNT(*),
                        COALESCE(SUM(size),0)
                    FROM jobs
                """).fetchone()

                if (
                    count >= MAX_VIDEOS
                    or (
                        used + written
                        > MAX_ACTIVE_BYTES
                    )
                ):
                    raise ValueError(
                        "เกินจำนวน 150 คลิป "
                        "หรือพื้นที่รวม 1200 MB"
                    )

                job_id = uuid.uuid4().hex

                dest = UPLOADS / (
                    job_id
                    + Path(filename).suffix.lower()
                )

                os.replace(
                    tmp, dest
                )

                now = time.time()

                con.execute("""
                    INSERT INTO jobs(
                        id,
                        name_key,
                        filename,
                        sha256,
                        path,
                        size,
                        mode,
                        model,
                        status,
                        added_at,
                        updated_at
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    job_id,
                    _key(filename),
                    filename,
                    digest.hexdigest(),
                    str(dest),
                    written,
                    (
                        mode
                        if mode in (
                            "Fast",
                            "Balanced",
                            "Deep",
                        )
                        else "Balanced"
                    ),
                    model,
                    "QUEUED",
                    now,
                    now,
                ))

            return True

        finally:
            Path(tmp).unlink(
                missing_ok=True
            )

    def start(self):
        self.paused.clear()

    def pause(self):
        self.paused.set()

    def retry(self, model=None):
        if model is not None:
            self.set_model(model)

        with (
            self.lock,
            closing(_db()) as con,
            con,
        ):
            con.execute("""
                UPDATE jobs
                SET
                    status='QUEUED',
                    error=NULL,
                    verdict=NULL,
                    reason=NULL,
                    details=NULL,
                    updated_at=?
                WHERE status IN (
                    'ERROR',
                    'WAITING'
                )
            """, (
                time.time(),
            ))

        self.start()

    def _take(self):
        with (
            self.lock,
            closing(_db()) as con,
            con,
        ):
            if self.paused.is_set():
                return None

            con.execute(
                "BEGIN IMMEDIATE"
            )

            row = con.execute("""
                SELECT *
                FROM jobs
                WHERE status='QUEUED'
                ORDER BY added_at,id
                LIMIT 1
            """).fetchone()

            if row is None:
                return None

            con.execute("""
                UPDATE jobs
                SET
                    status='RUNNING',
                    updated_at=?
                WHERE id=?
            """, (
                time.time(),
                row["id"],
            ))

            return dict(row)

    def _mark_failure(
        self,
        job_id,
        status,
        error,
    ):
        with (
            self.lock,
            closing(_db()) as con,
            con,
        ):
            con.execute("""
                UPDATE jobs
                SET
                    status=?,
                    error=?,
                    updated_at=?
                WHERE id=?
                AND status='RUNNING'
            """, (
                status,
                str(error)[:700],
                time.time(),
                job_id,
            ))

    def _work(self):
        while True:
            if self.paused.is_set():
                time.sleep(0.5)
                continue

            job = self._take()

            if job is None:
                time.sleep(0.6)
                continue

            try:
                if not self.api_key:
                    raise AnalysisError(
                        "กรุณาตั้ง GEMINI_API_KEY"
                    )

                result = inspect(
                    job["path"],
                    self.api_key,
                    job["model"],
                    job["mode"],
                    THUMBS / (
                        f"{job['sha256']}.jpg"
                    ),
                    reference="",
                )

                if result.get("decision") not in (
                    "PASS",
                    "FAIL",
                    "REVIEW",
                ):
                    raise AnalysisError(
                        "ตัวตรวจส่งผลไม่ถูกต้อง"
                    )

                with (
                    self.lock,
                    closing(_db()) as con,
                    con,
                ):
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
                            error=NULL,
                            updated_at=?
                        WHERE id=?
                        AND status='RUNNING'
                    """, (
                        result["decision"],
                        result.get(
                            "product_type_th", ""
                        ),
                        result.get(
                            "reason_th", ""
                        ),
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        ),
                        result.get("duration"),
                        result.get("elapsed"),
                        time.time(),
                        job["id"],
                    ))

            except QuotaError as exc:
                self._mark_failure(
                    job["id"],
                    "WAITING",
                    str(exc),
                )

                self.pause()

            except Exception as exc:
                self._mark_failure(
                    job["id"],
                    "ERROR",
                    str(exc),
                )

    def clear_round(self):
        self.pause()

        with (
            self.lock,
            closing(_db()) as con,
            con,
        ):
            con.execute(
                "BEGIN IMMEDIATE"
            )

            count = con.execute("""
                SELECT COUNT(*)
                FROM jobs
                WHERE status='RUNNING'
            """).fetchone()[0]

            if count:
                raise ValueError(
                    "ยังมีคลิปกำลังตรวจ "
                    "รอให้จบก่อนล้างรอบ"
                )

            con.execute(
                "DELETE FROM jobs"
            )

            exists = con.execute("""
                SELECT 1
                FROM sqlite_master
                WHERE type='table'
                AND name='product_refs'
            """).fetchone()

            if exists:
                con.execute(
                    "DELETE FROM product_refs"
                )

            for folder in (
                UPLOADS,
                THUMBS,
                EXPORTS,
            ):
                for path in folder.iterdir():
                    if path.is_file():
                        path.unlink(
                            missing_ok=True
                        )

        with self.lock, closing(_db()) as con:
            con.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            )

    def clear_legacy(self):
        self.pause()

        for p in (
            ROOT / "uploads",
            ROOT / "thumbnails",
            ROOT / "active_only",
        ):
            if p.is_dir():
                shutil.rmtree(p)
