
"""V3.9 Hybrid queue.

MP4 alone or MP4 with optional product JSON.
Current round only. PASS / FAIL / REVIEW.
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
import zipfile

from contextlib import closing
from pathlib import Path

from qc_core import AnalysisError, QuotaError, inspect


ROOT = Path(
    os.getenv("QC_DATA_DIR", "runtime_data")
).resolve()

ACTIVE = ROOT / "json_current_round"
UPLOADS = ACTIVE / "videos"
THUMBS = ACTIVE / "thumbnails"
EXPORTS = ACTIVE / "exports"
DB = ACTIVE / "work.sqlite3"

VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}

MAX_VIDEO = 50 * 1024**2
MAX_ZIP = 100 * 1024**2
MAX_JSON = 150 * 1024
MAX_ZIP_EXPANDED = 250 * 1024**2

MAX_ACTIVE_BYTES = 1200 * 1024**2
MAX_VIDEOS = 150
MIN_DISK_FREE = 260 * 1024**2


def _db():
    conn = sqlite3.connect(
        DB, timeout=30
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA busy_timeout=30000"
    )

    return conn


def _name(value):
    raw = unicodedata.normalize(
        "NFC",
        str(value or "").replace("\\", "/"),
    )

    return raw.split("/")[-1].strip()[:190]


def _key(value):
    return _name(value).casefold()


def _reference_text(data):
    title = str(
        data.get("title", "")
    ).strip()[:370]

    caption = str(
        data.get("caption", "")
    ).strip()[:290]

    extras = {}

    for field in (
        "size",
        "quantity",
        "capacity",
        "category",
        "specifications",
    ):
        value = data.get(field)

        if (
            value is not None
            and value != ""
        ):
            extras[field] = value

    specs = json.dumps(
        extras,
        ensure_ascii=False,
        default=str,
    )[:260]

    return (
        f"ชื่อสินค้าที่ขายจริง: {title}\n"
        f"สเปกที่ร้านระบุ: {specs}\n"
        f"รายละเอียดโฆษณาที่ให้มา: {caption}"
    )[:890]


class ActiveQueue:

    def __init__(self, api_key):
        for folder in (
            ACTIVE,
            UPLOADS,
            THUMBS,
            EXPORTS,
        ):
            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.api_key = api_key

        self.lock = threading.RLock()

        self.paused = threading.Event()
        self.paused.set()

        with self.lock, closing(_db()) as conn:
            conn.execute(
                "PRAGMA journal_mode=WAL"
            )

            conn.executescript("""
                CREATE TABLE IF NOT EXISTS product_refs (
                    name_key TEXT PRIMARY KEY,
                    video_filename TEXT NOT NULL,
                    json_filename TEXT NOT NULL,
                    title TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    product_link TEXT NOT NULL,
                    reference_text TEXT NOT NULL,
                    json_digest TEXT NOT NULL,
                    added_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name_key TEXT NOT NULL UNIQUE,
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
                ON jobs(status, added_at);

                UPDATE jobs
                SET status='QUEUED'
                WHERE status IN ('RUNNING','WAIT_JSON');
            """)

            conn.commit()

        self.worker = threading.Thread(
            target=self._work,
            daemon=True,
        )

        self.worker.start()

    def update_key(self, api_key):
        self.api_key = api_key

    # ==========================================
    # QUERY CURRENT ROUND
    # ==========================================

    def jobs(self):
        with self.lock, closing(_db()) as conn:
            return [
                dict(row)
                for row in conn.execute("""
                    SELECT
                        j.*,
                        p.title AS product_title,
                        p.json_filename AS matched_json,
                        p.reference_text AS product_reference
                    FROM jobs j
                    LEFT JOIN product_refs p
                    ON j.name_key=p.name_key
                    ORDER BY j.added_at,j.id
                """)
            ]

    def stats(self):
        with self.lock, closing(_db()) as conn:
            count, used = conn.execute("""
                SELECT
                    COUNT(*),
                    COALESCE(SUM(size),0)
                FROM jobs
            """).fetchone()

            references = conn.execute("""
                SELECT COUNT(*)
                FROM product_refs
            """).fetchone()[0]

            return (
                int(count),
                int(used),
                int(references),
            )

    def unmatched_references(self):
        with self.lock, closing(_db()) as conn:
            return [
                dict(row)
                for row in conn.execute("""
                    SELECT
                        p.video_filename,
                        p.title
                    FROM product_refs p
                    LEFT JOIN jobs j
                    ON p.name_key=j.name_key
                    WHERE j.id IS NULL
                    ORDER BY p.added_at
                """)
            ]

    def free_disk(self):
        return shutil.disk_usage(ROOT).free

    # ==========================================
    # JSON OPTIONAL
    # ==========================================

    def add_json(
        self,
        raw,
        filename="product.json",
    ):
        if not 0 < len(raw) <= MAX_JSON:
            raise ValueError(
                "JSON ต้องมีขนาด 1–150 KB"
            )

        try:
            data = json.loads(
                raw.decode("utf-8-sig")
            )

        except (
            UnicodeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"อ่าน JSON ไม่สำเร็จ: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "JSON ต้องเป็น object "
                "มี title และ video_filename"
            )

        video_name = _name(
            data.get("video_filename", "")
        )

        if (
            not video_name
            or Path(video_name).suffix.lower()
            not in VIDEO_EXTS
        ):
            raise ValueError(
                "ต้องมี video_filename "
                "ลงท้ายด้วยไฟล์วิดีโอ"
            )

        title = data.get("title")

        if (
            not isinstance(title, str)
            or not title.strip()
        ):
            raise ValueError(
                "JSON ต้องมี title"
            )

        caption = data.get(
            "caption", ""
        )

        link = data.get(
            "product_link", ""
        )

        if (
            not isinstance(caption, str)
            or not isinstance(link, str)
        ):
            raise ValueError(
                "caption และ product_link "
                "ต้องเป็นข้อความ"
            )

        name_key = _key(video_name)

        reference_text = _reference_text(
            data
        )

        digest = hashlib.sha256(
            reference_text.encode("utf-8")
        ).hexdigest()

        with (
            self.lock,
            closing(_db()) as conn,
            conn,
        ):
            conn.execute(
                "BEGIN IMMEDIATE"
            )

            existing = conn.execute("""
                SELECT json_digest
                FROM product_refs
                WHERE name_key=?
            """, (
                name_key,
            )).fetchone()

            if (
                existing
                and existing["json_digest"] == digest
            ):
                return False

            conn.execute("""
                INSERT INTO product_refs(
                    name_key,
                    video_filename,
                    json_filename,
                    title,
                    caption,
                    product_link,
                    reference_text,
                    json_digest,
                    added_at
                )
                VALUES(?,?,?,?,?,?,?,?,?)

                ON CONFLICT(name_key)
                DO UPDATE SET
                    video_filename=excluded.video_filename,
                    json_filename=excluded.json_filename,
                    title=excluded.title,
                    caption=excluded.caption,
                    product_link=excluded.product_link,
                    reference_text=excluded.reference_text,
                    json_digest=excluded.json_digest,
                    added_at=excluded.added_at
            """, (
                name_key,
                video_name,
                _name(filename),
                title[:500],
                caption[:1000],
                link[:500],
                reference_text,
                digest,
                time.time(),
            ))

            # Completed MP4 becomes QUEUED again.
            # For RUNNING clips, revision is checked
            # when the current AI call finishes.

            conn.execute("""
                UPDATE jobs
                SET
                    status='QUEUED',
                    verdict=NULL,
                    category=NULL,
                    reason=NULL,
                    details=NULL,
                    duration=NULL,
                    elapsed=NULL,
                    error=NULL,
                    updated_at=?
                WHERE name_key=?
                AND status!='RUNNING'
            """, (
                time.time(),
                name_key,
            ))

        return True

    # ==========================================
    # VIDEO OPTIONAL REFERENCE
    # ==========================================

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
                "ไม่รองรับไฟล์วิดีโอนี้"
            )

        if not 0 < size <= MAX_VIDEO:
            raise ValueError(
                "วิดีโอต้องไม่เกิน 50MB"
            )

        fd, temp = tempfile.mkstemp(
            prefix="incoming_",
            dir=UPLOADS,
        )

        digest = hashlib.sha256()
        written = 0

        try:
            with os.fdopen(fd, "wb") as output:
                while True:
                    chunk = stream.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    written += len(chunk)

                    if written > MAX_VIDEO:
                        raise ValueError(
                            "วิดีโอเกิน 50MB"
                        )

                    output.write(chunk)
                    digest.update(chunk)

            if written != size:
                raise ValueError(
                    "ไฟล์อัปโหลดไม่ครบ"
                )

            sha = digest.hexdigest()
            name_key = _key(filename)

            with (
                self.lock,
                closing(_db()) as conn,
                conn,
            ):
                conn.execute(
                    "BEGIN IMMEDIATE"
                )

                old = conn.execute("""
                    SELECT sha256
                    FROM jobs
                    WHERE name_key=?
                """, (
                    name_key,
                )).fetchone()

                if old:
                    if old["sha256"] == sha:
                        return False

                    raise ValueError(
                        "ชื่อ MP4 ซ้ำแต่เนื้อหาต่างกัน "
                        "ให้เปลี่ยนชื่อไฟล์"
                    )

                count, used = conn.execute("""
                    SELECT
                        COUNT(*),
                        COALESCE(SUM(size),0)
                    FROM jobs
                """).fetchone()

                if (
                    count >= MAX_VIDEOS
                    or used + written
                    > MAX_ACTIVE_BYTES
                ):
                    raise ValueError(
                        "รอบนี้เต็ม 150 คลิป "
                        "หรือพื้นที่รวม 1200MB"
                    )

                if (
                    self.free_disk()
                    < MIN_DISK_FREE
                ):
                    raise ValueError(
                        "ดิสก์เซิร์ฟเวอร์ไม่พอ"
                    )

                job_id = uuid.uuid4().hex

                destination = UPLOADS / (
                    job_id
                    + Path(filename).suffix.lower()
                )

                os.replace(
                    temp,
                    destination,
                )

                now = time.time()

                # KEY CHANGE:
                # MP4 is QUEUED with or without JSON.

                conn.execute("""
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
                    name_key,
                    filename,
                    sha,
                    str(destination),
                    written,
                    mode,
                    model,
                    "QUEUED",
                    now,
                    now,
                ))

            return True

        finally:
            Path(temp).unlink(
                missing_ok=True
            )

    # ==========================================
    # LEGACY SINGLE / ZIP UPLOAD
    # ==========================================

    def add_upload(
        self,
        uploaded,
        mode,
        model,
    ):
        name = _name(uploaded.name)
        ext = Path(name).suffix.lower()

        result = {
            "video": 0,
            "json": 0,
            "duplicate": 0,
            "errors": [],
        }

        def load_json(
            stream,
            json_name,
            expected,
        ):
            if not 0 < expected <= MAX_JSON:
                raise ValueError(
                    "JSON เกิน 150KB หรือว่าง"
                )

            raw = stream.read(
                MAX_JSON + 1
            )

            if len(raw) != expected:
                raise ValueError(
                    "JSON อัปโหลดไม่ครบ"
                )

            added = self.add_json(
                raw,
                json_name,
            )

            result[
                "json" if added else "duplicate"
            ] += 1

        def load_video(
            stream,
            video_name,
            expected,
        ):
            added = self.add_video(
                stream,
                video_name,
                expected,
                mode,
                model,
            )

            result[
                "video" if added else "duplicate"
            ] += 1

        if ext != ".zip":
            try:
                uploaded.seek(0)

                if ext == ".json":
                    load_json(
                        uploaded,
                        name,
                        int(uploaded.size),
                    )

                elif ext in VIDEO_EXTS:
                    load_video(
                        uploaded,
                        name,
                        int(uploaded.size),
                    )

                else:
                    raise ValueError(
                        "รองรับ JSON, MP4 หรือ ZIP"
                    )

            except Exception as exc:
                result["errors"].append(
                    f"{name}: {exc}"
                )

            return result

        if uploaded.size > MAX_ZIP:
            result["errors"].append(
                "ZIP ต้องไม่เกิน 100MB"
            )

            return result

        try:
            uploaded.seek(0)

            with zipfile.ZipFile(
                uploaded
            ) as archive:
                files = [
                    item
                    for item in archive.infolist()
                    if (
                        not item.is_dir()
                        and not item.filename.startswith(
                            "__MACOSX/"
                        )
                    )
                ]

                total = sum(
                    item.file_size
                    for item in files
                )

                if (
                    len(files) > 250
                    or total > MAX_ZIP_EXPANDED
                ):
                    raise ValueError(
                        "ZIP มีไฟล์มาก "
                        "หรือขนาดหลังแตกเกิน 250MB"
                    )

                # JSON before videos.
                files.sort(
                    key=lambda item: (
                        0
                        if Path(
                            item.filename
                        ).suffix.lower() == ".json"
                        else 1
                    )
                )

                for item in files:
                    child = _name(
                        item.filename
                    )

                    suffix = Path(
                        child
                    ).suffix.lower()

                    if (
                        suffix not in VIDEO_EXTS
                        and suffix != ".json"
                    ):
                        continue

                    try:
                        if item.flag_bits & 1:
                            raise ValueError(
                                "ZIP มีรหัสผ่าน"
                            )

                        if (
                            suffix == ".json"
                            and item.file_size > MAX_JSON
                        ):
                            raise ValueError(
                                "JSON เกิน 150KB"
                            )

                        if (
                            suffix in VIDEO_EXTS
                            and item.file_size > MAX_VIDEO
                        ):
                            raise ValueError(
                                "วิดีโอเกิน 50MB"
                            )

                        with archive.open(
                            item
                        ) as source:
                            if suffix == ".json":
                                load_json(
                                    source,
                                    child,
                                    item.file_size,
                                )

                            else:
                                load_video(
                                    source,
                                    child,
                                    item.file_size,
                                )

                    except Exception as exc:
                        result["errors"].append(
                            f"{child}: {exc}"
                        )

        except (
            OSError,
            zipfile.BadZipFile,
            ValueError,
        ) as exc:
            result["errors"].append(
                f"เปิด ZIP ไม่ได้: {exc}"
            )

        return result

    # ==========================================
    # QUEUE CONTROL
    # ==========================================

    def start(self):
        self.paused.clear()

    def pause(self):
        self.paused.set()

    def retry(self):
        with (
            self.lock,
            closing(_db()) as conn,
            conn,
        ):
            conn.execute("""
                UPDATE jobs
                SET
                    status='QUEUED',
                    error=NULL,
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
            closing(_db()) as conn,
            conn,
        ):
            if self.paused.is_set():
                return None

            conn.execute(
                "BEGIN IMMEDIATE"
            )

            # LEFT JOIN: JSON is optional.
            row = conn.execute("""
                SELECT
                    j.*,
                    COALESCE(
                        p.reference_text,
                        ''
                    ) AS reference_text,
                    COALESCE(
                        p.json_digest,
                        ''
                    ) AS ref_digest
                FROM jobs j
                LEFT JOIN product_refs p
                ON j.name_key=p.name_key
                WHERE j.status='QUEUED'
                ORDER BY j.added_at,j.id
                LIMIT 1
            """).fetchone()

            if not row:
                return None

            conn.execute("""
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

    # ==========================================
    # BACKGROUND WORKER
    # ==========================================

    def _work(self):
        while True:
            if self.paused.is_set():
                time.sleep(0.6)
                continue

            task = self._take()

            if not task:
                time.sleep(0.8)
                continue

            try:
                if not self.api_key:
                    raise AnalysisError(
                        "ไม่พบ GEMINI_API_KEY"
                    )

                result = inspect(
                    Path(task["path"]),
                    self.api_key,
                    task["model"],
                    task["mode"],
                    THUMBS / (
                        f"{task['sha256']}.jpg"
                    ),
                    reference=task["reference_text"],
                )

                if result.get("decision") not in (
                    "PASS",
                    "FAIL",
                    "REVIEW",
                ):
                    raise AnalysisError(
                        "ตัวตรวจต้องส่ง "
                        "PASS / FAIL / REVIEW"
                    )

                with (
                    self.lock,
                    closing(_db()) as conn,
                    conn,
                ):
                    latest = conn.execute("""
                        SELECT
                            COALESCE(
                                p.json_digest,
                                ''
                            )
                        FROM jobs j
                        LEFT JOIN product_refs p
                        ON j.name_key=p.name_key
                        WHERE j.id=?
                    """, (
                        task["id"],
                    )).fetchone()

                    # JSON changed while running:
                    # do not keep the outdated result.

                    if (
                        latest
                        and latest[0]
                        != task["ref_digest"]
                    ):
                        conn.execute("""
                            UPDATE jobs
                            SET
                                status='QUEUED',
                                verdict=NULL,
                                reason=NULL,
                                details=NULL,
                                elapsed=NULL,
                                error=NULL,
                                updated_at=?
                            WHERE id=?
                            AND status='RUNNING'
                        """, (
                            time.time(),
                            task["id"],
                        ))

                    else:
                        conn.execute("""
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
                                "video_type", ""
                            ),
                            result.get(
                                "reason_th", ""
                            ),
                            json.dumps(
                                result,
                                ensure_ascii=False,
                            ),
                            result.get(
                                "duration", 0
                            ),
                            result.get(
                                "elapsed", 0
                            ),
                            time.time(),
                            task["id"],
                        ))

            except QuotaError as exc:
                self._mark_failure(
                    task["id"],
                    "WAITING",
                    str(exc),
                )

                self.pause()

            except Exception as exc:
                self._mark_failure(
                    task["id"],
                    "ERROR",
                    str(exc),
                )

    def _mark_failure(
        self,
        job_id,
        status,
        message,
    ):
        with (
            self.lock,
            closing(_db()) as conn,
            conn,
        ):
            conn.execute("""
                UPDATE jobs
                SET
                    status=?,
                    error=?,
                    updated_at=?
                WHERE id=?
                AND status='RUNNING'
            """, (
                status,
                str(message)[:650],
                time.time(),
                job_id,
            ))

    # ==========================================
    # CLEANUP
    # ==========================================

    def clear_round(self):
        self.pause()

        with (
            self.lock,
            closing(_db()) as conn,
            conn,
        ):
            conn.execute(
                "BEGIN IMMEDIATE"
            )

            running = conn.execute("""
                SELECT COUNT(*)
                FROM jobs
                WHERE status='RUNNING'
            """).fetchone()[0]

            if running:
                raise ValueError(
                    "มีคลิปกำลังตรวจ "
                    "รอให้จบก่อนล้างรอบ"
                )

            conn.execute(
                "DELETE FROM jobs"
            )

            conn.execute(
                "DELETE FROM product_refs"
            )

            for folder in (
                UPLOADS,
                THUMBS,
                EXPORTS,
            ):
                for file in folder.iterdir():
                    if file.is_file():
                        file.unlink(
                            missing_ok=True
                        )

        with self.lock, closing(_db()) as conn:
            conn.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            )

            conn.execute("VACUUM")

    def clear_legacy(self):
        self.pause()

        old_dirs = (
            ROOT / "uploads",
            ROOT / "thumbnails",
            ROOT / "active_only",
        )

        old_files = (
            ROOT / "queue.sqlite3",
            ROOT / "queue.sqlite3-wal",
            ROOT / "queue.sqlite3-shm",
        )

        for path in old_dirs:
            if path.is_dir():
                shutil.rmtree(path)

        for path in old_files:
            path.unlink(
                missing_ok=True
            )
