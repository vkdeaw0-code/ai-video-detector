
"""Chunked uploads from browser to Streamlit disk."""
from __future__ import annotations

import base64
import binascii
import re
import shutil
import threading
import time
from pathlib import Path

from qc_store import (
    ACTIVE,
    MAX_JSON,
    MAX_VIDEO,
    MIN_DISK_FREE,
    VIDEO_EXTS,
    _name,
)

CHUNK_BYTES = 1572864
STAGING = ACTIVE / "bulk_incoming"

IDENTIFIER = re.compile(
    r"^[a-zA-Z0-9_-]{8,64}$"
)


class ChunkReceiver:

    def __init__(self, service):
        self.service = service
        self.lock = threading.RLock()
        self.inflight = {}

        STAGING.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.cleanup_orphans()

    def cleanup_orphans(self):
        with self.lock:
            active_paths = {
                record["path"]
                for record in self.inflight.values()
            }

            now = time.time()

            for item in STAGING.glob("*.partial"):
                try:
                    if (
                        item not in active_paths
                        and now - item.stat().st_mtime > 3600
                    ):
                        item.unlink(missing_ok=True)
                except OSError:
                    pass

    def clear_staging(self):
        with self.lock:
            for record in self.inflight.values():
                record["path"].unlink(
                    missing_ok=True
                )

            self.inflight.clear()

            for item in STAGING.glob("*.partial"):
                item.unlink(missing_ok=True)

    def handle(
        self,
        payload,
        session_id,
        mode,
        model,
        auto_start,
    ):
        if not isinstance(payload, dict):
            raise ValueError(
                "รูปแบบข้อมูลอัปโหลดไม่ถูกต้อง"
            )

        event_type = str(
            payload.get("type", "")
        )

        token = str(
            payload.get("file_id", "")
        )

        if (
            not IDENTIFIER.fullmatch(token)
            or not IDENTIFIER.fullmatch(session_id)
        ):
            raise ValueError(
                "รหัสการอัปโหลดไม่ถูกต้อง"
            )

        key = (session_id, token)

        with self.lock:
            try:
                if event_type == "start":
                    return self._start(payload, key)

                if event_type == "chunk":
                    return self._chunk(payload, key)

                if event_type == "end":
                    return self._end(
                        key,
                        mode,
                        model,
                        auto_start,
                    )

                if event_type == "abort":
                    self._discard(key)
                    return {
                        "ok": True,
                        "message": "ยกเลิกไฟล์แล้ว",
                    }

                raise ValueError(
                    "ไม่รู้จักคำสั่งอัปโหลด"
                )

            except Exception:
                if event_type != "start":
                    self._discard(key)
                raise

    def _start(self, payload, key):
        if key in self.inflight:
            self._discard(key)

        name = _name(
            payload.get("filename", "")
        )

        suffix = Path(name).suffix.lower()

        try:
            size = int(
                payload.get("size", -1)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ขนาดไฟล์ไม่ถูกต้อง"
            ) from exc

        if suffix == ".json":
            limit = MAX_JSON
        elif suffix in VIDEO_EXTS:
            limit = MAX_VIDEO
        else:
            raise ValueError(
                f"ไม่รองรับไฟล์ {name}"
            )

        if not 0 < size <= limit:
            raise ValueError(
                f"ไฟล์ {name} ใหญ่เกินกำหนด"
            )

        if (
            shutil.disk_usage(STAGING).free
            < MIN_DISK_FREE + min(
                size, 8 * 1024**2
            )
        ):
            raise ValueError(
                "พื้นที่ดิสก์เซิร์ฟเวอร์ไม่พอ"
            )

        path = STAGING / (
            f"{key[0]}_{key[1]}.partial"
        )

        with open(path, "wb"):
            pass

        self.inflight[key] = {
            "filename": name,
            "path": path,
            "size": size,
            "received": 0,
        }

        return {
            "ok": True,
            "received": 0,
        }

    def _chunk(self, payload, key):
        current = self.inflight.get(key)

        if current is None:
            raise ValueError(
                "ไม่พบไฟล์ที่เริ่มอัปโหลด"
            )

        try:
            offset = int(
                payload.get("offset", -1)
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "ตำแหน่งข้อมูลไม่ถูกต้อง"
            ) from exc

        if offset != current["received"]:
            raise ValueError(
                "ลำดับข้อมูลไม่ตรง"
            )

        encoded = payload.get("data")

        if (
            not isinstance(encoded, str)
            or len(encoded)
            > CHUNK_BYTES * 4 // 3 + 16
        ):
            raise ValueError(
                "ชิ้นข้อมูลใหญ่เกินไป"
            )

        try:
            chunk = base64.b64decode(
                encoded,
                validate=True,
            )
        except (
            ValueError,
            binascii.Error,
        ) as exc:
            raise ValueError(
                "ข้อมูลอัปโหลดเสียหาย"
            ) from exc

        if not chunk or len(chunk) > CHUNK_BYTES:
            raise ValueError(
                "ขนาดชิ้นข้อมูลไม่ถูกต้อง"
            )

        if (
            current["received"] + len(chunk)
            > current["size"]
        ):
            raise ValueError(
                "ข้อมูลเกินขนาดไฟล์จริง"
            )

        if (
            shutil.disk_usage(STAGING).free
            < MIN_DISK_FREE + len(chunk)
        ):
            raise ValueError(
                "พื้นที่ดิสก์ไม่พอ"
            )

        with open(
            current["path"], "ab"
        ) as output:
            output.write(chunk)

        current["received"] += len(chunk)

        return {
            "ok": True,
            "received": current["received"],
        }

    def _end(
        self,
        key,
        mode,
        model,
        auto_start,
    ):
        current = self.inflight.get(key)

        if current is None:
            raise ValueError(
                "ไม่พบไฟล์ที่อัปโหลด"
            )

        if current["received"] != current["size"]:
            raise ValueError(
                "ไฟล์อัปโหลดไม่ครบ"
            )

        try:
            suffix = Path(
                current["filename"]
            ).suffix.lower()

            with open(
                current["path"], "rb"
            ) as stream:

                if suffix == ".json":
                    data = stream.read(
                        MAX_JSON + 1
                    )

                    saved = self.service.add_json(
                        data,
                        current["filename"],
                    )

                else:
                    saved = self.service.add_video(
                        stream,
                        current["filename"],
                        current["size"],
                        mode,
                        model,
                    )

            if (
                auto_start
                and self.service.api_key
            ):
                self.service.start()

            return {
                "ok": True,
                "saved": bool(saved),
                "filename": current["filename"],
            }

        finally:
            self._discard(key)

    def _discard(self, key):
        current = self.inflight.pop(
            key, None
        )

        if current:
            current["path"].unlink(
                missing_ok=True
            )
