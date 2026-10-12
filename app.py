
"""AI Video QC V3.9 Hybrid Streamlit App.

Upload MP4 only or MP4 + optional JSON.
Includes PASS, FAIL, REVIEW, CSV and ZIP exports.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import uuid
import zipfile

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from bulk_upload import ChunkReceiver
from qc_core import ALLOWED_MODELS, ENGINE_VERSION
from qc_store import (
    EXPORTS,
    MAX_ACTIVE_BYTES,
    MAX_VIDEOS,
    ActiveQueue,
)


st.set_page_config(
    page_title="AI Video QC V3.9",
    page_icon="🎬",
    layout="wide",
)

uploader = components.declare_component(
    "bulk_json_mp4_one_drop",
    path=str(
        Path(__file__).parent / "bulk_uploader"
    ),
)


# ==========================================
# SECRETS / LOGIN
# ==========================================

def secret(name):
    try:
        return str(
            st.secrets.get(name, "")
            or os.getenv(name, "")
        )

    except (
        FileNotFoundError,
        KeyError,
    ):
        return str(
            os.getenv(name, "")
        )


password = secret("APP_PASSWORD")
api_key = secret("GEMINI_API_KEY")

if not password:
    st.error(
        "ตั้ง APP_PASSWORD "
        "ใน Streamlit Cloud Secrets ก่อน"
    )

    st.stop()

if not st.session_state.get("authorized"):
    st.title("🔐 AI Video QC")

    entry = st.text_input(
        "รหัสผ่าน",
        type="password",
    )

    if st.button(
        "เข้าสู่ระบบ",
        type="primary",
    ):
        if hmac.compare_digest(
            entry,
            password,
        ):
            st.session_state.authorized = True
            st.rerun()

        else:
            st.error(
                "รหัสผ่านไม่ถูกต้อง"
            )

    st.stop()


# ==========================================
# QUEUE SERVICE
# ==========================================

@st.cache_resource
def get_service(version):
    return ActiveQueue(api_key)


service = get_service("hybrid-v39")

service.update_key(api_key)


@st.cache_resource
def get_receiver(version):
    return ChunkReceiver(service)


receiver = get_receiver("chunk-v39")


# ==========================================
# CSV
# ==========================================

def make_csv(rows):
    buffer = io.StringIO()

    writer = csv.writer(buffer)

    writer.writerow([
        "video_filename",
        "json_filename",
        "product_title",
        "json_used",
        "status",
        "decision",
        "product_type",
        "reason",
        "speech_summary",
        "seconds",
        "recheck",
        "review_required",
        "error",
    ])

    for row in rows:
        try:
            details = json.loads(
                row.get("details")
                or "{}"
            )

        except (
            ValueError,
            TypeError,
        ):
            details = {}

        writer.writerow([
            row["filename"],
            row.get("matched_json") or "",
            row.get("product_title") or "",
            details.get(
                "product_reference_used", ""
            ),
            row.get("status"),
            row.get("verdict") or "",
            details.get(
                "product_type_th"
            ) or row.get("category") or "",
            row.get("reason") or "",
            details.get(
                "speech_summary_th"
            ) or "",
            row.get("elapsed") or "",
            details.get(
                "audit_done", ""
            ),
            details.get(
                "review_required", ""
            ),
            row.get("error") or "",
        ])

    return buffer.getvalue().encode(
        "utf-8-sig"
    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("⚙️ การตรวจ")

    mode = st.selectbox(
        "คุณภาพการดึงภาพ",
        ["Balanced", "Fast", "Deep"],
    )

    models = sorted(ALLOWED_MODELS)

    preferred = "gemini-2.5-flash-lite"

    model = st.selectbox(
        "Gemini API model",
        models,
        index=(
            models.index(preferred)
            if preferred in models
            else 0
        ),
    )

    auto_start = st.checkbox(
        "เริ่มตรวจเมื่อ MP4 อัปโหลดเสร็จ",
        value=True,
    )

    st.caption(
        "JSON ไม่บังคับ "
        "หากมีจะใช้เพิ่มข้อมูลตรวจสอบ"
    )

    if st.button(
        "▶️ เริ่ม / ทำงานต่อ",
        disabled=not api_key,
        use_container_width=True,
    ):
        service.start()

    if st.button(
        "⏸️ พักคิว",
        use_container_width=True,
    ):
        service.pause()

    st.caption(
        f"ดิสก์ว่าง "
        f"{service.free_disk() / 1024**3:.2f} GB"
    )


# ==========================================
# MAIN DASHBOARD
# ==========================================

st.title(
    "🎬 AI Video QC V3.9 — Hybrid Quality"
)

st.caption(
    "MP4 อย่างเดียว หรือ MP4 + JSON • "
    "ตรวจสินค้า ภาพ เสียง และความสอดคล้อง • "
    "PASS / FAIL / REVIEW"
)

if not api_key:
    st.warning(
        "ไม่พบ GEMINI_API_KEY: "
        "อัปโหลดได้ แต่ยังตรวจไม่ได้"
    )

count, used, refs = service.stats()
rows = service.jobs()

matched = sum(
    bool(row.get("matched_json"))
    for row in rows
)

columns = st.columns(4)

metrics = [
    ("MP4", count),
    ("JSON", refs),
    ("MP4 มี JSON", matched),
    (
        "พื้นที่ MB",
        f"{used / 1024**2:.0f}/"
        f"{MAX_ACTIVE_BYTES / 1024**2:.0f}",
    ),
]

for col, (heading, value) in zip(
    columns,
    metrics,
):
    col.metric(heading, value)

st.progress(
    min(
        1.0,
        used / MAX_ACTIVE_BYTES,
    ),
    text=(
        f"สูงสุด {MAX_VIDEOS} คลิปต่อรอบ "
        "(ขึ้นกับดิสก์จริง)"
    ),
)


# ==========================================
# OLD ENGINE WARNING
# ==========================================

old = []

for row in rows:
    if (
        row["status"] == "DONE"
        and row.get("details")
    ):
        try:
            report = json.loads(
                row["details"]
            )

            if (
                report.get("engine")
                != ENGINE_VERSION
            ):
                old.append(
                    row["filename"]
                )

        except (
            ValueError,
            TypeError,
        ):
            old.append(
                row["filename"]
            )

if old:
    st.warning(
        f"มีผลตรวจจากเวอร์ชันเก่า "
        f"{len(old)} คลิป "
        "กรุณาดาวน์โหลดแล้วล้างรอบ "
        "ก่อนทดสอบ V3.9"
    )


# ==========================================
# BULK UPLOAD
# ==========================================

st.subheader(
    "① อัปโหลด MP4 (JSON เป็นทางเลือก)"
)

st.info(
    "ลาก MP4 1–150 ไฟล์ได้โดยไม่มี JSON "
    "หรือเลือก MP4 + JSON พร้อมกันก็ได้ "
    "ระบบคัด JSON ก่อน MP4 ให้อัตโนมัติ "
    "หากเพิ่ม JSON ทีหลังจะตรวจคลิปนั้นใหม่"
)

st.caption(
    "ชื่อวิดีโอต้องตรงกับ video_filename "
    "ของ JSON เมื่อมี JSON • "
    "เปิดหน้าเว็บไว้จนส่งไฟล์เสร็จ • "
    "MP4 ไม่เกิน 50 MB ต่อไฟล์"
)

if "bulk_nonce" not in st.session_state:
    st.session_state.bulk_nonce = (
        uuid.uuid4().hex
    )

if "bulk_ack" not in st.session_state:
    st.session_state.bulk_ack = None

if "bulk_last_id" not in st.session_state:
    st.session_state.bulk_last_id = ""


event = uploader(
    nonce=st.session_state.bulk_nonce,
    ack=st.session_state.bulk_ack,
    key="bulk_json_mp4_input",
    default=None,
)

if isinstance(event, dict):
    event_id = str(
        event.get("id", "")
    )

    if (
        event_id
        and event_id
        != st.session_state.bulk_last_id
    ):
        try:
            response = receiver.handle(
                event,
                session_id=st.session_state.bulk_nonce,
                mode=mode,
                model=model,
                auto_start=bool(
                    api_key and auto_start
                ),
            )

            acknowledgement = {
                "id": event_id,
                **response,
            }

        except Exception as exc:
            acknowledgement = {
                "id": event_id,
                "ok": False,
                "message": str(exc)[:450],
            }

        st.session_state.bulk_last_id = event_id

        st.session_state.bulk_ack = (
            acknowledgement
        )

        st.rerun()


unmatched = service.unmatched_references()

if unmatched:
    with st.expander(
        f"มี JSON {len(unmatched)} ไฟล์ "
        "ที่ยังไม่มี MP4 ตรงกัน"
    ):
        st.dataframe(
            unmatched,
            hide_index=True,
            use_container_width=True,
        )


# ==========================================
# RESULTS
# ==========================================

@st.fragment(run_every="4s")
def results_ui():
    current = service.jobs()

    counts = {
        tag: sum(
            row.get("verdict") == tag
            for row in current
        )
        for tag in (
            "PASS",
            "FAIL",
            "REVIEW",
        )
    }

    done = sum(
        row["status"] == "DONE"
        for row in current
    )

    blocked = sum(
        row["status"] in (
            "ERROR",
            "WAITING",
        )
        for row in current
    )

    st.divider()

    st.subheader("② ผลตรวจ")

    cols = st.columns(5)

    titles = [
        "คลิปทั้งหมด",
        "ตรวจแล้ว",
        "✅ PASS",
        "❌ FAIL",
        "👁️ REVIEW",
    ]

    values = [
        len(current),
        done,
        counts["PASS"],
        counts["FAIL"],
        counts["REVIEW"],
    ]

    for col, title, value in zip(
        cols,
        titles,
        values,
    ):
        col.metric(title, value)

    st.progress(
        (
            done / len(current)
            if current else 0
        ),
        text=(
            f"ตรวจแล้ว {done}/{len(current)} "
            f"• ติดขัด {blocked}"
        ),
    )

    if counts["REVIEW"]:
        st.warning(
            f"มี {counts['REVIEW']} คลิป "
            "รอตรวจด้วยคน "
            "อย่าเผยแพร่ก่อนตรวจยืนยัน"
        )

    if blocked:
        st.error(
            f"มี {blocked} คลิป "
            "ERROR/รอโควตา "
            "ไม่ใช่คุณภาพ FAIL"
        )

        if st.button(
            "🔄 ลองตรวจคลิปติดขัดใหม่",
            disabled=not api_key,
        ):
            service.retry()
            st.rerun()

    if not current:
        return

    # ======================================
    # FILTER TABLE
    # ======================================

    query = st.text_input(
        "ค้นหาคลิป",
        placeholder="ชื่อ MP4 หรือชื่อสินค้า",
    )

    status_filter = st.radio(
        "ตัวกรอง",
        [
            "ทั้งหมด",
            "PASS",
            "FAIL",
            "REVIEW",
            "กำลังตรวจ",
            "ERROR",
        ],
        horizontal=True,
    )

    selected = []

    for row in current:
        searchable = (
            row["filename"]
            + " "
            + (
                row.get("product_title")
                or ""
            )
        ).casefold()

        if query.casefold() not in searchable:
            continue

        if (
            status_filter in (
                "PASS",
                "FAIL",
                "REVIEW",
            )
            and row["verdict"] != status_filter
        ):
            continue

        if (
            status_filter == "กำลังตรวจ"
            and row["status"]
            not in ("QUEUED", "RUNNING")
        ):
            continue

        if (
            status_filter == "ERROR"
            and row["status"]
            not in ("ERROR", "WAITING")
        ):
            continue

        selected.append(row)

    st.dataframe(
        [
            {
                "วิดีโอ": row["filename"],
                "JSON": (
                    row.get("matched_json")
                    or "ไม่ใช้"
                ),
                "สินค้า (JSON)": (
                    row.get("product_title")
                    or "—"
                ),
                "ผล": (
                    row.get("verdict")
                    or row["status"]
                ),
                "เหตุผล": (
                    row.get("reason")
                    or row.get("error")
                    or "—"
                ),
                "วินาที": (
                    row.get("elapsed")
                    or 0
                ),
            }
            for row in selected
        ],
        hide_index=True,
        use_container_width=True,
    )

    # ======================================
    # VIDEO DETAIL
    # ======================================

    if selected:
        choices = {
            (
                f"{row['filename']} "
                f"[{row['id'][:6]}]"
            ): row
            for row in selected
        }

        chosen_label = st.selectbox(
            "ดูรายละเอียดคลิป",
            list(choices),
        )

        row = choices[chosen_label]

        with st.expander(
            "▶️ ภาพ / เสียง / "
            "เหตุผล / หลักฐานละเอียด"
        ):
            if Path(
                row["path"]
            ).is_file():
                st.video(
                    row["path"]
                )

            st.write(
                "**JSON:**",
                row.get("matched_json")
                or "ไม่มี JSON",
            )

            st.write(
                "**เหตุผล:**",
                row.get("reason")
                or row.get("error")
                or "รอประมวลผล",
            )

            if row.get(
                "product_reference"
            ):
                st.text(
                    row["product_reference"]
                )

            if row.get("details"):
                try:
                    details = json.loads(
                        row["details"]
                    )

                    st.write(
                        "**ประเภทสินค้าจากภาพ:**",
                        details.get(
                            "product_type_th"
                        ),
                    )

                    st.write(
                        "**สรุปคำพูด:**",
                        details.get(
                            "speech_summary_th"
                        ),
                    )

                    fields = (
                        "scale_review",
                        "audio_review",
                        "match_review",
                        "visual_review",
                        "mirror_review",
                        "scale_audit",
                        "speech_audit",
                        "match_audit",
                        "mirror_audit",
                        "review_required",
                        "timings",
                    )

                    st.json({
                        name: details.get(name)
                        for name in fields
                    })

                except (
                    ValueError,
                    TypeError,
                ):
                    st.warning(
                        "อ่านรายละเอียดไม่ได้"
                    )

    # ======================================
    # CSV EXPORT
    # ======================================

    st.download_button(
        "📄 ดาวน์โหลดรายงาน CSV",
        make_csv(current),
        file_name="qc_v39_current.csv",
        mime="text/csv",
    )

    # ======================================
    # ZIP EXPORT
    # ======================================

    st.markdown(
        "**📦 ดาวน์โหลด MP4 แยก "
        "PASS / FAIL / REVIEW**"
    )

    verdict = st.selectbox(
        "กลุ่ม ZIP",
        ["PASS", "FAIL", "REVIEW"],
    )

    eligible = [
        row
        for row in current
        if (
            row.get("verdict") == verdict
            and Path(
                row["path"]
            ).is_file()
        )
    ]

    groups = []
    group = []
    size = 0

    for row in eligible:
        if (
            group
            and size + row["size"]
            > 30 * 1024**2
        ):
            groups.append(group)
            group = []
            size = 0

        group.append(row)
        size += row["size"]

    if group:
        groups.append(group)

    if groups:
        index = st.selectbox(
            "ชุดไฟล์ ZIP",
            range(len(groups)),
            format_func=lambda n: (
                f"ชุด {n + 1}/{len(groups)} "
                f"({len(groups[n])} คลิป)"
            ),
        )

        fingerprint = hashlib.sha256(
            "|".join(
                row["id"]
                for row in groups[index]
            ).encode()
        ).hexdigest()[:12]

        if st.button(
            "เตรียม ZIP ชุดที่เลือก"
        ):
            destination = EXPORTS / (
                f"v39_{verdict}_{fingerprint}.zip"
            )

            with zipfile.ZipFile(
                destination,
                "w",
                compression=zipfile.ZIP_STORED,
            ) as archive:
                for row in groups[index]:
                    archive.write(
                        row["path"],
                        arcname=(
                            f"{row['id'][:6]}_"
                            f"{row['filename']}"
                        ),
                    )

            st.session_state.export_ready = (
                verdict,
                fingerprint,
                str(destination),
            )

        ready = st.session_state.get(
            "export_ready"
        )

        if (
            ready
            and ready[:2] == (
                verdict,
                fingerprint,
            )
            and Path(
                ready[2]
            ).is_file()
        ):
            with open(
                ready[2], "rb"
            ) as file:
                st.download_button(
                    "⬇️ ดาวน์โหลด ZIP",
                    file,
                    file_name=(
                        f"v39_{verdict}_"
                        f"{index + 1}.zip"
                    ),
                    mime="application/zip",
                )

    else:
        st.caption(
            "ยังไม่มีไฟล์กลุ่มนี้"
        )

    # ======================================
    # CLEAR ROUND
    # ======================================

    st.divider()

    st.subheader(
        "③ ล้างรอบ / เริ่มใหม่"
    )

    st.warning(
        "ดาวน์โหลด CSV และ ZIP ก่อนล้างรอบ "
        "ข้อมูลจะถูกลบ"
    )

    confirm = st.checkbox(
        "ยืนยันว่าดาวน์โหลดผลที่ต้องการแล้ว"
    )

    if st.button(
        "🗑️ ล้างข้อมูลรอบปัจจุบัน",
        disabled=not confirm,
    ):
        try:
            service.clear_round()
            receiver.clear_staging()

            st.session_state.bulk_nonce = (
                uuid.uuid4().hex
            )

            st.session_state.bulk_ack = None
            st.session_state.bulk_last_id = ""

            st.session_state.pop(
                "export_ready",
                None,
            )

            st.rerun()

        except ValueError as exc:
            st.error(str(exc))


results_ui()

st.caption(
    "จำนวน FAIL ไม่ได้ถูกบังคับ • "
    "การวิเคราะห์ AI ยังผิดพลาดได้ • "
    "การจัดเก็บขึ้นอยู่กับดิสก์ของ Streamlit Cloud"
)
