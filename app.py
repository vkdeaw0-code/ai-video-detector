
"""Streamlit UI for AI Video QC.
Deploy app.py on Streamlit Community Cloud.
"""
from __future__ import annotations

import csv
import hmac
import io
import json
import os
from pathlib import Path
import zipfile

import streamlit as st

from qc_core import ALLOWED_MODELS
from qc_store import (
    MAX_BATCH_BYTES,
    THUMBS,
    JobManager,
)

st.set_page_config(
    page_title="AI Clip QC",
    page_icon="🎬",
    layout="wide",
)


def secret(name: str) -> str:
    try:
        return str(
            st.secrets.get(
                name, ""
            ) or os.getenv(name, "")
        )
    except (
        FileNotFoundError,
        KeyError,
    ):
        return str(
            os.getenv(name, "")
        )


PASSWORD = secret("APP_PASSWORD")
API_KEY = secret("GEMINI_API_KEY")


if not PASSWORD:
    st.error(
        "🔐 ต้องตั้ง APP_PASSWORD ใน "
        "Streamlit App Settings → Secrets "
        "ก่อนเปิดใช้งาน"
    )
    st.stop()


if not st.session_state.get(
    "authorized"
):
    st.title(
        "🔐 AI Clip QC — เข้าสู่ระบบ"
    )

    password = st.text_input(
        "รหัสผ่านของแอป",
        type="password",
    )

    if st.button(
        "เข้าสู่ระบบ",
        type="primary",
    ):
        if hmac.compare_digest(
            password,
            PASSWORD,
        ):
            st.session_state.authorized = True
            st.rerun()
        else:
            st.error(
                "รหัสผ่านไม่ถูกต้อง"
            )

    st.stop()


@st.cache_resource
def manager():
    return JobManager(API_KEY)


svc = manager()
svc.update_key(API_KEY)


st.title(
    "🎬 คัดคลิป AI ตามประเภทสินค้า"
)

st.caption(
    "ตรวจภาพ/สินค้า/มือ/คน/เสียงภาษาไทย "
    "• PASS / FAIL "
    "• คิวตรวจอัตโนมัติ "
    "• Cache"
)


if not API_KEY:
    st.error(
        "กรุณาเพิ่ม GEMINI_API_KEY "
        "ใน Secrets เพื่อเปิดระบบตรวจ"
        "ภาพและเสียงด้วย AI"
    )


with st.sidebar:
    st.subheader(
        "⚙️ ตั้งค่าการตรวจ"
    )

    mode = st.selectbox(
        "ความละเอียด",
        [
            "Balanced",
            "Fast",
            "Deep",
        ],
        index=0,
        help=(
            "Balanced: 2fps, "
            "Fast: 1fps, "
            "Deep: 3fps; "
            "เพิ่มภาพเฉพาะช่วงที่เคลื่อนไหวมาก"
        ),
    )

    model = st.selectbox(
        "โมเดล Gemini",
        sorted(
            ALLOWED_MODELS
        ),
        index=sorted(
            ALLOWED_MODELS
        ).index(
            "gemini-3.5-flash-lite"
        ),
    )

    st.caption(
        "ใช้ Free Tier เฉพาะเมื่อ "
        "API Key/โปรเจกต์มีสิทธิ์"
        "และโควตาคงเหลือ"
    )

    if st.button(
        "⏸️ พักคิวหลังจบคลิปปัจจุบัน",
        use_container_width=True,
    ):
        svc.stop()

    if st.button(
        "▶️ ทำงานต่อ",
        use_container_width=True,
        type="primary",
        disabled=not bool(API_KEY),
    ):
        svc.start()

    disk = svc.disk_usage()

    st.caption(
        "พื้นที่ว่างบนเซิร์ฟเวอร์: "
        f"{disk.free / 1024**3:.2f} GB"
    )

    st.info(
        "Community Cloud อาจพัก/"
        "รีสตาร์ตแอปได้ "
        "ข้อมูลบนดิสก์ไม่ใช่ที่เก็บถาวร"
    )

    if st.button(
        "🧹 ลบไฟล์คลิปเก่าที่จบงานแล้ว (>7 วัน)"
    ):
        st.success(
            "ลบไฟล์ต้นฉบับ "
            f"{svc.cleanup(7)} ไฟล์"
        )


if "batch" not in st.session_state:
    old_batches = svc.batches()

    st.session_state.batch = (
        old_batches[0]["batch"]
        if old_batches
        else svc.new_batch()
    )


if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0


existing = svc.batches()

batch_options = [
    x["batch"] for x in existing
]

left, right = st.columns(
    [3, 1]
)


with left:
    if batch_options:
        chosen = st.selectbox(
            "เลือกรอบการตรวจเดิม",
            [
                st.session_state.batch
            ] + [
                x
                for x in batch_options
                if x != st.session_state.batch
            ],
            key="batch_choice",
        )

        st.session_state.batch = chosen

    st.caption(
        f"Batch ID: {st.session_state.batch}"
    )


with right:
    st.write("")

    if st.button(
        "➕ สร้างรอบใหม่",
        use_container_width=True,
    ):
        st.session_state.batch = (
            svc.new_batch()
        )

        st.session_state.pop(
            "batch_choice",
            None,
        )

        st.session_state.upload_version += 1
        st.rerun()


st.subheader(
    "1. เลือกคลิป (รวมไม่เกิน 1GB ต่อรอบ)"
)

files = st.file_uploader(
    "เลือกไฟล์หลายคลิป",
    type=[
        "mp4",
        "mov",
        "webm",
        "avi",
        "mkv",
    ],
    accept_multiple_files=True,
    key=(
        f"upload_"
        f"{st.session_state.upload_version}"
    ),
)


if files:
    total = sum(
        f.size for f in files
    )

    st.write(
        f"เลือก {len(files)} คลิป "
        f"• {total / 1024**2:.1f} MB "
        "/ 1024 MB"
    )

    if total > MAX_BATCH_BYTES:
        st.error(
            "ขนาดไฟล์รวมที่เลือกเกิน 1GB "
            "กรุณาแบ่งเป็นหลายรอบ"
        )

    if st.button(
        "📥 บันทึกคลิปลงคิว",
        type="primary",
        disabled=total > MAX_BATCH_BYTES,
    ):
        current = svc.jobs(
            st.session_state.batch
        )

        used = sum(
            j["size"] for j in current
        )

        if used + total > MAX_BATCH_BYTES:
            st.error(
                "จำนวนข้อมูลรวมในรอบนี้เกิน 1GB "
                "ให้สร้างรอบใหม่"
            )

        elif (
            svc.disk_usage().free
            < total + 200 * 1024 * 1024
        ):
            st.error(
                "พื้นที่เซิร์ฟเวอร์ไม่เพียงพอ"
                "สำหรับจัดเก็บคลิปทั้งหมด"
            )

        else:
            ok = 0
            errors = []

            for f in files:
                try:
                    ok += int(
                        svc.submit(
                            f,
                            st.session_state.batch,
                            mode,
                            model,
                        )
                    )

                except Exception as exc:
                    errors.append(
                        f"{f.name}: {exc}"
                    )

            if errors:
                st.error(
                    "; ".join(
                        errors[:5]
                    )
                )

            st.success(
                f"เพิ่มคิวแล้ว {ok} คลิป"
            )

            st.session_state.upload_version += 1
            st.rerun()


col_start, col_pause = st.columns(
    [2, 1]
)


with col_start:
    if st.button(
        "▶️ เริ่มตรวจทุกคลิปที่รอคิว",
        type="primary",
        use_container_width=True,
        disabled=not bool(API_KEY),
    ):
        svc.start()

        st.success(
            "เริ่มประมวลผลในคิวแล้ว"
        )


with col_pause:
    if st.button(
        "⏸️ พักหลังจบคลิปนี้",
        use_container_width=True,
    ):
        svc.stop()


st.subheader(
    "2. ผลการตรวจแบบเรียลไทม์"
)


def display_result():
    rows = svc.jobs(
        st.session_state.batch
    )

    n = len(rows)

    passed = sum(
        j["verdict"] == "PASS"
        for j in rows
    )

    failed = sum(
        j["verdict"] == "FAIL"
        for j in rows
    )

    finished = sum(
        j["status"] == "DONE"
        for j in rows
    )

    errors = sum(
        j["status"] in (
            "ERROR",
            "WAITING",
        )
        for j in rows
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    for col, label, val in [
        (c1, "ทั้งหมด", n),
        (c2, "ตรวจแล้ว", finished),
        (c3, "✅ ผ่าน", passed),
        (c4, "❌ ไม่ผ่าน", failed),
        (c5, "⚠️ ตรวจไม่สำเร็จ", errors),
    ]:
        col.metric(
            label,
            val,
        )

    st.progress(
        finished / n if n else 0,
        text=(
            f"ตรวจแล้ว "
            f"{finished}/{n} คลิป"
        ),
    )

    if svc.paused.is_set():
        st.warning(
            "คิวถูกพักไว้: "
            "อาจเกิดจากกดพักหรือ "
            "Gemini API ถึงโควตา"
        )

    if errors:
        if st.button(
            "🔄 ตรวจใหม่เฉพาะคลิป"
            "ที่ผิดพลาด/รอโควตา",
            key="retry_failed",
            disabled=not bool(API_KEY),
        ):
            svc.retry(
                st.session_state.batch
            )

            st.rerun()

    st.caption(
        "สถานะ QUEUED/RUNNING/ERROR/"
        "WAITING คือสถานะงาน "
        "ไม่ใช่ผลคุณภาพ FAIL"
    )

    status_map = {
        "QUEUED": "รอคิว",
        "RUNNING": "กำลังตรวจ",
        "DONE": "ตรวจแล้ว",
        "WAITING": "รอโควตา",
        "ERROR": "ตรวจไม่สำเร็จ",
    }

    type_map = {
        "HUMAN": "คนขายสินค้า",
        "HANDS": "มือโชว์สินค้า",
        "PRODUCT": "สินค้าอย่างเดียว",
        "MIXED": "หลายประเภท",
    }

    choices = st.radio(
        "กรองผลลัพธ์",
        [
            "ทั้งหมด",
            "PASS",
            "FAIL",
            "กำลังตรวจ/ผิดพลาด",
        ],
        horizontal=True,
        key="result_filter",
    )

    filtered = [
        j for j in rows
        if (
            choices == "ทั้งหมด"
            or j["verdict"] == choices
            or (
                choices == "กำลังตรวจ/ผิดพลาด"
                and j["status"] != "DONE"
            )
        )
    ]

    page_size = 15

    total_pages = max(
        1,
        (
            len(filtered)
            + page_size - 1
        ) // page_size,
    )

    page = st.number_input(
        "หน้า",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )

    subset = filtered[
        (page - 1) * page_size:
        page * page_size
    ]

    st.dataframe(
        [
            {
                "ชื่อคลิป": j["filename"],
                "สถานะ": (
                    j["verdict"]
                    or status_map.get(
                        j["status"],
                        j["status"],
                    )
                ),
                "ประเภท": type_map.get(
                    j["category"],
                    "—",
                ),
                "เหตุผล": (
                    j["reason"]
                    or j["error"]
                    or "—"
                ),
                "เวลา (วินาที)": (
                    j["elapsed"] or 0
                ),
                "ใช้ Cache": (
                    "ใช่"
                    if j["cache_hit"]
                    else ""
                ),
            }
            for j in subset
        ],
        use_container_width=True,
        hide_index=True,
    )

    if rows:
        options = {
            f"{j['filename']} ({j['id'][:5]})": j
            for j in filtered
        }

        if options:
            selected = st.selectbox(
                "▶️ เปิดดูคลิปและเหตุผลละเอียด",
                list(options),
                key="video_pick",
            )

            j = options[selected]

            if Path(
                j["path"]
            ).exists():

                st.video(
                    j["path"]
                )

            else:
                st.info(
                    "ไฟล์ต้นฉบับถูกลบหรือ"
                    "เซิร์ฟเวอร์รีสตาร์ตแล้ว "
                    "แต่ผลวิเคราะห์ยังอาจมี"
                    "ในฐานข้อมูล"
                )

            thumb = (
                THUMBS
                / f"{j['sha256']}.jpg"
            )

            if thumb.exists():
                st.image(
                    str(thumb),
                    width=230,
                )

            st.write(
                "**ผล:**",
                (
                    j["verdict"]
                    or status_map.get(
                        j["status"],
                        j["status"],
                    )
                ),
            )

            st.write(
                "**เหตุผล:**",
                (
                    j["reason"]
                    or j["error"]
                    or "รอการตรวจ"
                ),
            )

            if j["details"]:
                with st.expander(
                    "รายการตรวจพบและสรุปเสียง"
                ):
                    details = json.loads(
                        j["details"]
                    )

                    st.write(
                        "สรุปเสียง/คำพูด:",
                        details.get(
                            "speech_summary_th",
                            "",
                        ),
                    )

                    for issue in details.get(
                        "issues", []
                    ):
                        st.write(
                            f"- เวลา "
                            f"~{issue.get('time_sec', '?')}s "
                            f"| {issue.get('category', '')} "
                            f"| {issue.get('severity', '')}: "
                            f"{issue.get('detail_th', '')}"
                        )

    if rows:
        output = io.StringIO()

        writer = csv.writer(
            output
        )

        writer.writerow([
            "filename",
            "status",
            "decision",
            "video_type",
            "reason",
            "duration",
            "elapsed",
            "error",
        ])

        for j in rows:
            writer.writerow([
                j["filename"],
                j["status"],
                j["verdict"],
                j["category"],
                j["reason"],
                j["duration"],
                j["elapsed"],
                j["error"],
            ])

        st.download_button(
            "📄 ดาวน์โหลดรายงาน CSV",
            output.getvalue().encode(
                "utf-8-sig"
            ),
            file_name=(
                f"qc_report_"
                f"{st.session_state.batch}.csv"
            ),
            mime="text/csv",
        )

        st.markdown(
            "**ดาวน์โหลดไฟล์เป็น ZIP "
            "แยกตามผล "
            "(แบ่งชุดละไม่เกิน ~100MB "
            "เพื่อจำกัด RAM)**"
        )

        for verdict in (
            "PASS",
            "FAIL",
        ):
            eligible = [
                j for j in rows
                if (
                    j["verdict"] == verdict
                    and Path(
                        j["path"]
                    ).is_file()
                )
            ]

            pieces = []
            current = []
            current_size = 0

            for j in eligible:
                if (
                    current
                    and current_size
                    + j["size"]
                    > 100 * 1024 * 1024
                ):
                    pieces.append(
                        current
                    )

                    current = []
                    current_size = 0

                current.append(j)
                current_size += j["size"]

            if current:
                pieces.append(
                    current
                )

            for ix, piece in enumerate(
                pieces, 1
            ):
                size = sum(
                    j["size"]
                    for j in piece
                )

                if size > 150 * 1024 * 1024:
                    st.warning(
                        f"{piece[0]['filename']} "
                        "ใหญ่เกินสำหรับ ZIP บนคลาวด์ "
                        "แนะนำดาวน์โหลดไฟล์ต้นฉบับแยก"
                    )
                    continue

                button_key = (
                    f"zip_{verdict}_{ix}_"
                    f"{st.session_state.batch}"
                )

                if st.button(
                    f"📦 เตรียม ZIP {verdict} "
                    f"ส่วน {ix}/{len(pieces)} "
                    f"({size / 1024**2:.0f}MB)",
                    key=button_key,
                ):
                    buffer = io.BytesIO()

                    with zipfile.ZipFile(
                        buffer,
                        "w",
                        zipfile.ZIP_STORED,
                        allowZip64=True,
                    ) as z:
                        for item in piece:
                            z.write(
                                item["path"],
                                arcname=(
                                    f"{item['id'][:6]}_"
                                    f"{Path(item['filename']).name}"
                                ),
                            )

                    st.session_state[
                        "prepared_zip"
                    ] = (
                        verdict,
                        ix,
                        st.session_state.batch,
                        buffer.getvalue(),
                    )

        ready = st.session_state.get(
            "prepared_zip"
        )

        if (
            ready
            and ready[2]
            == st.session_state.batch
        ):
            st.download_button(
                f"⬇️ ดาวน์โหลด ZIP "
                f"{ready[0]} "
                f"ส่วน {ready[1]}",
                ready[3],
                file_name=(
                    f"{ready[0]}_"
                    f"{ready[2]}_"
                    f"part{ready[1]}.zip"
                ),
                mime="application/zip",
            )


@st.fragment(
    run_every="3s"
)
def live_results():
    display_result()


live_results()

st.caption(
    "ผล AI เป็นการคัดกรองเบื้องต้น "
    "ควรสุ่มตรวจด้วยมนุษย์"
    "ก่อนนำไปใช้เชิงพาณิชย์"
)
