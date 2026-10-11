
"""AI Video QC — Batch Manager V3.
Compatible with qc_core.py Scale Guard V2.
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
from qc_store import MAX_BATCH_BYTES, THUMBS, JobManager

st.set_page_config(
    page_title="AI Video QC | รอบงาน",
    page_icon="🎬",
    layout="wide",
)


def get_secret(name):
    try:
        return str(
            st.secrets.get(name, "")
            or os.getenv(name, "")
        )
    except (FileNotFoundError, KeyError):
        return str(os.getenv(name, ""))


PASSWORD = get_secret("APP_PASSWORD")
API_KEY = get_secret("GEMINI_API_KEY")

if not PASSWORD:
    st.error(
        "ตั้ง APP_PASSWORD ใน Streamlit Cloud "
        "→ App settings → Secrets ก่อน"
    )
    st.stop()

if not st.session_state.get("authorized"):
    st.title("🔐 AI Video QC")
    password = st.text_input(
        "รหัสผ่านเข้าแอป",
        type="password",
    )
    if st.button("เข้าสู่ระบบ", type="primary"):
        if hmac.compare_digest(password, PASSWORD):
            st.session_state.authorized = True
            st.rerun()
        else:
            st.error("รหัสผ่านไม่ถูกต้อง")
    st.stop()


@st.cache_resource
def get_manager():
    return JobManager(API_KEY)


svc = get_manager()
svc.update_key(API_KEY)


def report_csv(rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "filename",
        "status",
        "decision",
        "type",
        "reason",
        "duration",
        "seconds",
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
    return output.getvalue().encode("utf-8-sig")


def change_to_batch(batch_id):
    st.session_state.batch_navigation_target = batch_id
    st.session_state.upload_version = (
        st.session_state.get("upload_version", 0) + 1
    )
    st.session_state.pop("last_upload_attempt", None)
    st.session_state.pop("upload_errors", None)
    st.session_state.pop("zip_ready", None)
    st.session_state.pop("upload_summary", None)
    st.session_state.pop("upload_summary_batch", None)


with st.sidebar:
    st.subheader("⚙️ ตัวเลือกการตรวจ")

    mode = st.selectbox(
        "ความละเอียด",
        ["Balanced", "Fast", "Deep"],
        index=0,
    )

    models = sorted(ALLOWED_MODELS)
    default_model = (
        models.index("gemini-3.5-flash-lite")
        if "gemini-3.5-flash-lite" in models
        else 0
    )

    model = st.selectbox(
        "โมเดล Gemini",
        models,
        index=default_model,
    )

    st.caption(
        "ใช้ตามสิทธิ์และโควตา Free Tier "
        "จริงของบัญชี"
    )

    auto_start = st.checkbox(
        "เริ่มตรวจอัตโนมัติเมื่อบันทึกคลิป",
        value=False,
    )

    if st.button(
        "⏸️ พักคิวหลังจบคลิปปัจจุบัน",
        use_container_width=True,
    ):
        svc.stop()

    if st.button(
        "▶️ ทำงานต่อ",
        use_container_width=True,
        disabled=not bool(API_KEY),
    ):
        svc.start()

    st.caption(
        "พื้นที่ว่างเซิร์ฟเวอร์: "
        f"{svc.disk_usage().free / 1024**3:.1f} GB"
    )

    if st.button(
        "🧹 ลบไฟล์ต้นฉบับเก่าที่ตรวจจบแล้ว (>7 วัน)",
        use_container_width=True,
    ):
        st.success(
            "ลบไฟล์ต้นฉบับเก่าแล้ว "
            f"{svc.cleanup(7)} ไฟล์ "
            "(ประวัติผลตรวจยังอยู่)"
        )

    st.info(
        "Streamlit Community Cloud ไม่รับประกัน"
        "การเก็บไฟล์/SQLite ถาวรเมื่อเซิร์ฟเวอร์"
        "ถูกเปลี่ยนหรือรีสตาร์ต"
    )


st.title("🎬 AI Video QC — จัดการรอบตรวจคลิป")

st.caption(
    "อัปโหลดอัตโนมัติ → เห็นไฟล์ที่บันทึกแล้ว "
    "→ ตรวจ PASS/FAIL → ปิดรอบ → เริ่มรอบใหม่"
)

if not API_KEY:
    st.warning(
        "ยังไม่พบ GEMINI_API_KEY ใน Secrets "
        "— บันทึกคลิปได้แต่เริ่มตรวจ AI ไม่ได้"
    )


all_batches = svc.batches()

if not all_batches:
    svc.create_batch()
    all_batches = svc.batches()

ids = [b["batch"] for b in all_batches]

pending_navigation = st.session_state.pop(
    "batch_navigation_target",
    None,
)

if pending_navigation in ids:
    st.session_state.batch_selector = pending_navigation

if st.session_state.get("batch_selector") not in ids:
    open_batches = [
        b for b in all_batches
        if b["state"] == "OPEN"
    ]
    st.session_state.batch_selector = (
        open_batches or all_batches
    )[0]["batch"]

if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0

map_batches = {
    b["batch"]: b for b in all_batches
}


def describe_batch(batch_id):
    b = map_batches[batch_id]

    flag = (
        "🔒 ปิดแล้ว"
        if b["state"] == "CLOSED"
        else "📂 เปิดอยู่"
    )

    return (
        f"{b['label']} • "
        f"{b['done']}/{b['n']} คลิป • {flag}"
    )


st.subheader("① เลือกรอบงาน")

if st.session_state.get("upload_errors"):
    st.error(
        "ไฟล์บางส่วนยังบันทึกไม่สำเร็จ: "
        "แก้ไขหรือล้างรายการที่เลือกก่อนเปลี่ยนรอบ"
    )

selector_disabled = bool(
    st.session_state.get("upload_errors")
)

selected = st.selectbox(
    "ประวัติรอบทั้งหมด (เลือกกลับไปดูย้อนหลังได้)",
    options=ids,
    key="batch_selector",
    format_func=describe_batch,
    disabled=selector_disabled,
)

current = map_batches[selected]

left, center, right = st.columns([2, 2, 2])

with left:
    st.metric(
        "ไฟล์ในรอบนี้",
        f"{current['n']} คลิป",
    )

with center:
    st.metric(
        "ตรวจเสร็จ",
        f"{current['done']}/{current['n']}",
    )

with right:
    st.metric(
        "ใช้พื้นที่รอบนี้",
        f"{current['bytes'] / 1024**2:.1f} MB / 1024 MB",
    )


with st.expander(
    "✏️ ตั้งชื่อรอบ เพื่อไม่ให้สับสน "
    "(เช่น สินค้าครัว รอบเช้า)"
):
    with st.form("rename_batch_form"):
        new_name = st.text_input(
            "ชื่อรอบ",
            value=current["label"],
            max_chars=70,
        )

        if st.form_submit_button("บันทึกชื่อรอบ"):
            try:
                svc.rename_batch(selected, new_name)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


if st.button(
    "➕ สร้างรอบใหม่ โดยเก็บรอบนี้ไว้",
    disabled=selector_disabled,
    help=(
        "งานที่ยังไม่เสร็จในรอบเดิมยังอยู่ในคิว "
        "ระบบไม่ได้ลบหรือย้ายคลิป"
    ),
):
    new_id = svc.create_batch()
    change_to_batch(new_id)
    st.rerun()


st.divider()

st.subheader("② เพิ่มคลิปเข้ารอบ")

st.caption(
    "เมื่อเลือกไฟล์ ระบบจะบันทึกเข้ารอบให้อัตโนมัติ "
    "แล้วแสดงรายชื่อในตารางด้านล่าง"
)

if current["state"] == "CLOSED":
    st.info(
        "รอบนี้ปิดแล้ว เปิดดูผลย้อนหลังหรือสร้างรอบใหม่ได้ "
        "แต่เพิ่มคลิปเข้ารอบเดิมไม่ได้"
    )

else:
    uploaded = st.file_uploader(
        "เลือกหลายคลิปพร้อมกัน • สูงสุดรวม 1 GB ต่อรอบ",
        type=[
            "mp4",
            "mov",
            "webm",
            "avi",
            "mkv",
        ],
        accept_multiple_files=True,
        key=(
            f"upload_{selected}_"
            f"{st.session_state.upload_version}"
        ),
    )

    if uploaded:
        total = sum(f.size for f in uploaded)

        st.caption(
            f"เลือกมา {len(uploaded)} ไฟล์ "
            f"({total / 1024**2:.1f} MB) "
            "— กำลังตรวจว่าบันทึกเข้ารอบได้ครบหรือไม่"
        )

        signature = (
            selected,
            st.session_state.upload_version,
            tuple((f.name, f.size) for f in uploaded),
        )

        if total > MAX_BATCH_BYTES:
            st.error(
                "ไฟล์ที่เลือกครั้งนี้รวมกันเกิน 1 GB "
                "กรุณาเลือกให้น้อยลง"
            )

        elif signature != st.session_state.get(
            "last_upload_attempt"
        ):
            st.session_state.last_upload_attempt = signature

            inserted = 0
            duplicate = 0
            errors = []

            progress = st.progress(
                0,
                text="กำลังบันทึกคลิปลงรอบ...",
            )

            for index, file in enumerate(uploaded):
                try:
                    was_added = svc.submit(
                        file,
                        selected,
                        mode,
                        model,
                    )

                    inserted += int(was_added)
                    duplicate += int(not was_added)

                except Exception as exc:
                    errors.append(
                        f"{file.name}: {exc}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded),
                    text=(
                        f"บันทึก {index + 1}/"
                        f"{len(uploaded)}"
                    ),
                )

            progress.empty()

            if errors:
                st.session_state.upload_errors = errors
                st.session_state.upload_summary = (
                    f"บันทึกได้ {inserted}, "
                    f"ซ้ำ {duplicate}, "
                    f"ยังไม่สำเร็จ {len(errors)}"
                )
                st.session_state.upload_summary_batch = selected

            else:
                st.session_state.pop(
                    "upload_errors",
                    None,
                )

                st.session_state.upload_summary = (
                    f"✅ บันทึกเข้ารอบแล้ว {inserted} คลิป "
                    f"• ข้ามไฟล์ซ้ำ {duplicate} คลิป"
                )

                st.session_state.upload_summary_batch = selected

                # ล้างช่อง Upload หลังบันทึกสำเร็จเท่านั้น
                st.session_state.upload_version += 1

                if auto_start and API_KEY and inserted:
                    svc.start()

                st.rerun()

        if st.session_state.get("upload_errors"):
            st.warning(
                st.session_state.get(
                    "upload_summary",
                    "บันทึกไม่ครบ",
                )
            )

            for error in st.session_state.upload_errors[:10]:
                st.error(error)

            a, b = st.columns(2)

            with a:
                if st.button(
                    "🔄 ลองบันทึกส่วนที่เหลืออีกครั้ง"
                ):
                    st.session_state.pop(
                        "last_upload_attempt",
                        None,
                    )
                    st.rerun()

            with b:
                if st.button(
                    "🗑️ ยกเลิกไฟล์ที่ยังไม่บันทึก"
                ):
                    st.session_state.pop(
                        "upload_errors",
                        None,
                    )
                    st.session_state.upload_version += 1
                    st.rerun()


if (
    st.session_state.get("upload_summary")
    and st.session_state.get("upload_summary_batch")
    == selected
):
    st.success(st.session_state.upload_summary)

    st.caption(
        "ไฟล์ที่เลือกจะถูกล้างจากช่อง Upload "
        "หลังบันทึกสำเร็จ แต่ไฟล์ยังอยู่ในรอบ "
        "และแสดงในตารางด้านล่าง"
    )


col1, col2 = st.columns([2, 1])

with col1:
    if st.button(
        "▶️ เริ่มตรวจคลิปที่รอคิว",
        type="primary",
        use_container_width=True,
        disabled=not bool(API_KEY),
    ):
        svc.start()
        st.success("เริ่มตรวจคลิปที่รอคิวแล้ว")

with col2:
    if st.button(
        "⏸️ พักหลังจบคลิปนี้",
        use_container_width=True,
    ):
        svc.stop()


st.divider()

st.subheader("③ รายชื่อไฟล์และผลตรวจในรอบนี้")


@st.fragment(run_every="3s")
def results_panel(batch_id):
    rows = svc.jobs(batch_id)
    batch = svc.batch(batch_id)

    done = sum(
        j["status"] == "DONE"
        for j in rows
    )

    passed = sum(
        j["verdict"] == "PASS"
        for j in rows
    )

    failed = sum(
        j["verdict"] == "FAIL"
        for j in rows
    )

    pending = sum(
        j["status"] in ("QUEUED", "RUNNING")
        for j in rows
    )

    problems = sum(
        j["status"] in ("ERROR", "WAITING")
        for j in rows
    )

    cols = st.columns(5)

    for col, name, value in zip(
        cols,
        [
            "ทั้งหมด",
            "เสร็จแล้ว",
            "✅ PASS",
            "❌ FAIL",
            "⚠️ ติดขัด",
        ],
        [
            len(rows),
            done,
            passed,
            failed,
            problems,
        ],
    ):
        col.metric(name, value)

    st.progress(
        done / len(rows) if rows else 0,
        text=f"ตรวจเสร็จ {done}/{len(rows)} คลิป",
    )

    if pending and svc.paused.is_set():
        st.warning(
            "มีคลิปรอคิว แต่คิวกำลังพัก — กดเริ่มตรวจได้"
        )

    if problems:
        st.error(
            f"มี {problems} คลิปที่ตรวจไม่สำเร็จ/"
            "รอโควตา (ไม่ใช่ FAIL)"
        )

        if st.button(
            "🔄 ลองตรวจเฉพาะคลิปที่ติดขัดอีกครั้ง",
            key=f"retry_{batch_id}",
            disabled=not bool(API_KEY),
        ):
            svc.retry(batch_id)
            st.rerun()

    query = st.text_input(
        "ค้นหาชื่อคลิป",
        placeholder="เช่น 001.mp4",
        key=f"search_{batch_id}",
    )

    choice = st.radio(
        "แสดงผล",
        [
            "ทั้งหมด",
            "PASS",
            "FAIL",
            "รอ/ผิดพลาด",
        ],
        horizontal=True,
        key=f"filter_{batch_id}",
    )

    filtered = [
        j for j in rows
        if query.lower() in j["filename"].lower()
        and (
            choice == "ทั้งหมด"
            or j["verdict"] == choice
            or (
                choice == "รอ/ผิดพลาด"
                and j["status"] != "DONE"
            )
        )
    ]

    st.dataframe(
        [
            {
                "ไฟล์": f"{i + 1:03d}. {j['filename']}",
                "ผล": j["verdict"] or j["status"],
                "ประเภท": {
                    "HUMAN": "คนขาย",
                    "HANDS": "มือโชว์",
                    "PRODUCT": "โชว์สินค้า",
                    "MIXED": "ผสม",
                }.get(j["category"], "—"),
                "เหตุผล": (
                    j["reason"]
                    or j["error"]
                    or "—"
                ),
                "เวลา(s)": j["elapsed"] or 0,
            }
            for i, j in enumerate(filtered)
        ],
        use_container_width=True,
        hide_index=True,
        height=360,
    )

    if filtered:
        options = {
            f"{j['filename']} [{j['id'][:7]}]": j
            for j in filtered
        }

        picked = st.selectbox(
            "ดูวิดีโอและสาเหตุแบบละเอียด",
            list(options),
            key=f"pick_{batch_id}",
        )

        job = options[picked]

        with st.expander(
            "▶️ เปิดคลิป / รายละเอียด",
            expanded=False,
        ):
            if Path(job["path"]).is_file():
                st.video(job["path"])
            else:
                st.warning(
                    "ไม่พบไฟล์ต้นฉบับบนเซิร์ฟเวอร์"
                )

            thumbnail = (
                THUMBS / f"{job['sha256']}.jpg"
            )

            if thumbnail.exists():
                st.image(
                    str(thumbnail),
                    width=240,
                )

            st.write(
                "**ผลตรวจ:**",
                job["verdict"] or job["status"],
            )

            st.write(
                "**สาเหตุ:**",
                job["reason"]
                or job["error"]
                or "รอการตรวจ",
            )

            if job["details"]:
                try:
                    details = json.loads(
                        job["details"]
                    )

                    st.write(
                        "**คำพูด:**",
                        details.get(
                            "speech_summary_th",
                            "",
                        ),
                    )

                    for issue in details.get(
                        "issues", []
                    ):
                        st.write(
                            f"~{issue.get('time_sec', '?')}s: "
                            f"{issue.get('detail_th', '')}"
                        )

                except (ValueError, TypeError):
                    st.warning(
                        "ข้อมูลรายละเอียดไม่สมบูรณ์"
                    )

    if rows:
        st.download_button(
            "📄 ดาวน์โหลดผลรอบนี้ (CSV)",
            report_csv(rows),
            file_name=f"video_qc_{batch_id}.csv",
            mime="text/csv",
            key=f"csv_{batch_id}",
        )

        for verdict in ("PASS", "FAIL"):
            eligible = [
                j for j in rows
                if (
                    j["verdict"] == verdict
                    and Path(j["path"]).is_file()
                )
            ]

            chunks = []
            chunk = []
            size = 0

            for j in eligible:
                if (
                    chunk
                    and size + j["size"]
                    > 100 * 1024**2
                ):
                    chunks.append(chunk)
                    chunk = []
                    size = 0

                chunk.append(j)
                size += j["size"]

            if chunk:
                chunks.append(chunk)

            for index, group in enumerate(
                chunks, 1
            ):
                if sum(
                    j["size"] for j in group
                ) > 150 * 1024**2:
                    st.warning(
                        f"ไฟล์ {group[0]['filename']} "
                        "ใหญ่เกินสำหรับ ZIP ภายในแอป"
                    )
                    continue

                if st.button(
                    f"📦 เตรียม {verdict} ZIP "
                    f"ส่วน {index}/{len(chunks)}",
                    key=(
                        f"make_{batch_id}_"
                        f"{verdict}_{index}"
                    ),
                ):
                    buf = io.BytesIO()

                    with zipfile.ZipFile(
                        buf,
                        mode="w",
                        compression=zipfile.ZIP_STORED,
                    ) as z:
                        for item in group:
                            z.write(
                                item["path"],
                                arcname=(
                                    f"{item['id'][:6]}_"
                                    f"{item['filename']}"
                                ),
                            )

                    st.session_state.zip_ready = (
                        batch_id,
                        verdict,
                        index,
                        buf.getvalue(),
                    )

        ready = st.session_state.get("zip_ready")

        if ready and ready[0] == batch_id:
            st.download_button(
                f"⬇️ ดาวน์โหลด {ready[1]} "
                f"ส่วน {ready[2]}",
                ready[3],
                file_name=(
                    f"{batch_id}_{ready[1]}_"
                    f"{ready[2]}.zip"
                ),
                mime="application/zip",
                key=f"download_zip_{batch_id}",
            )

    st.divider()

    if batch and batch["state"] == "CLOSED":
        st.success(
            "🔒 รอบนี้ปิดแล้ว — ผลตรวจยังเปิดดูได้"
            "จากรายการรอบย้อนหลัง"
        )

    elif rows and done == len(rows):
        st.success(
            "🎉 ตรวจครบทุกคลิปในรอบนี้แล้ว"
        )

        if st.button(
            "✅ ปิดรอบนี้และเริ่มรอบถัดไป",
            type="primary",
            key=f"next_{batch_id}",
        ):
            try:
                svc.close_batch(batch_id)
                new_id = svc.create_batch()

                change_to_batch(new_id)

                st.session_state.upload_summary = (
                    "เริ่มรอบใหม่แล้ว รอบเก่า"
                    "ถูกเก็บในประวัติ ไม่ได้ลบไฟล์"
                )
                st.session_state.upload_summary_batch = (
                    new_id
                )
                st.rerun()

            except ValueError as exc:
                st.error(str(exc))

    else:
        st.caption(
            "เมื่อทุกคลิปตรวจสำเร็จ ปุ่ม "
            "‘ปิดรอบนี้และเริ่มรอบถัดไป’ "
            "จะปรากฏที่นี่"
        )


results_panel(selected)

st.caption(
    "ข้อควรระวัง: ไฟล์และประวัติบน "
    "Community Cloud อาจหายหากเซิร์ฟเวอร์รีเซ็ต "
    "โปรดดาวน์โหลด CSV/ZIP ทุกครั้งเมื่อจบรอบ"
)
