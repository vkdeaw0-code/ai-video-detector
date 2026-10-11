
"""AI Video QC — One-Drop MP4 + JSON bulk uploader."""
from __future__ import annotations

import csv
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
from qc_core import ALLOWED_MODELS
from qc_store import (
    EXPORTS,
    MAX_ACTIVE_BYTES,
    MAX_VIDEOS,
    ActiveQueue,
)

bulk_component = components.declare_component(
    "bulk_json_mp4_one_drop",
    path=str(Path(__file__).parent / "bulk_uploader"),
)

st.set_page_config(
    page_title="AI Video QC | Bulk Upload",
    page_icon="🎬",
    layout="wide",
)


def secret(name):
    try:
        return str(
            st.secrets.get(name, "")
            or os.getenv(name, "")
        )
    except (FileNotFoundError, KeyError):
        return str(os.getenv(name, ""))


PASSWORD = secret("APP_PASSWORD")
API_KEY = secret("GEMINI_API_KEY")

if not PASSWORD:
    st.error(
        "กรุณาตั้ง APP_PASSWORD "
        "ใน Streamlit Secrets"
    )
    st.stop()

if not st.session_state.get("authorized"):
    st.title("🔐 AI Video QC")

    typed = st.text_input(
        "รหัสผ่านเข้าเว็บ",
        type="password",
    )

    if st.button("เข้าสู่ระบบ", type="primary"):
        if hmac.compare_digest(typed, PASSWORD):
            st.session_state.authorized = True
            st.rerun()
        else:
            st.error("รหัสผ่านไม่ถูกต้อง")

    st.stop()


@st.cache_resource
def get_service(version):
    return ActiveQueue(API_KEY)


svc = get_service("json-match-v2")
svc.update_key(API_KEY)


@st.cache_resource
def get_receiver(version):
    return ChunkReceiver(svc)


receiver = get_receiver("chunk-stream-v1")


def make_csv(rows):
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "video_filename",
        "json_filename",
        "product_title",
        "status",
        "decision",
        "video_type",
        "reason",
        "speech_summary_th",
        "elapsed_seconds",
        "audit_done",
        "audit_trigger",
        "error",
    ])

    for row in rows:
        try:
            details = json.loads(
                row.get("details") or "{}"
            )
        except (TypeError, ValueError):
            details = {}

        writer.writerow([
            row["filename"],
            row.get("matched_json") or "",
            row.get("product_title") or "",
            row["status"],
            row.get("verdict") or "",
            row.get("category") or "",
            row.get("reason") or "",
            details.get("speech_summary_th", ""),
            row.get("elapsed") or "",
            details.get("audit_done", ""),
            details.get("audit_trigger", ""),
            row.get("error") or "",
        ])

    return output.getvalue().encode("utf-8-sig")


with st.sidebar:
    st.header("⚙️ ตั้งค่าการตรวจ")

    mode = st.selectbox(
        "ความละเอียด",
        ["Balanced", "Fast", "Deep"],
    )

    models = sorted(ALLOWED_MODELS)

    if not models:
        st.error("ไม่พบโมเดลใน qc_core.py")
        st.stop()

    preferred = "gemini-2.5-flash-lite"

    model = st.selectbox(
        "Gemini Model",
        models,
        index=(
            models.index(preferred)
            if preferred in models else 0
        ),
    )

    auto_start = st.checkbox(
        "เริ่มตรวจอัตโนมัติเมื่อจับคู่ครบ",
        value=True,
    )

    if st.button(
        "▶️ เริ่ม / ทำงานต่อ",
        disabled=not API_KEY,
        use_container_width=True,
    ):
        svc.start()

    if st.button(
        "⏸️ พักคิวตรวจ",
        use_container_width=True,
    ):
        svc.pause()

    st.caption(
        "พื้นที่ดิสก์ว่าง: "
        f"{svc.free_disk() / 1024**3:.2f} GB"
    )


st.title("🎬 AI Video QC — Bulk JSON + MP4")

st.caption(
    "ลากไฟล์ครั้งเดียว • จับคู่สินค้าอัตโนมัติ "
    "• ตรวจทีละคลิป • ไม่มี Cache ผลเก่า"
)

if not API_KEY:
    st.warning(
        "ยังไม่มี GEMINI_API_KEY "
        "ใน Streamlit Secrets"
    )


# ==================================================
# DASHBOARD
# ==================================================

count, used, references = svc.stats()
rows = svc.jobs()

matched = sum(
    bool(row.get("matched_json"))
    for row in rows
)

st.subheader("① ข้อมูลรอบปัจจุบัน")

c1, c2, c3, c4 = st.columns(4)

c1.metric("วิดีโอ", count)
c2.metric("JSON", references)
c3.metric("จับคู่แล้ว", f"{matched}/{count}")
c4.metric(
    "พื้นที่วิดีโอ",
    f"{used / 1024**2:.0f}/"
    f"{MAX_ACTIVE_BYTES / 1024**2:.0f} MB",
)

st.progress(
    min(1.0, used / MAX_ACTIVE_BYTES),
    text=f"สูงสุด {MAX_VIDEOS} คลิปต่อรอบ",
)


# ==================================================
# ONE-DROP UPLOADER
# ==================================================

st.subheader("② ลาก MP4 + JSON ทั้งหมดครั้งเดียว")

st.info(
    "ลาก MP4 และ JSON ทั้งหมดลงช่องด้านล่าง "
    "ระบบจะส่งไฟล์เป็นส่วน ๆ ขนาด 1.5MB "
    "จับคู่ JSON ด้วยชื่อวิดีโอ "
    "และเริ่มตรวจอัตโนมัติ"
)

st.caption(
    "ต้องเปิดหน้าเว็บค้างไว้ระหว่างอัปโหลด "
    "และ video_filename ใน JSON "
    "ต้องตรงกับชื่อ MP4 จริง"
)

if "bulk_nonce" not in st.session_state:
    st.session_state.bulk_nonce = uuid.uuid4().hex

if "bulk_ack" not in st.session_state:
    st.session_state.bulk_ack = None

if "bulk_last_id" not in st.session_state:
    st.session_state.bulk_last_id = ""

event = bulk_component(
    nonce=st.session_state.bulk_nonce,
    ack=st.session_state.bulk_ack,
    key="bulk_json_mp4_input",
    default=None,
)

if isinstance(event, dict):
    event_id = str(event.get("id", ""))

    if (
        event_id
        and event_id != st.session_state.bulk_last_id
    ):
        try:
            reply = receiver.handle(
                event,
                session_id=st.session_state.bulk_nonce,
                mode=mode,
                model=model,
                auto_start=bool(
                    API_KEY and auto_start
                ),
            )

            ack = {
                "id": event_id,
                **reply,
            }

        except Exception as exc:
            ack = {
                "id": event_id,
                "ok": False,
                "message": str(exc)[:450],
            }

        st.session_state.bulk_last_id = event_id
        st.session_state.bulk_ack = ack

        st.rerun()


waiting_json = [
    row["filename"]
    for row in rows
    if row["status"] == "WAIT_JSON"
]

if waiting_json:
    st.warning(
        "คลิปที่ยังไม่มี JSON ตรงกัน: "
        + ", ".join(waiting_json[:12])
    )

unmatched = svc.unmatched_references()

if unmatched:
    with st.expander(
        f"JSON ที่ยังไม่มีวิดีโอ "
        f"{len(unmatched)} ไฟล์"
    ):
        st.dataframe(
            unmatched,
            hide_index=True,
            use_container_width=True,
        )


col1, col2 = st.columns(2)

with col1:
    if st.button(
        "▶️ เริ่มตรวจคลิปที่จับคู่แล้ว",
        type="primary",
        disabled=not API_KEY,
        use_container_width=True,
    ):
        svc.start()
        st.rerun()

with col2:
    if st.button(
        "⏸️ พักหลังจบคลิปปัจจุบัน",
        use_container_width=True,
    ):
        svc.pause()
        st.rerun()


# ==================================================
# RESULTS
# ==================================================

@st.fragment(run_every="4s")
def results_panel():
    current = svc.jobs()

    done = sum(
        row["status"] == "DONE"
        for row in current
    )

    passed = sum(
        row["verdict"] == "PASS"
        for row in current
    )

    failed = sum(
        row["verdict"] == "FAIL"
        for row in current
    )

    errors = sum(
        row["status"] in ("ERROR", "WAITING")
        for row in current
    )

    need_json = sum(
        row["status"] == "WAIT_JSON"
        for row in current
    )

    st.divider()
    st.subheader("③ ผลตรวจ")

    columns = st.columns(5)

    for col, title, value in zip(
        columns,
        [
            "ทั้งหมด",
            "ตรวจเสร็จ",
            "✅ PASS",
            "❌ FAIL",
            "⚠️ ติดขัด",
        ],
        [
            len(current),
            done,
            passed,
            failed,
            errors,
        ],
    ):
        col.metric(title, value)

    st.progress(
        done / len(current) if current else 0,
        text=(
            f"ตรวจแล้ว {done}/{len(current)} คลิป "
            f"| รอ JSON {need_json}"
        ),
    )

    if errors:
        st.error(
            f"มี {errors} งานที่ตรวจไม่สำเร็จ "
            "ไม่ใช่คุณภาพ FAIL"
        )

        if st.button(
            "🔄 ลองตรวจงานที่ผิดพลาดอีกครั้ง",
            disabled=not API_KEY,
        ):
            svc.retry()
            st.rerun()

    if current:
        search = st.text_input(
            "ค้นหาชื่อไฟล์หรือชื่อสินค้า",
            placeholder="001.mp4",
        )

        group_filter = st.radio(
            "แสดงผล",
            [
                "ทั้งหมด",
                "PASS",
                "FAIL",
                "รอ JSON",
                "ผิดพลาด",
            ],
            horizontal=True,
        )

        displayed = []

        for row in current:
            search_text = (
                row["filename"]
                + " "
                + (row.get("product_title") or "")
            ).casefold()

            if search.casefold() not in search_text:
                continue

            if (
                group_filter == "PASS"
                and row["verdict"] != "PASS"
            ):
                continue

            if (
                group_filter == "FAIL"
                and row["verdict"] != "FAIL"
            ):
                continue

            if (
                group_filter == "รอ JSON"
                and row["status"] != "WAIT_JSON"
            ):
                continue

            if (
                group_filter == "ผิดพลาด"
                and row["status"]
                not in ("ERROR", "WAITING")
            ):
                continue

            displayed.append(row)

        st.dataframe(
            [
                {
                    "ไฟล์": row["filename"],
                    "JSON": (
                        row.get("matched_json")
                        or "ยังไม่พบ"
                    ),
                    "สินค้า": (
                        row.get("product_title")
                        or "—"
                    ),
                    "ผล": (
                        row["verdict"]
                        or row["status"]
                    ),
                    "เหตุผล": (
                        row["reason"]
                        or row["error"]
                        or "—"
                    ),
                    "เวลา(s)": row["elapsed"] or 0,
                }
                for row in displayed
            ],
            use_container_width=True,
            hide_index=True,
        )

        if displayed:
            options = {
                (
                    f"{row['filename']} "
                    f"[{row['id'][:6]}]"
                ): row
                for row in displayed
            }

            selected = options[
                st.selectbox(
                    "เปิดรายละเอียดคลิป",
                    list(options),
                )
            ]

            with st.expander(
                "▶️ วิดีโอและรายละเอียด"
            ):
                path = Path(selected["path"])

                if path.is_file():
                    st.video(str(path))

                st.write(
                    "**สินค้าใน JSON:**",
                    selected.get("product_title")
                    or "ยังไม่จับคู่",
                )

                st.write(
                    "**ชื่อ JSON:**",
                    selected.get("matched_json")
                    or "ไม่พบ",
                )

                if selected.get("product_reference"):
                    st.text(
                        selected["product_reference"]
                    )

                st.write(
                    "**ผล:**",
                    selected["verdict"]
                    or selected["status"],
                )

                st.write(
                    "**เหตุผล:**",
                    selected["reason"]
                    or selected["error"]
                    or "รอตรวจ",
                )

                if selected.get("details"):
                    try:
                        report = json.loads(
                            selected["details"]
                        )

                        st.write(
                            "**สรุปคำพูด:**",
                            report.get(
                                "speech_summary_th",
                                "",
                            ),
                        )

                        with st.expander(
                            "หลักฐานตรวจละเอียด"
                        ):
                            st.json({
                                "scale_review": report.get(
                                    "scale_review"
                                ),
                                "scale_audit": report.get(
                                    "scale_audit"
                                ),
                                "audio_review": report.get(
                                    "audio_review"
                                ),
                                "speech_audit": report.get(
                                    "speech_audit"
                                ),
                                "audit_trigger": report.get(
                                    "audit_trigger"
                                ),
                            })

                    except (TypeError, ValueError):
                        st.warning(
                            "อ่านรายละเอียดผลตรวจไม่ได้"
                        )

        st.download_button(
            "📄 ดาวน์โหลด CSV",
            make_csv(current),
            file_name="qc_report.csv",
            mime="text/csv",
        )

        st.markdown(
            "**ดาวน์โหลด MP4 แยก PASS / FAIL**"
        )

        verdict_group = st.selectbox(
            "กลุ่มไฟล์",
            ["PASS", "FAIL"],
            key="zip_group",
        )

        eligible = [
            row for row in current
            if (
                row["verdict"] == verdict_group
                and Path(row["path"]).is_file()
            )
        ]

        groups = []
        chunk = []
        chunk_bytes = 0

        for row in eligible:
            if (
                chunk
                and chunk_bytes + row["size"]
                > 35 * 1024**2
            ):
                groups.append(chunk)
                chunk = []
                chunk_bytes = 0

            chunk.append(row)
            chunk_bytes += row["size"]

        if chunk:
            groups.append(chunk)

        if groups:
            index = st.selectbox(
                "ชุด ZIP",
                range(len(groups)),
                format_func=lambda i: (
                    f"ชุดที่ {i + 1}/{len(groups)} "
                    f"({len(groups[i])} คลิป)"
                ),
            )

            if st.button(
                "📦 เตรียม ZIP"
            ):
                destination = (
                    EXPORTS / "current_download.zip"
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
                    verdict_group,
                    index,
                    str(destination),
                )

            ready = st.session_state.get(
                "export_ready"
            )

            if (
                ready
                and ready[:2] == (
                    verdict_group, index
                )
                and Path(ready[2]).is_file()
            ):
                with open(ready[2], "rb") as file:
                    st.download_button(
                        "⬇️ ดาวน์โหลด ZIP",
                        file,
                        file_name=(
                            f"qc_{verdict_group}_"
                            f"{index + 1}.zip"
                        ),
                        mime="application/zip",
                    )

    st.divider()
    st.subheader("④ ล้างข้อมูลและเริ่มรอบใหม่")

    st.warning(
        "ดาวน์โหลด CSV และ ZIP ก่อนล้างรอบ "
        "เพราะไฟล์จะถูกลบ"
    )

    confirm = st.checkbox(
        "ฉันดาวน์โหลดข้อมูลที่ต้องการแล้ว"
    )

    if st.button(
        "🗑️ ลบข้อมูลรอบนี้ทั้งหมด",
        disabled=not confirm,
    ):
        try:
            svc.clear_round()
            receiver.clear_staging()

            st.session_state.bulk_nonce = (
                uuid.uuid4().hex
            )
            st.session_state.bulk_ack = None
            st.session_state.bulk_last_id = ""

            st.session_state.pop(
                "export_ready", None
            )

            st.rerun()

        except ValueError as exc:
            st.error(str(exc))


results_panel()
