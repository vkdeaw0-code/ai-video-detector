
"""V3.9.1 Streamlit MP4-only bulk video QC dashboard."""

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

from qc_core import (
    ENGINE_VERSION,
    MODELS,
    DEFAULT_MODEL,
    available_models,
    AnalysisError,
)

from qc_store import (
    ActiveQueue,
    EXPORTS,
    MAX_ACTIVE_BYTES,
    MAX_VIDEOS,
)


st.set_page_config(
    page_title="Video QC V3.9.1",
    page_icon="🎬",
    layout="wide",
)


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


# ==========================================
# LOGIN
# ==========================================

if not password:
    st.error(
        "กรุณาตั้ง APP_PASSWORD "
        "ใน Streamlit Secrets"
    )
    st.stop()


if not st.session_state.get("authorized"):
    st.title(
        "🔐 Video QC V3.9.1"
    )

    entry = st.text_input(
        "รหัสผ่านเว็บ",
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
# QUEUE
# ==========================================

@st.cache_resource
def get_service(version):
    return ActiveQueue(api_key)


service = get_service(
    "qc391-mp4-only"
)

service.update_key(api_key)


@st.cache_resource
def get_receiver(version):
    return ChunkReceiver(service)


receiver = get_receiver(
    "qc391-mp4-chunks"
)


bulk = components.declare_component(
    "bulk_mp4_only_v391",
    path=str(
        Path(__file__).parent
        / "bulk_uploader"
    ),
)


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def model_choices(key):
    return available_models(key)


def report_csv(rows):
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "filename",
        "status",
        "decision",
        "product_type",
        "reason",
        "speech",
        "seconds",
        "model",
        "second_ai_call",
        "engine",
        "error",
    ])

    for row in rows:
        try:
            details = json.loads(
                row.get("details")
                or "{}"
            )

        except (
            TypeError,
            ValueError,
        ):
            details = {}

        writer.writerow([
            row["filename"],
            row["status"],
            row.get("verdict") or "",
            (
                details.get("product_type_th")
                or row.get("category")
                or ""
            ),
            row.get("reason") or "",
            (
                details.get("speech_summary_th")
                or ""
            ),
            row.get("elapsed") or "",
            row.get("model") or "",
            details.get("audit_done") or False,
            details.get("engine") or "",
            row.get("error") or "",
        ])

    return output.getvalue().encode(
        "utf-8-sig"
    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header(
        "⚙️ ตั้งค่าตรวจ"
    )

    mode = st.selectbox(
        "ความละเอียดของภาพ",
        ["Balanced", "Fast", "Deep"],
    )

    try:
        found_models = (
            model_choices(api_key)
            if api_key
            else []
        )

    except AnalysisError as exc:
        found_models = []

        st.warning(
            str(exc)[:200]
        )

    if found_models:
        options = found_models

        st.caption(
            "โมเดลที่ Google แจ้งว่า "
            "API Key นี้เข้าถึงได้"
        )

    else:
        options = list(MODELS)

        st.caption(
            "ตรวจสอบสิทธิ์โมเดลไม่ได้ "
            "เลือกจากรายการทางการ"
        )

    model = st.selectbox(
        "Gemini model",
        options,
        index=(
            options.index(DEFAULT_MODEL)
            if DEFAULT_MODEL in options
            else 0
        ),
    )

    auto_start = st.checkbox(
        "เริ่มตรวจเมื่ออัปโหลด MP4 เสร็จ",
        value=True,
    )

    if st.button(
        "🔁 ใช้โมเดลนี้กับคลิปที่รอ/ผิดพลาด",
        use_container_width=True,
    ):
        service.set_model(model)

        st.success(
            "เปลี่ยนโมเดลในคิวเรียบร้อย"
        )

    if st.button(
        "▶️ เริ่ม / ทำงานต่อ",
        disabled=not api_key,
        use_container_width=True,
    ):
        service.start()

    if st.button(
        "⏸️ พักหลังจบคลิปปัจจุบัน",
        use_container_width=True,
    ):
        service.pause()

    if st.button(
        "🔍 ตรวจสิทธิ์โมเดลใหม่",
        use_container_width=True,
    ):
        model_choices.clear()
        st.rerun()

    st.caption(
        f"ดิสก์ว่าง "
        f"{service.free_disk() / 1024**3:.2f} GB"
    )


# ==========================================
# MAIN DASHBOARD
# ==========================================

st.title(
    "🎬 AI Video QC V3.9.1 — MP4 Only"
)

st.caption(
    "ไม่ใช้ JSON • ประเมินสเกล เสียงพูด "
    "คำพูดตรงภาพ รูปทรง มือและกระจก • "
    "PASS / FAIL / REVIEW"
)

if not api_key:
    st.error(
        "ยังไม่ได้ตั้ง GEMINI_API_KEY "
        "ใน Streamlit Secrets"
    )


count, used, _ = service.stats()

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "MP4", count
)

metric2.metric(
    "พื้นที่คลิป",
    (
        f"{used / 1024**2:.0f}/"
        f"{MAX_ACTIVE_BYTES / 1024**2:.0f} MB"
    ),
)

metric3.metric(
    "โหมด", mode
)

st.progress(
    min(
        used / MAX_ACTIVE_BYTES,
        1.0,
    ),
    text=(
        f"สูงสุด {MAX_VIDEOS} คลิปต่อรอบ "
        "ตามพื้นที่จริงของเซิร์ฟเวอร์"
    ),
)


# ==========================================
# BULK UPLOAD
# ==========================================

st.subheader(
    "① ลากหรือเลือก MP4 หลายไฟล์ในครั้งเดียว"
)

st.info(
    "ไม่ต้องเตรียม JSON ระบบอ่านภาพและเสียง "
    "จากคลิปโดยตรง เปิดหน้าเว็บนี้ไว้ระหว่างอัปโหลด"
)

if "bulk_nonce" not in st.session_state:
    st.session_state.bulk_nonce = (
        uuid.uuid4().hex
    )

if "bulk_ack" not in st.session_state:
    st.session_state.bulk_ack = None

if "bulk_last_id" not in st.session_state:
    st.session_state.bulk_last_id = ""


event = bulk(
    nonce=st.session_state.bulk_nonce,
    ack=st.session_state.bulk_ack,
    key="bulk_mp4_input_v391",
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
            answer = receiver.handle(
                event,
                session_id=st.session_state.bulk_nonce,
                mode=mode,
                model=model,
                auto_start=bool(
                    api_key and auto_start
                ),
            )

            ack = {
                "id": event_id,
                **answer,
            }

        except Exception as exc:
            ack = {
                "id": event_id,
                "ok": False,
                "message": str(exc)[:450],
            }

        st.session_state.bulk_last_id = (
            event_id
        )

        st.session_state.bulk_ack = ack

        st.rerun()


# ==========================================
# RESULTS
# ==========================================

@st.fragment(run_every="5s")
def results_panel():
    rows = service.jobs()

    done = sum(
        x["status"] == "DONE"
        for x in rows
    )

    passed = sum(
        x.get("verdict") == "PASS"
        for x in rows
    )

    failed = sum(
        x.get("verdict") == "FAIL"
        for x in rows
    )

    reviews = sum(
        x.get("verdict") == "REVIEW"
        for x in rows
    )

    errors = sum(
        x["status"] in (
            "ERROR", "WAITING"
        )
        for x in rows
    )

    st.divider()
    st.subheader("② ผลตรวจ")

    cols = st.columns(6)

    titles = (
        "ทั้งหมด",
        "เสร็จ",
        "✅ PASS",
        "❌ FAIL",
        "👁 REVIEW",
        "⚠ ERROR/โควตา",
    )

    values = (
        len(rows),
        done,
        passed,
        failed,
        reviews,
        errors,
    )

    for col, title, value in zip(
        cols,
        titles,
        values,
    ):
        col.metric(
            title, value
        )

    st.progress(
        (
            done / len(rows)
            if rows
            else 0
        ),
        text=(
            f"ตรวจแล้ว {done}/{len(rows)} คลิป"
        ),
    )

    if errors:
        st.error(
            f"มี {errors} คลิปตรวจไม่สำเร็จ "
            "(ไม่ใช่ผล FAIL)"
        )

        if st.button(
            "🔄 ตรวจคลิปที่ ERROR อีกครั้ง "
            "โดยใช้โมเดลที่เลือก",
            disabled=not api_key,
        ):
            service.retry(model)
            st.rerun()

    if reviews:
        st.warning(
            f"มี {reviews} คลิปต้องตรวจด้วยคน "
            "ก่อนเผยแพร่"
        )

    if not rows:
        return

    # ======================================
    # FILTER
    # ======================================

    query = st.text_input(
        "ค้นหาชื่อ MP4 หรือประเภทสินค้า",
        key="qc_search",
    )

    selected_filter = st.radio(
        "กรองผล",
        [
            "ทั้งหมด",
            "PASS",
            "FAIL",
            "REVIEW",
            "กำลังตรวจ",
            "ERROR",
        ],
        horizontal=True,
        key="qc_filter",
    )

    filtered = []

    for r in rows:
        searchable = (
            r["filename"]
            + " "
            + str(r.get("category") or "")
        ).casefold()

        if query.casefold() not in searchable:
            continue

        if (
            selected_filter in (
                "PASS",
                "FAIL",
                "REVIEW",
            )
            and r.get("verdict")
            != selected_filter
        ):
            continue

        if (
            selected_filter == "กำลังตรวจ"
            and r["status"]
            not in (
                "RUNNING",
                "QUEUED",
            )
        ):
            continue

        if (
            selected_filter == "ERROR"
            and r["status"]
            not in (
                "ERROR",
                "WAITING",
            )
        ):
            continue

        filtered.append(r)

    st.dataframe(
        [
            {
                "MP4": r["filename"],
                "ผล": (
                    r.get("verdict")
                    or r["status"]
                ),
                "สินค้า": (
                    r.get("category")
                    or "—"
                ),
                "เหตุผล": (
                    r.get("reason")
                    or r.get("error")
                    or "—"
                ),
                "เวลา(s)": round(
                    float(
                        r.get("elapsed") or 0
                    ),
                    2,
                ),
            }
            for r in filtered
        ],
        use_container_width=True,
        hide_index=True,
    )

    # ======================================
    # DETAIL
    # ======================================

    if filtered:
        choices = {
            (
                f"{r['filename']} "
                f"[{r['id'][:6]}]"
            ): r
            for r in filtered
        }

        name = st.selectbox(
            "ดูวิดีโอและหลักฐานการตรวจ",
            list(choices),
            key="qc_detail_select",
        )

        row = choices[name]

        with st.expander(
            "▶️ เปิดคลิป / ผลตรวจแบบละเอียด"
        ):
            if Path(
                row["path"]
            ).is_file():
                st.video(
                    row["path"]
                )

            st.write(
                "**เหตุผล:**",
                (
                    row.get("reason")
                    or row.get("error")
                    or "รอประมวลผล"
                ),
            )

            if row.get("details"):
                try:
                    detail = json.loads(
                        row["details"]
                    )

                    st.write(
                        "**สินค้าที่ตรวจพบ:**",
                        detail.get(
                            "product_type_th", ""
                        ),
                    )

                    st.write(
                        "**คำพูดโดยสรุป:**",
                        detail.get(
                            "speech_summary_th", ""
                        ),
                    )

                    st.json({
                        k: detail.get(k)
                        for k in (
                            "scale_review",
                            "audio_review",
                            "match_review",
                            "visual_review",
                            "mirror_review",
                            "scale_audit",
                            "speech_audit",
                            "mirror_audit",
                            "issues",
                            "timings",
                            "engine",
                        )
                    })

                except (
                    ValueError,
                    TypeError,
                ):
                    st.warning(
                        "อ่านรายละเอียดผลตรวจไม่ได้"
                    )

    # ======================================
    # CSV
    # ======================================

    st.download_button(
        "📄 ดาวน์โหลด CSV",
        report_csv(rows),
        file_name="qc_391_report.csv",
        mime="text/csv",
    )

    # ======================================
    # ZIP
    # ======================================

    st.markdown(
        "**ดาวน์โหลด MP4 ตามกลุ่ม "
        "PASS / FAIL / REVIEW**"
    )

    verdict = st.selectbox(
        "กลุ่ม ZIP",
        ["PASS", "FAIL", "REVIEW"],
        key="qc_zip_verdict",
    )

    eligible = [
        r for r in rows
        if (
            r.get("verdict") == verdict
            and Path(r["path"]).is_file()
        )
    ]

    groups = []
    group = []
    size = 0

    for r in eligible:
        if (
            group
            and (
                size + r["size"]
                > 35 * 1024**2
            )
        ):
            groups.append(group)
            group = []
            size = 0

        group.append(r)
        size += r["size"]

    if group:
        groups.append(group)

    if groups:
        index = st.selectbox(
            "แบ่ง ZIP ตามขนาด",
            range(len(groups)),
            format_func=lambda n: (
                f"ชุด {n + 1}/{len(groups)} "
                f"({len(groups[n])} คลิป)"
            ),
            key="qc_zip_index",
        )

        identity = hashlib.sha256(
            (
                verdict
                + "|"
                + "|".join(
                    r["id"]
                    for r in groups[index]
                )
            ).encode()
        ).hexdigest()[:14]

        dest = EXPORTS / (
            f"qc391_{verdict}_{identity}.zip"
        )

        if st.button(
            "📦 เตรียม ZIP",
            key="qc_zip_build",
        ):
            with zipfile.ZipFile(
                dest,
                "w",
                compression=zipfile.ZIP_STORED,
            ) as z:
                for r in groups[index]:
                    z.write(
                        r["path"],
                        arcname=r["filename"],
                    )

            st.session_state.qc_export_ready = (
                str(dest)
            )

        if (
            st.session_state.get(
                "qc_export_ready"
            ) == str(dest)
            and dest.is_file()
        ):
            with dest.open("rb") as file:
                st.download_button(
                    "⬇️ ดาวน์โหลด ZIP",
                    file,
                    file_name=(
                        f"qc_{verdict}_{index + 1}.zip"
                    ),
                    mime="application/zip",
                    key="qc_zip_download",
                )

    else:
        st.caption(
            "ยังไม่มีผลในกลุ่มนี้"
        )

    # ======================================
    # CLEAR ROUND
    # ======================================

    st.divider()

    st.subheader(
        "③ ล้างข้อมูลและเริ่มรอบใหม่"
    )

    st.warning(
        "ดาวน์โหลด CSV/ZIP ก่อนกดล้างรอบ: "
        "คลิปและผลที่เก็บในเซิร์ฟเวอร์จะถูกลบ"
    )

    confirmed = st.checkbox(
        "ฉันดาวน์โหลดข้อมูลที่ต้องการแล้ว",
        key="qc_clear_confirm",
    )

    if st.button(
        "🗑️ ล้างรอบทั้งหมด",
        disabled=not confirmed,
        key="qc_clear",
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
                "qc_export_ready",
                None,
            )

            st.rerun()

        except ValueError as exc:
            st.error(
                str(exc)
            )


results_panel()

st.caption(
    f"รุ่นตัวตรวจ: {ENGINE_VERSION} • "
    "ไม่มี JSON • AI อาจตรวจผิดได้ • "
    "150 คลิปเป็นเพดานซอฟต์แวร์ "
    "ไม่ใช่การรับประกันทรัพยากรคลาวด์"
)
