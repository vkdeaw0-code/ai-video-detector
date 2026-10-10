import io
import json
import os
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from profiles import PROFILES
from engine import (
    analyze_clip,
    flatten_report,
    public_report,
    CATEGORIES,
)


st.set_page_config(
    page_title="AI Clip QC",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 คัดคลิป AI ตามประเภทสินค้า")
st.caption(
    "ของใช้: เทียบสเกลกับคำพูด · "
    "คนรีวิวสินค้าสัตว์เลี้ยง: เน้นภาษาไทย"
)


def secret(name):
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def reference_image(upload):
    if upload is None:
        return None

    image = ImageOps.exif_transpose(
        Image.open(io.BytesIO(upload.getvalue()))
    )

    return ImageOps.contain(
        image.convert("RGB"), (1300, 1300)
    )


with st.sidebar:
    st.header("ตั้งค่าการตรวจ")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=(
            os.getenv("OPENAI_API_KEY", "")
            or secret("OPENAI_API_KEY")
        ),
    )

    image_model = st.selectbox(
        "โมเดลภาพ",
        ["gpt-4.1", "gpt-4.1-mini"],
    )

    audio_model = st.text_input(
        "โมเดลฟังเสียง",
        "gpt-audio-1.5",
    )

    transcription_model = st.text_input(
        "โมเดลถอดเสียง",
        "gpt-transcribe",
    )

    sample_fps = st.slider(
        "ภาพตัวอย่างต่อวินาที",
        1.0, 4.0, 2.0, 0.5,
    )

    max_errors = st.number_input(
        "ยอมรับคำผิดไม่เกิน (คำ)",
        0, 20, 2,
    )

    overflow = st.selectbox(
        "เมื่อยืนยันคำผิดเกินเกณฑ์",
        ["FAIL", "REVIEW"],
    )

    pass_score = st.slider(
        "คะแนนขั้นต่ำผ่าน",
        75, 100, 85,
    )

    review_score = st.slider(
        "คะแนนเริ่มรอตรวจ",
        40, min(84, pass_score - 1), 70,
    )

    default_profile = st.selectbox(
        "เกณฑ์เริ่มต้น",
        list(PROFILES),
        format_func=lambda key: PROFILES[key],
    )

    st.caption(
        "คำผิดตั้งแต่ 3 คำที่ยืนยันแล้ว: "
        "ไม่ผ่านตามค่าเริ่มต้น"
    )

    if not api_key:
        st.warning(
            "ไม่มี API Key: ตรวจเทคนิคได้ "
            "แต่ผลเป็น REVIEW"
        )


with st.expander("อ่านก่อนใช้งาน"):
    st.write(
        "ระบบฟังเสียงจริง ไม่ตัดตกจากตัวสะกดที่ AI "
        "ถอดเสียงผิด ชื่อแบรนด์และสำเนียง "
        "ไม่ใช่คำผิดโดยอัตโนมัติ"
    )

    st.write(
        "ขนาดเชิงตัวเลข เช่น cm/L ต้องมีข้อมูลสินค้าจริง "
        "หรือสเกลวัดในภาพ มิฉะนั้นรอตรวจ "
        "ไม่แปลงมือเป็นหน่วยวัดตายตัว"
    )

    st.write(
        "เสียงและภาพตัวอย่างถูกส่งไปยัง API "
        "มีค่าใช้จ่ายตามการใช้ "
        "เวลาเสียงเป็นค่าประมาณ "
        "AI เห็นเฉพาะภาพที่สุ่ม"
    )


videos = st.file_uploader(
    "เลือกวิดีโอหลายไฟล์",
    type=["mp4", "mov", "avi", "webm", "mkv"],
    accept_multiple_files=True,
)

seed = {
    **{
        f"{n}.mp4": "HOUSEHOLD_HANDS"
        for n in (2, 7, 8, 10)
    },
    **{
        f"{n}.mp4": "PET_PRESENTER"
        for n in (13, 14, 15, 16, 17)
    },
}

settings, refs = [], {}

if videos:
    examples = st.checkbox(
        "ใช้ประเภทที่คุณระบุสำหรับคลิปตัวอย่าง "
        "2, 7, 8, 10 และ 13–17",
        True,
    )

    st.caption(
        "ถ้าเป็นคลิปใหม่ที่บังเอิญชื่อซ้ำ "
        "ให้ปิดช่องด้านบนหรือเลือกเกณฑ์เอง "
        "ข้อมูลสินค้าแต่ละแถวใช้เฉพาะคลิปนั้น"
    )

    rows = [
        {
            "ลำดับ": i + 1,
            "ไฟล์": f.name,
            "เกณฑ์": PROFILES[
                seed.get(f.name, default_profile)
                if examples
                else default_profile
            ],
            "ชื่อสินค้า": "",
            "ขนาด/ข้อมูลจริง": "",
            "บทพูดอ้างอิง": "",
            "คำยกเว้น": "",
        }
        for i, f in enumerate(videos)
    ]

    settings = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        disabled=["ลำดับ", "ไฟล์"],
        column_config={
            "เกณฑ์": st.column_config.SelectboxColumn(
                options=list(PROFILES.values()),
                required=True,
            )
        },
        key=(
            "settings_"
            + str(examples)
            + default_profile
            + "|".join(f.name for f in videos)
        ),
    ).to_dict("records")

    with st.expander("ภาพสินค้าอ้างอิงแยกรายคลิป (ถ้ามี)"):
        for i, f in enumerate(videos):
            refs[i] = st.file_uploader(
                f"ภาพสำหรับ {i + 1}. {f.name}",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"ref_{i}_{f.name}",
            )


if "reports" not in st.session_state:
    st.session_state.reports = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

size_mb = (
    sum(f.size for f in videos) / 1024**2
    if videos
    else 0
)

if len(videos or []) > 50 or size_mb > 350:
    st.warning(
        "แบ่งการตรวจเป็นรอบละไม่เกิน 50 คลิป "
        "และรวมไม่เกิน 350 MB"
    )


if st.button(
    "เริ่มตรวจทั้งหมด",
    type="primary",
    disabled=(
        not videos
        or len(videos) > 50
        or size_mb > 350
    ),
):
    st.session_state.reports = []
    st.session_state.feedback = {}

    bar = st.progress(0.0)
    status = st.empty()

    labels = {
        label: key
        for key, label in PROFILES.items()
    }

    for i, f in enumerate(videos):
        row = settings[i]

        try:
            report = analyze_clip(
                f.getvalue(),
                f.name,
                api_key=api_key,
                model=image_model,
                reference=reference_image(refs.get(i)),
                expected_name=row["ชื่อสินค้า"],
                profile=labels[row["เกณฑ์"]],
                product_facts=row["ขนาด/ข้อมูลจริง"],
                reference_script=row["บทพูดอ้างอิง"],
                glossary=[
                    w.strip()
                    for w in row["คำยกเว้น"].split(",")
                    if w.strip()
                ],
                sample_fps=sample_fps,
                max_word_errors=int(max_errors),
                word_overflow_action=overflow,
                pass_score=pass_score,
                review_score=review_score,
                use_transcription=True,
                use_audio_ai=True,
                audio_model=audio_model.strip(),
                transcription_model=transcription_model.strip(),
                progress=lambda msg: status.info(
                    f"{f.name}: {msg}"
                ),
            )

        except Exception as exc:
            report = {
                "filename": f.name,
                "status": "ERROR",
                "score": 0,
                "score_is_partial": True,
                "type_label": "ตรวจไม่สำเร็จ",
                "issues": [],
                "evidence": {},
                "analysis_error": str(exc),
            }

        report["upload_index"] = i
        st.session_state.reports.append(report)
        bar.progress((i + 1) / len(videos))

    status.success("ประมวลผลครบแล้ว")


reports = st.session_state.reports

if not reports:
    st.info(
        "เลือกคลิป ตั้งค่ารายแถว แล้วกดเริ่มตรวจ"
    )
    st.stop()

counts = Counter(r["status"] for r in reports)

for column, name in zip(
    st.columns(4),
    ["PASS", "FAIL", "REVIEW", "ERROR"],
):
    column.metric(name, counts.get(name, 0))

table = pd.DataFrame(
    [flatten_report(r) for r in reports]
)

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
)

st.download_button(
    "ดาวน์โหลด CSV",
    table.to_csv(index=False).encode("utf-8-sig"),
    "clip_report.csv",
    "text/csv",
)

st.download_button(
    "ดาวน์โหลด JSON",
    json.dumps(
        [public_report(r) for r in reports],
        ensure_ascii=False,
        indent=2,
    ).encode(),
    "clip_report.json",
    "application/json",
)

index = st.selectbox(
    "เปิดรายละเอียด",
    range(len(reports)),
    format_func=lambda i: (
        f"{reports[i]['filename']} — "
        f"{reports[i]['status']}"
    ),
)

r = reports[index]

st.write("**เกณฑ์:**", r.get("profile_label", "–"))

st.write(
    "**คะแนน:**",
    "ยังตรวจไม่ครบ"
    if r.get("score_is_partial")
    else r.get("score"),
)

for reason in r.get("decision_reasons", []):
    st.write(reason)

if r.get("analysis_error"):
    st.warning(r["analysis_error"])

if (
    videos
    and r["upload_index"] < len(videos)
    and videos[r["upload_index"]].name == r["filename"]
):
    st.video(
        videos[r["upload_index"]].getvalue()
    )

if r.get("transcript"):
    st.write(
        "**ข้อความถอดเสียง:**",
        r["transcript"],
    )

for key, title, records in [
    ("speech_check", "คำพูดภาษาไทย", "errors"),
    ("scale_check", "สเกลกับคำพูด", "claims"),
]:
    gate = r.get(key)

    if gate:
        st.subheader(title)
        st.write(gate["status"], gate["reason"])

        if key == "speech_check":
            st.write(
                "คำผิดยืนยัน:",
                gate.get("confirmed_count", 0),
                "คำ · ที่สงสัย:",
                gate.get("possible_count", 0),
                "คำ",
            )

            st.write(
                "ข้อความจากการฟัง:",
                gate.get("heard_transcript", ""),
            )

        if gate.get(records):
            st.dataframe(
                pd.DataFrame(gate[records]),
                hide_index=True,
                width="stretch",
            )

for i, issue in enumerate(r.get("issues", [])):
    label = CATEGORIES.get(
        issue["category"], issue["category"]
    )

    with st.expander(
        f"{label}: {issue['description']}"
    ):
        st.write(issue)

        evidence = r.get("evidence", {}).get(str(i))

        if evidence:
            st.image(
                evidence[1],
                caption=f"เฟรมจริง {evidence[0]:.2f}s",
                width=360,
            )

st.caption(
    "เวลาในตารางคำพูดเป็นช่วงประมาณจากโมเดล "
    "ไม่ใช่เวลา forced alignment"
)

manual = st.selectbox(
    "ผลที่คุณตรวจเอง",
    ["ยังไม่ประเมิน", "PASS", "FAIL", "REVIEW"],
    key=f"human_{index}",
)

reason = st.text_input(
    "เหตุผลที่คนตรวจเห็น",
    key=f"reason_{index}",
)

if st.button("บันทึกผลคนตรวจ"):
    st.session_state.feedback[index] = {
        "filename": r["filename"],
        "ai_result": r["status"],
        "human_result": manual,
        "reason": reason,
    }

if st.session_state.feedback:
    feedback = pd.DataFrame(
        st.session_state.feedback.values()
    )

    st.download_button(
        "ดาวน์โหลดผลคนตรวจ",
        feedback.to_csv(index=False).encode("utf-8-sig"),
        "human_feedback.csv",
        "text/csv",
    )
