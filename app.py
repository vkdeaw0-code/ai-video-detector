
from __future__ import annotations
import csv
import hashlib
import io
import tempfile
import time
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st

from qc_engine import (
    MAX_TOTAL_BYTES, SUPPORTED, inspect_video, save_upload
)

st.set_page_config(
    page_title="AI Video QC V4",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Video QC V4.0 | Cloud Lite")
st.caption(
    "ตรวจภาพ + เสียง + ความสอดคล้องของสินค้า "
    "• ประมวลผลทีละคลิป • ไม่แปลง Error เป็น PASS"
)

st.warning(
    "ข้อจำกัดจริง: Streamlit Cloud ฟรีเก็บไฟล์จาก "
    "st.file_uploader ใน RAM แม้ตั้งเพดานรวม 1 GB "
    "ก็ไม่ได้รับประกันว่าจะอัปโหลด 1 GB สำเร็จ "
    "แนะนำทยอยอัปโหลดชุดเล็กเมื่อ RAM จำกัด"
)

if "results" not in st.session_state:
    st.session_state.results = []

if "done" not in st.session_state:
    st.session_state.done = set()

with st.sidebar:
    st.header("ตั้งค่าการตรวจ")

    mode = st.radio(
        "โหมดวิเคราะห์",
        [
            "Local ฟรี (เบื้องต้น)",
            "Gemini (ตรวจภาพและฟังเสียง)"
        ],
        help=(
            "โหมดฟรีไม่สามารถยืนยันความผิดรูปและ"
            "คำพูดทุกกรณี จึงให้ REVIEW เป็นหลัก"
        )
    )

    fps = st.slider(
        "จำนวนภาพสำหรับการคัดกรอง (เฟรม/วินาที)",
        1, 4, 2
    )

    asr = st.checkbox(
        "ถอดเสียงภาษาไทยด้วย Whisper ในเครื่อง",
        value=mode.startswith("Local"),
        help=(
            "อาจช้ากว่า 10 วินาที/คลิป "
            "เมื่อรัน CPU ของ Cloud ฟรี"
        )
    )

    whisper = st.selectbox(
        "ขนาดโมเดลถอดเสียง",
        ["tiny", "base"],
        index=0,
        disabled=not asr
    )

    try:
        default_key = st.secrets.get(
            "GEMINI_API_KEY", ""
        )
    except Exception:
        default_key = ""

    api_key = (
        st.text_input(
            "Gemini API key (ไม่บังคับ)",
            value=default_key,
            type="password"
        )
        if mode.startswith("Gemini")
        else ""
    )

    model = st.selectbox(
        "Gemini model",
        [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash"
        ],
        index=0,
        disabled=not mode.startswith("Gemini")
    )

    if mode.startswith("Gemini") and not api_key:
        st.error(
            "โหมด Gemini ต้องมี API key; "
            "หากไม่มีให้เลือก Local ฟรี"
        )

    st.caption(
        "บริการ AI ภายนอกอาจมีโควตาหรือค่าใช้จ่าย "
        "ดูรายละเอียดใน Google AI Studio"
    )

    st.divider()

    st.metric(
        "RAM ว่างขณะนี้",
        f"{psutil.virtual_memory().available / 1024**2:.0f} MB"
    )

    st.caption(
        "ผลในหน้านี้เป็นข้อมูลชั่วคราว "
        "หาก Cloud รีสตาร์ตอาจสูญหาย "
        "กรุณาดาวน์โหลด CSV"
    )

st.subheader(
    "1) ข้อมูลสินค้าที่ถูกต้อง (ไม่บังคับ แต่แนะนำ)"
)

st.caption(
    "หากแต่ละคลิปเป็นสินค้าคนละตัว ให้อัปโหลด CSV "
    "เพื่อช่วยตรวจชื่อ/ยี่ห้อ: "
    "filename,product,brand,required_terms "
    "โดยแยก required_terms ด้วย ;"
)

catalog_file = st.file_uploader(
    "ตารางข้อมูลสินค้า .csv",
    type="csv",
    key="catalog"
)

catalog = {}

if catalog_file:
    try:
        text = catalog_file.getvalue().decode(
            "utf-8-sig"
        )

        reader = csv.DictReader(io.StringIO(text))

        if not {
            "filename", "product", "brand"
        }.issubset(set(reader.fieldnames or [])):
            st.error(
                "CSV ต้องมีคอลัมน์ "
                "filename,product,brand"
            )
        else:
            for row in reader:
                name = Path(
                    str(row.get("filename", ""))
                ).name

                if name:
                    catalog[name] = row

            st.success(
                f"อ่านข้อมูลสินค้าแล้ว {len(catalog)} รายการ"
            )

    except Exception as exc:
        st.error(f"อ่าน CSV ไม่สำเร็จ: {exc}")

st.subheader(
    "2) เพิ่มคลิป (รวมไม่เกิน 1 GB ต่อชุดที่เลือก)"
)

uploads = st.file_uploader(
    "เลือกวิดีโอหลายไฟล์",
    type=[
        x[1:] for x in sorted(SUPPORTED)
    ],
    accept_multiple_files=True,
    key="videos"
)

total = (
    sum(f.size for f in uploads)
    if uploads else 0
)

st.caption(
    f"เลือก {len(uploads or [])} ไฟล์ | "
    f"รวม {total / 1_000_000:.1f} MB / 1,000 MB"
)

if total > MAX_TOTAL_BYTES:
    st.error(
        "ขนาดรวมเกิน 1 GB: กรุณาลดจำนวนไฟล์"
    )

elif total > 250_000_000:
    st.warning(
        "ไฟล์ชุดนี้ใหญ่กว่า 250 MB: "
        "Cloud ฟรีมีความเสี่ยง RAM เต็มก่อนเริ่มตรวจ "
        "แม้ตัวโปรแกรมจะจัดคิวทีละคลิป"
    )

settings = {
    "fps": fps,
    "asr": asr,
    "whisper": whisper,
    "api_key": (
        api_key if mode.startswith("Gemini") else ""
    ),
    "gemini_model": model
}

valid = (
    bool(uploads)
    and total <= MAX_TOTAL_BYTES
    and (
        not mode.startswith("Gemini")
        or bool(api_key)
    )
)

if st.button(
    "▶ เริ่มตรวจชุดนี้",
    type="primary",
    disabled=not valid
):
    progress = st.progress(0)
    output = st.empty()

    total_files = len(uploads)

    with st.status(
        "กำลังประมวลผลทีละคลิป...",
        expanded=True
    ) as status_box:

        for i, f in enumerate(uploads, 1):

            content_hash = hashlib.sha256(
                f.getbuffer()
            ).hexdigest()

            identity = (
                content_hash,
                mode,
                fps,
                asr,
                whisper,
                model,
                bool(api_key),
                catalog.get(f.name, {}).get("product", ""),
                catalog.get(f.name, {}).get("brand", ""),
                catalog.get(f.name, {}).get(
                    "required_terms", ""
                )
            )

            digest = hashlib.sha256(
                repr(identity).encode()
            ).hexdigest()

            if digest in st.session_state.done:
                st.write(
                    f"ข้าม {f.name} "
                    "(ตรวจแล้วในเซสชันนี้)"
                )
                progress.progress(i / total_files)
                continue

            t0 = time.monotonic()

            st.write(
                f"กำลังตรวจ {i}/{total_files}: {f.name}"
            )

            try:
                if (
                    Path(f.name).suffix.lower()
                    not in SUPPORTED
                ):
                    raise ValueError(
                        "นามสกุลไฟล์ไม่รองรับ"
                    )

                available = (
                    psutil.virtual_memory().available
                )

                if available < 180 * 1024**2:
                    raise MemoryError(
                        "RAM เหลือน้อยกว่า 180 MB "
                        "หยุดไฟล์นี้เพื่อป้องกันเว็บล่ม"
                    )

                with tempfile.TemporaryDirectory(
                    prefix="qc_clip_"
                ) as tmp:

                    file_path = (
                        Path(tmp)
                        / (
                            "clip"
                            + Path(f.name).suffix.lower()
                        )
                    )

                    save_upload(
                        f,
                        file_path,
                        max_bytes=MAX_TOTAL_BYTES
                    )

                    result = inspect_video(
                        file_path,
                        f.name,
                        settings,
                        catalog.get(f.name)
                    )

            except Exception as exc:
                result = {
                    "filename": f.name,
                    "status": "ERROR",
                    "reason": (
                        f"{type(exc).__name__}: "
                        f"{str(exc)[:220]}"
                    ),
                    "process_s": round(
                        time.monotonic() - t0, 2
                    ),
                    "duration_s": None,
                    "resolution": "",
                    "transcript": "",
                    "segments": [],
                    "visual_findings": [],
                    "audio_findings": [],
                    "issues": [],
                    "mode": mode,
                    "product": "",
                    "brand": ""
                }

            st.session_state.results.append(
                result
            )

            if result["status"] != "ERROR":
                st.session_state.done.add(digest)

            progress.progress(i / total_files)

            output.info(
                f"{f.name} → {result['status']} "
                f"| {result['process_s']:.1f} วินาที"
            )

        status_box.update(
            label="ตรวจครบแล้ว (โปรดดูผลด้านล่าง)",
            state="complete"
        )

st.subheader("3) รายงานผล")

results = st.session_state.results

if results:

    df = pd.DataFrame([
        {
            "filename": r["filename"],
            "status": r["status"],
            "duration_s": r["duration_s"],
            "process_s": r["process_s"],
            "reason": r["reason"],
            "brand": r.get("brand", ""),
            "product": r.get("product", ""),
            "transcript": r.get(
                "transcript", ""
            ),
            "visual_findings": " | ".join(
                r.get("visual_findings", [])
            ),
            "audio_findings": " | ".join(
                r.get("audio_findings", [])
            ),
            "issues": str(
                r.get("issues", [])
            )
        }
        for r in results
    ])

    cols = st.columns(4)

    for col, label in zip(
        cols,
        ["PASS", "FAIL", "REVIEW", "ERROR"]
    ):
        col.metric(
            label,
            int((df["status"] == label).sum())
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇ ดาวน์โหลดรายงาน CSV",
        data=df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="video_qc_results.csv",
        mime="text/csv"
    )

    with st.expander(
        "ดูคำพูดและหลักฐานโดยละเอียด"
    ):
        for r in reversed(results):

            st.markdown(
                f"**{r['filename']} "
                f"— {r['status']}**"
            )

            st.write(
                "เหตุผล:",
                r["reason"]
            )

            st.write(
                "คำพูดที่ตรวจจับ:",
                r.get("transcript")
                or "(ยังไม่มีผลถอดเสียงที่ยืนยันได้)"
            )

            st.write(
                "ภาพ:",
                r.get("visual_findings")
                or "ยังไม่พบสัญญาณผิดปกติจากตัวตรวจพื้นฐาน"
            )

            st.write(
                "เสียง:",
                r.get("audio_findings")
                or "ยังไม่พบข้อสังเกตเพิ่มเติม"
            )

            st.write(
                "หลักฐาน:",
                r.get("issues")
                or "ยังไม่มีข้อมูลเพิ่มเติม"
            )

    if st.button(
        "ล้างผลที่บันทึกในเซสชัน"
    ):
        st.session_state.results = []
        st.session_state.done = set()
        st.rerun()

else:
    st.info(
        "ยังไม่มีผลตรวจ เริ่มอัปโหลดแล้วกด "
        "“เริ่มตรวจชุดนี้”"
    )

st.subheader(
    "4) เทียบกับผลตรวจของคุณ"
)

st.caption(
    "ไฟล์ CSV มีสองคอลัมน์ filename,truth "
    "โดย truth เป็น PASS หรือ FAIL "
    "เพื่อวัดผลจริงจากชุดที่มีเฉลย"
)

labels_file = st.file_uploader(
    "อัปโหลด labels.csv (ไม่บังคับ)",
    type="csv",
    key="labels"
)

if labels_file and results:
    try:
        labels = pd.read_csv(labels_file)

        if not {
            "filename", "truth"
        }.issubset(labels.columns):
            raise ValueError(
                "ต้องมีคอลัมน์ filename และ truth"
            )

        actual = pd.DataFrame([
            {
                "filename": r["filename"],
                "pred": r["status"]
            }
            for r in results
        ]).drop_duplicates(
            "filename",
            keep="last"
        )

        matched = actual.merge(
            labels[["filename", "truth"]],
            on="filename",
            how="inner"
        )

        matched = matched[
            matched["truth"].isin(
                ["PASS", "FAIL"]
            )
        ]

        if matched.empty:
            st.info(
                "ยังไม่มีชื่อไฟล์ในรายงาน"
                "ที่ตรงกับตารางเฉลย"
            )
        else:
            n = len(matched)

            reviewed = int(
                (matched["pred"] == "REVIEW").sum()
            )

            errors = int(
                (matched["pred"] == "ERROR").sum()
            )

            auto = matched[
                matched["pred"].isin(
                    ["PASS", "FAIL"]
                )
            ]

            acc = (
                float(
                    (
                        auto["pred"]
                        == auto["truth"]
                    ).mean()
                )
                if len(auto)
                else None
            )

            bad = matched[
                matched["truth"] == "FAIL"
            ]

            escaped = int(
                (bad["pred"] == "PASS").sum()
            )

            st.write(
                f"มีเฉลยตรงกัน {n} คลิป "
                f"| REVIEW {reviewed} "
                f"| ERROR {errors} "
                f"| คลิปผิดที่หลุด PASS {escaped}"
            )

            st.write(
                "ความแม่นยำเฉพาะคลิปที่ระบบ"
                "ตัดสิน PASS/FAIL:",
                f"{acc:.1%}"
                if acc is not None
                else (
                    "ยังคำนวณไม่ได้ "
                    "เพราะยังไม่ตัดสินอัตโนมัติ"
                )
            )

            st.dataframe(
                matched,
                hide_index=True,
                use_container_width=True
            )

    except Exception as exc:
        st.error(
            f"อ่านไฟล์เฉลยไม่สำเร็จ: {exc}"
        )
