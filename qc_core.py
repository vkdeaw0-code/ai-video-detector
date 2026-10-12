
"""AI Video QC - Product Scale Guard V2.
Compatible with existing app.py and qc_store.py.
"""

from __future__ import annotations

import base64
import json
import math
import re
import subprocess
import tempfile
import time
import wave

from array import array
from pathlib import Path

import requests
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageOps,
    ImageStat,
)


# เปลี่ยนเวอร์ชันเพื่อไม่ใช้ Cache ผลตรวจเก่า
ENGINE_VERSION = "2026-10-11-product-scale-v2"

FPS = {
    "Fast": 1,
    "Balanced": 2,
    "Deep": 3,
}

ALLOWED_MODELS = {
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
}


PROMPT = """
คุณเป็นหัวหน้าฝ่ายตรวจคุณภาพวิดีโอโฆษณาสินค้า AI
(Thai AI Video Quality Inspector)

เป้าหมาย:
คัดเลือกวิดีโอสินค้า AI ที่มีคุณภาพดี
ภาพดูเนียน สินค้าไม่ผิดสัดส่วน
และขนาดสินค้าต้องตรงกับชนิดและคำพูดที่นำเสนอ

สำคัญมาก:
ห้ามให้ PASS เพียงเพราะภาพดูสวยหรือไม่บิดเบี้ยว

คลิปความยาวประมาณ 10 วินาที
ได้รับภาพตัวอย่างตามลำดับเวลา
ภาพเดี่ยวสำหรับดูสัดส่วน
และอาจได้รับเสียงต้นฉบับ

=============================
ขั้นที่ 1 จำแนกประเภทวิดีโอ
=============================

HUMAN:
มีคนยืนขายสินค้า เห็นหน้าหรือร่างกายคนขาย

HANDS:
ไม่มีคนขายเต็มตัว แต่มีมือหรือแขนพรีเซนสินค้า

PRODUCT:
แสดงสินค้าเป็นหลัก ไม่มีคนหรือมือพรีเซน

MIXED:
สลับระหว่างหลายรูปแบบ

ต้องระบุว่าสินค้าหลักคืออะไร
ห้ามนำสินค้าประกอบฉากมาเป็นสินค้าหลัก

==================================
ขั้นที่ 2 ตรวจขนาดสินค้า SCALE AUDIT
==================================

ให้ความสำคัญเป็นพิเศษกับ HANDS, PRODUCT และ MIXED

ตรวจสอบ:

1. สินค้าคืออะไร
2. สินค้านี้มีวัตถุประสงค์ใช้ทำอะไร
3. ขนาดที่แสดงสมเหตุสมผลกับการใช้งานหรือไม่
4. ขนาดเทียบกับมือหรือนิ้วมืออย่างไร
5. ขนาดเทียบกับวัตถุรอบข้างอย่างไร
6. มีการอ้างขนาดหรือประโยชน์ใช้สอยในเสียงหรือไม่
7. ขนาดสินค้าเล็กเกินกว่าจะใช้งานตามที่นำเสนอหรือไม่
8. สินค้าดูเหมือนแบบจำลองย่อส่วนโดยไม่ได้ตั้งใจหรือไม่
9. สินค้าบิดเบี้ยวหรือเปลี่ยนขนาดระหว่างคลิปหรือไม่
10. สินค้าตรงกับประเภทที่ผู้ขายกล่าวถึงหรือไม่

ตัวอย่างแนวคิด:

- ชั้นวางของที่อ้างว่าใช้เก็บของขนาดปกติ
  แต่ในภาพดูเหมือนชั้นจำลองขนาดเล็ก
  เมื่อเทียบกับมือและสิ่งของที่ต้องวาง
  ถือว่าเป็นความผิดพลาดสำคัญ

- ถังขยะที่โฆษณาเพื่อใช้งานทั่วไป
  แต่มีขนาดเล็กผิดบริบทการใช้งาน
  ให้พิจารณา FAIL เมื่อมีหลักฐานชัดเจน

- ชั้นเก็บอุปกรณ์ครัวที่อ้างว่ารองรับเครื่องใช้
  แต่สัดส่วนของชั้นไม่สัมพันธ์กับขนาดเครื่องใช้
  ถือเป็นข้อผิดพลาดเชิงการใช้งาน

- ห่วงบาสมินิที่ระบุชัดว่าเป็นของเล่นตั้งโต๊ะ
  ถือเป็นสินค้าขนาดเล็กโดยตั้งใจ
  ห้ามตัดสิน FAIL เพราะเล็กเพียงอย่างเดียว

คำเตือน:

ห้ามคิดว่าสินค้าผิดขนาดเพียงเพราะ
สินค้าอยู่ไกลกล้อง หรือมีการซูมภาพ

ห้ามถือว่าสินค้าทุกชนิดต้องใหญ่

ถ้ามีคำอธิบายว่าเป็นสินค้า MINI
ของเล่น ตั้งโต๊ะ หรือรุ่นขนาดเล็ก
ให้ตรวจตามประเภทสินค้านั้นจริง

หากไม่มีหลักฐานเพียงพอ
ห้ามแต่งขนาดจริงขึ้นมา

===================================
ขั้นที่ 3 ตรวจสอบคำพูดและสินค้า
===================================

ฟังเสียงภาษาไทยจริงจากคลิป

ให้ตรวจ:

- ผู้พูดขายสินค้าประเภทใด
- กล่าวถึงขนาดหรือความจุอย่างไร
- กล่าวถึงประโยชน์ใช้สอยอย่างไร
- กล่าวถึงตำแหน่งใช้งานหรือไม่
- สินค้าที่เห็นสอดคล้องกับคำพูดหรือไม่
- เสียงพูดขาด สะดุด หรือผิดเพี้ยนหรือไม่

หากผู้พูดอ้างว่าสินค้าใช้งานขนาดปกติ
แต่ภาพแสดงขนาดที่เล็กจนไม่รองรับการใช้งานนั้น
และมีหลักฐานชัดเจน ให้ FAIL

ห้ามเดาคำพูดเมื่อฟังไม่ชัด
ห้ามอ้างว่ามีคำพูดหากคลิปไม่มีเสียง

==================================
ขั้นที่ 4 กฎตรวจคลิป HUMAN
==================================

หากเป็นคนยืนขายสินค้า:

เน้นเฉพาะ:

- สินค้าหลักตรงกลางหรือสินค้าที่คนขายถือ
- สินค้าบิดเบี้ยวหรือเปลี่ยนรูปทรงหรือไม่
- มือและนิ้วที่จับสินค้าผิดรูปหรือไม่
- การเคลื่อนไหวของสินค้าเป็นธรรมชาติหรือไม่
- สินค้าทะลุมือหรือร่างกายหรือไม่
- เสียงพูดแตก สะดุด หรือผิดเพี้ยนหรือไม่
- ปากและเสียงสัมพันธ์กันหรือไม่เมื่อเห็นชัด

ไม่ต้องจับผิดรายละเอียดเล็กน้อยของฉากหลัง

หากภาพสินค้าและเสียงดูดี
ไม่มีข้อบกพร่องสำคัญ ให้ PASS

==================================
ขั้นที่ 5 ตรวจคุณภาพทั่วไป
==================================

ตรวจ:

- Product Geometry
- Product Scale
- Product Identity
- Hand Consistency
- Object Interaction
- Motion Continuity
- Audio Distortion
- Thai Speech Quality
- Advertising Usability

ข้อผิดพลาดเล็กน้อยที่ไม่รบกวนการดู
ไม่จำเป็นต้อง FAIL

==================================
ขั้นที่ 6 การตัดสิน
==================================

ให้ PASS เมื่อภาพและสินค้าเหมาะสมกับโฆษณา
และไม่มีข้อผิดพลาดสำคัญ

ให้ FAIL เมื่อพบความผิดพลาดสำคัญ เช่น:

1. สินค้าเล็กผิดประเภทหรือผิดการใช้งานอย่างชัดเจน
2. คำพูดกล่าวอ้างการใช้งานที่ขนาดสินค้าไม่รองรับ
3. สินค้าเปลี่ยนขนาดหรือรูปร่างผิดธรรมชาติ
4. มือหรือคนผิดรูปอย่างรุนแรง
5. เสียงแตกหรือผิดเพี้ยนจนกระทบการรับชม

สำคัญ:
สินค้าที่ดูเนียนแต่เล็กผิดสเกลจริง
สามารถ FAIL ได้แม้จะไม่มีภาพบิดเบี้ยว

การตัดสินสเกลต้องมีหลักฐานอย่างน้อยสองด้าน
เช่น ขนาดเทียบมือและวัตถุประสงค์
หรือคำพูดกับภาพสินค้า

อย่าตัดสินจากความรู้สึกอย่างเดียว

==================================
JSON OUTPUT
==================================

ตอบ JSON object เท่านั้น

{
  "decision": "PASS",
  "video_type": "HANDS",
  "product_name_th": "ชื่อสินค้า",
  "reason_th": "เหตุผลภาษาไทย",
  "issues": [
    {
      "time_sec": 4.0,
      "category": "product",
      "severity": "critical",
      "detail_th": "รายละเอียด"
    }
  ],
  "speech_summary_th": "คำพูดที่ได้ยินจริง",
  "scale_review": {
    "size_grade": "normal",
    "category_scale_mismatch": false,
    "spoken_claim_conflict": false,
    "miniature_disclosed": false,
    "visual_evidence_th": "หลักฐานจากภาพ",
    "usage_evidence_th": "หลักฐานด้านการใช้งาน",
    "speech_evidence_th": "หลักฐานจากคำพูด",
    "confidence": 0.0,
    "reason_th": "สรุปสเกลสินค้า"
  }
}

size_grade เลือกได้เพียง:
normal, intentional_mini, suspicious,
wrong_scale, unknown

wrong_scale ใช้เมื่อมีหลักฐานชัดเจนเท่านั้น

confidence เป็นความมั่นใจว่าผิดสเกล
โดยอยู่ระหว่าง 0 และ 1

สำหรับ HUMAN:
ถ้าไม่ได้ตรวจสเกลให้ระบุ unknown
"""


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


def command(args, timeout=90):
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AnalysisError(
            f"FFmpeg ผิดพลาด: {exc}"
        ) from exc

    if p.returncode:
        raise AnalysisError(
            (p.stderr or "FFmpeg error")[-450:]
        )

    return p


def probe(path):
    try:
        data = json.loads(
            command([
                "ffprobe",
                "-v", "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of", "json",
                str(path),
            ]).stdout
        )

        duration = float(
            data.get("format", {}).get(
                "duration", 0
            ) or 0
        )

    except (ValueError, TypeError, KeyError) as exc:
        raise AnalysisError(
            "อ่านข้อมูลวิดีโอไม่ได้"
        ) from exc

    streams = data.get(
        "streams", []
    )

    has_video = any(
        x.get("codec_type") == "video"
        for x in streams
    )

    if not has_video:
        raise AnalysisError(
            "ไม่พบแทร็กวิดีโอ"
        )

    if not 0.5 <= duration <= 25:
        raise AnalysisError(
            "รองรับคลิปประมาณ 10 วินาที "
            f"(คลิปนี้ {duration:.1f} วินาที)"
        )

    return {
        "duration": duration,
        "has_audio": any(
            x.get("codec_type") == "audio"
            for x in streams
        ),
    }


def frames_from_video(path, folder, fps):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    vf = (
        f"fps={fps},"
        "scale=640:640:"
        "force_original_aspect_ratio=decrease:"
        "force_divisible_by=2"
    )

    command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(path),
        "-vf", vf,
        "-q:v", "5",
        str(folder / "f_%03d.jpg"),
    ], timeout=120)

    result = sorted(
        folder.glob("f_*.jpg")
    )

    if not result:
        raise AnalysisError(
            "ดึงเฟรมไม่ได้"
        )

    return result[:80]


def motion_peaks(frames, fps, duration):
    changes = []
    previous = None

    for i, file in enumerate(frames):
        with Image.open(file) as im:
            small = ImageOps.grayscale(
                im
            ).resize((48, 48))

        if previous is not None:
            difference = ImageChops.difference(
                small, previous
            )

            amount = ImageStat.Stat(
                difference
            ).mean[0]

            if amount > 45:
                changes.append((
                    amount,
                    min(
                        duration - 0.15,
                        i / fps,
                    ),
                ))

        previous = small

    changes.sort(reverse=True)

    selected = []

    for _, t in changes:
        if (
            t >= 0.15
            and all(
                abs(t - other) > 0.8
                for other in selected
            )
        ):
            selected.append(t)

        if len(selected) >= 2:
            break

    return sorted(selected)


def extra_frame(path, at, out):
    try:
        command([
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-ss", f"{at:.2f}",
            "-i", str(path),
            "-frames:v", "1",
            "-vf",
            "scale=640:640:"
            "force_original_aspect_ratio=decrease:"
            "force_divisible_by=2",
            str(out),
        ], timeout=25)

        return out.is_file()

    except AnalysisError:
        return False


def contact_sheets(frames, fps, folder):
    result = []

    for start in range(
        0, len(frames), 10
    ):
        sheet = Image.new(
            "RGB",
            (5 * 244, 2 * 280),
            (22, 27, 38),
        )

        draw = ImageDraw.Draw(sheet)

        for i, path in enumerate(
            frames[start:start + 10]
        ):
            index = start + i

            with Image.open(path) as im:
                small = ImageOps.contain(
                    im.convert("RGB"),
                    (236, 245),
                )

            x = (
                (i % 5) * 244
                + (244 - small.width) // 2
            )

            y = (
                (i // 5) * 280
                + (245 - small.height) // 2
                + 24
            )

            sheet.paste(
                small, (x, y)
            )

            draw.text(
                (
                    (i % 5) * 244 + 8,
                    (i // 5) * 280 + 5,
                ),
                f"t~{index / fps:.1f}s",
                fill="white",
            )

        output = folder / (
            f"sheet_{len(result):02d}.jpg"
        )

        sheet.save(
            output,
            "JPEG",
            quality=82,
            optimize=True,
        )

        result.append(output)

    return result


def inspect_frames(frames, fps):
    # ตรวจภาพเดี่ยวเพิ่ม ไม่ดูเพียงภาพรวมขนาดเล็ก
    positions = (
        0.10,
        0.32,
        0.57,
        0.80,
        0.94,
    )

    indexes = sorted(set(
        round(
            x * (len(frames) - 1)
        )
        for x in positions
    ))

    return [
        (i / fps, frames[i])
        for i in indexes
    ]


def extract_audio(path, folder, has_audio):
    if not has_audio:
        return None, {
            "has_audio": False
        }

    audio = folder / "audio.wav"

    command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio),
    ], timeout=60)

    with wave.open(
        str(audio), "rb"
    ) as wav:
        if wav.getsampwidth() != 2:
            raise AnalysisError(
                "รูปแบบเสียงไม่ถูกต้อง"
            )

        samples = array("h")
        samples.frombytes(
            wav.readframes(
                wav.getnframes()
            )
        )

    if not samples:
        return None, {
            "has_audio": True,
            "silent": True,
        }

    rms = math.sqrt(
        sum(v * v for v in samples)
        / len(samples)
    ) / 32768.0

    clipped_ratio = sum(
        abs(v) >= 32300
        for v in samples
    ) / len(samples)

    if rms < 0.001:
        return None, {
            "has_audio": True,
            "silent": True,
            "rms": round(rms, 5),
        }

    return audio, {
        "has_audio": True,
        "rms": round(rms, 4),
        "clipped_ratio": round(
            clipped_ratio, 4
        ),
        "possibly_silent": rms < 0.002,
    }


def inline_part(path, mime):
    data = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return {
        "inline_data": {
            "mime_type": mime,
            "data": data,
        }
    }


def _as_bool(value):
    return (
        value is True
        or (
            isinstance(value, str)
            and value.lower() == "true"
        )
    )


def apply_scale_rule(report):
    """
    ตรวจคำตัดสิน AI อีกชั้น

    เมื่อหลักฐานผิดสเกลชัดเจน
    สามารถเปลี่ยน PASS เป็น FAIL ได้
    """
    result = dict(report)

    category = str(
        result.get(
            "video_type", ""
        )
    ).upper()

    size = result.get(
        "scale_review"
    )

    if not isinstance(size, dict):
        size = {}

    result["scale_review"] = size
    result["scale_override"] = False

    # คลิปคนขายใช้เกณฑ์ตรวจภาพและเสียงเดิม
    if category == "HUMAN":
        return result

    # ไม่ตัดตกสินค้า MINI ที่ระบุชัดว่าขายแบบ MINI
    if _as_bool(
        size.get("miniature_disclosed")
    ):
        return result

    if category not in {
        "HANDS",
        "PRODUCT",
        "MIXED",
    }:
        return result

    grade = str(
        size.get(
            "size_grade", ""
        )
    ).lower()

    if grade != "wrong_scale":
        return result

    try:
        confidence = float(
            size.get(
                "confidence", 0
            )
        )

    except (TypeError, ValueError):
        confidence = 0

    if not math.isfinite(confidence):
        confidence = 0

    visual = str(
        size.get(
            "visual_evidence_th"
        ) or ""
    ).strip()

    usage = str(
        size.get(
            "usage_evidence_th"
        ) or ""
    ).strip()

    speech = str(
        size.get(
            "speech_evidence_th"
        ) or ""
    ).strip()

    category_mismatch = _as_bool(
        size.get(
            "category_scale_mismatch"
        )
    )

    spoken_mismatch = _as_bool(
        size.get(
            "spoken_claim_conflict"
        )
    )

    # ต้องมีหลักฐานอย่างน้อยสองส่วน
    confirmed = (
        confidence >= 0.82
        and bool(visual)
        and (
            (
                category_mismatch
                and bool(usage)
            )
            or (
                spoken_mismatch
                and bool(speech)
            )
        )
    )

    if not confirmed:
        return result

    result["scale_override"] = (
        result.get("decision") == "PASS"
    )

    result["decision"] = "FAIL"

    reason = str(
        size.get(
            "reason_th"
        ) or ""
    ).strip()

    result["reason_th"] = (
        "สเกลสินค้าไม่สอดคล้องกับ"
        "ชนิด/การใช้งานหรือคำพูด: "
        + (
            reason[:340]
            if reason
            else visual[:250]
        )
    )

    issues = result.get(
        "issues"
    )

    if not isinstance(issues, list):
        issues = []

    issues = list(issues)

    issues.append({
        "time_sec": 0.0,
        "category": "product",
        "severity": "critical",
        "detail_th": result[
            "reason_th"
        ],
    })

    result["issues"] = issues[:25]

    return result


def gemini_analyze(
    key,
    model,
    sheets,
    details,
    wav,
    duration,
    audio_stats,
):
    if model not in ALLOWED_MODELS:
        raise AnalysisError(
            "ชื่อโมเดลไม่ได้อยู่ในรายการ"
        )

    parts = [{
        "text": (
            PROMPT
            + "\nระยะเวลาคลิป="
            + f"{duration:.2f}"
            + " วินาที ข้อมูลเสียง="
            + json.dumps(
                audio_stats,
                ensure_ascii=False,
            )
        )
    }]

    for p in sheets:
        parts.append({
            "text": (
                "TIMELINE: ภาพหลายเวลา "
                "ตรวจความต่อเนื่องและขนาดสินค้า"
            )
        })

        parts.append(
            inline_part(
                p,
                "image/jpeg",
            )
        )

    for sec, p in details:
        parts.append({
            "text": (
                "เฟรมเดี่ยวตรวจสเกล "
                f"เวลาใกล้ {sec:.1f} วินาที "
                "เทียบสินค้า มือ และการใช้งาน"
            )
        })

        parts.append(
            inline_part(
                p,
                "image/jpeg",
            )
        )

    if wav is not None:
        parts.append({
            "text": (
                "เสียงจริงจากคลิปเดียวกัน "
                "ตรวจคำพูดภาษาไทย ขนาดสินค้า "
                "และประโยชน์ใช้สอยที่กล่าวอ้าง"
            )
        })

        parts.append(
            inline_part(
                wav,
                "audio/wav",
            )
        )

    endpoint = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/{model}:generateContent"
    )

    payload = {
        "contents": [{
            "role": "user",
            "parts": parts,
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2200,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(3):
        try:
            response = requests.post(
                endpoint,
                headers={
                    "x-goog-api-key": key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(15, 95),
            )

        except requests.RequestException as exc:
            if attempt == 2:
                raise AnalysisError(
                    f"เชื่อมต่อ Gemini ไม่ได้: {exc}"
                ) from exc

            time.sleep(
                2 * (attempt + 1)
            )
            continue

        if response.status_code == 429:
            raise QuotaError(
                "Gemini API ถึงโควตา "
                "กรุณารอแล้วเริ่มต่อ"
            )

        if (
            response.status_code
            in (500, 502, 503, 504)
            and attempt < 2
        ):
            time.sleep(
                2 * (attempt + 1)
            )
            continue

        if response.status_code >= 400:
            try:
                msg = response.json().get(
                    "error", {}
                ).get(
                    "message",
                    response.text[:300],
                )

            except ValueError:
                msg = response.text[:300]

            raise AnalysisError(
                f"Gemini HTTP "
                f"{response.status_code}: "
                f"{str(msg)[:350]}"
            )

        try:
            obj = response.json()

            text = "".join(
                p.get("text", "")
                for p in (
                    obj["candidates"][0]
                    ["content"]["parts"]
                )
            )

            raw = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text.strip(),
            )

            report = json.loads(raw)

            if not isinstance(report, dict):
                raise ValueError(
                    "JSON ไม่ใช่ object"
                )

            decision = str(
                report.get(
                    "decision", ""
                )
            ).upper()

            kind = str(
                report.get(
                    "video_type", ""
                )
            ).upper()

            if decision not in {
                "PASS", "FAIL"
            }:
                raise ValueError(
                    "decision ไม่ถูกต้อง"
                )

            if kind not in {
                "HUMAN",
                "HANDS",
                "PRODUCT",
                "MIXED",
            }:
                raise ValueError(
                    "video_type ไม่ถูกต้อง"
                )

            if not isinstance(
                report.get(
                    "issues", []
                ),
                list,
            ):
                raise ValueError(
                    "issues ต้องเป็น list"
                )

            if kind in {
                "HANDS",
                "PRODUCT",
                "MIXED",
            }:
                review = report.get(
                    "scale_review"
                )

                if (
                    not isinstance(
                        review, dict
                    )
                    or review.get(
                        "size_grade"
                    ) not in {
                        "normal",
                        "intentional_mini",
                        "suspicious",
                        "wrong_scale",
                        "unknown",
                    }
                ):
                    raise ValueError(
                        "ไม่ได้รับ scale_review "
                        "จาก Gemini"
                    )

            clean = {
                "decision": decision,
                "video_type": kind,
                "reason_th": str(
                    report.get(
                        "reason_th"
                    ) or "ไม่มีเหตุผล"
                )[:800],
                "product_name_th": str(
                    report.get(
                        "product_name_th"
                    ) or ""
                )[:180],
                "issues": [
                    x
                    for x in report.get(
                        "issues", []
                    )
                    if isinstance(
                        x, dict
                    )
                ][:24],
                "speech_summary_th": str(
                    report.get(
                        "speech_summary_th"
                    ) or ""
                )[:800],
                "audio_stats": audio_stats,
                "scale_review": (
                    report.get(
                        "scale_review"
                    )
                    if isinstance(
                        report.get(
                            "scale_review"
                        ),
                        dict,
                    )
                    else {}
                ),
            }

            return apply_scale_rule(
                clean
            )

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise AnalysisError(
                "Gemini ส่ง JSON ไม่ครบ: "
                f"{exc}"
            ) from exc

    raise AnalysisError(
        "Gemini ไม่ตอบกลับ"
    )


def inspect(
    path,
    key,
    model,
    mode,
    thumbnail,
):
    start = time.perf_counter()

    meta = probe(path)

    fps = FPS.get(
        mode, 2
    )

    with tempfile.TemporaryDirectory(
        prefix="qc_scale_v2_"
    ) as tmp:

        folder = Path(tmp)

        frames = frames_from_video(
            path,
            folder / "frames",
            fps,
        )

        thumbnail.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not thumbnail.exists():
            with Image.open(
                frames[0]
            ) as im:
                im.thumbnail(
                    (300, 300)
                )

                im.convert("RGB").save(
                    thumbnail,
                    "JPEG",
                    quality=74,
                )

        sheets = contact_sheets(
            frames,
            fps,
            folder,
        )

        # เพิ่มภาพเดี่ยวสำหรับตรวจขนาด
        details = inspect_frames(
            frames,
            fps,
        )

        if mode != "Fast":
            peaks = motion_peaks(
                frames,
                fps,
                meta["duration"],
            )

            for i, t in enumerate(peaks):
                out = folder / (
                    f"motion_{i}.jpg"
                )

                if extra_frame(
                    path,
                    max(0, t - 0.15),
                    out,
                ):
                    details.append(
                        (t, out)
                    )

        wav, stats = extract_audio(
            path,
            folder,
            meta["has_audio"],
        )

        result = gemini_analyze(
            key,
            model,
            sheets,
            details,
            wav,
            meta["duration"],
            stats,
        )

    result["duration"] = round(
        meta["duration"], 2
    )

    result["elapsed"] = round(
        time.perf_counter() - start,
        2,
    )

    return result
