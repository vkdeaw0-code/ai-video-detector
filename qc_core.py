
"""AI Video QC — Product Scale Guard V2.

Analyzes sampled frames and audio with Gemini.
Compatible with Batch Manager V3.
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

# เปลี่ยนเวอร์ชันเมื่อแก้เกณฑ์ตรวจ
# เพื่อไม่ให้ Cache ใช้คำตัดสินเดิม
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
คุณเป็นหัวหน้าฝ่ายตรวจคุณภาพวิดีโอสินค้า AI
(Thai advertising video quality inspector)

เป้าหมาย:
PASS สำหรับคลิปที่ดูดีและแสดงสินค้าตรงกับ
ชนิดสินค้า การใช้งาน และคำบรรยาย

FAIL สำหรับคลิปที่พบความผิดพลาดสำคัญ

วิดีโอมีความยาวประมาณ 10 วินาที
ได้รับภาพตามลำดับเวลา เฟรมเดี่ยว
และอาจมีเสียงจริงจากวิดีโอ

ต้องวิเคราะห์จากหลักฐานที่เห็นหรือได้ยินจริง
ห้ามตอบ PASS เพียงเพราะภาพดูสวย

================================
ขั้นที่ 1 จำแนกประเภทวิดีโอ
================================

ระบุสินค้าหลักที่ผู้ขายกำลังเสนอ
อย่านำสินค้าประกอบฉากด้านหลังมาเป็นสินค้าหลัก

HUMAN:
มีคนขายเต็มตัวหรือเห็นหน้าผู้ขายพูด

HANDS:
มีมือหรือแขนพรีเซนสินค้า
แต่ไม่เห็นคนขายเต็มตัว

PRODUCT:
แสดงสินค้าอย่างเดียว

MIXED:
สลับระหว่างหลายประเภท

================================
ขั้นที่ 2 ตรวจสอบเสียงและคำพูด
================================

ฟังเสียงจากวิดีโอและตรวจ:

- ชื่อสินค้าที่กล่าวถึง
- ขนาดสินค้า
- การใช้งานสินค้า
- ความจุ
- จำนวนชั้น
- วัตถุประสงค์การใช้งาน
- เสียงขาด
- เสียงพูดแตก
- เสียงพูดเพี้ยน
- คำพูดภาษาไทยที่ไม่สมเหตุสมผล

หากได้ยินไม่ชัด
ให้ระบุว่าไม่แน่ใจ

ห้ามแต่งคำพูดขึ้นเอง

================================
ขั้นที่ 3 PRODUCT SCALE AUDIT
================================

ใช้กับ HANDS, PRODUCT และ MIXED

ต้องตรวจสเกลสินค้าอย่างจริงจัง

ตรวจสอบ:

1. สินค้าประเภทอะไร
2. สินค้ามีไว้ใช้ทำอะไร
3. สินค้าที่เห็นมีขนาดสมเหตุสมผลหรือไม่
4. สินค้ามีขนาดเหมาะกับการใช้งานที่กล่าวอ้างหรือไม่
5. ขนาดเทียบกับมือและนิ้วเป็นอย่างไร
6. ขนาดเทียบกับวัตถุรอบข้างเป็นอย่างไร
7. สินค้าเหมือนแบบจำลองขนาดเล็กหรือไม่
8. สินค้าเล็กเกินกว่าจะใช้งานตามคำพูดหรือไม่
9. สินค้าบิดเบี้ยวหรือเปลี่ยนขนาดผิดปกติหรือไม่
10. สินค้าตรงกับประเภทที่โฆษณาหรือไม่

ให้เปรียบเทียบกับวัตถุอ้างอิง เช่น

- มือ
- นิ้ว
- กระป๋องอาหาร
- ขวดน้ำ
- อุปกรณ์ครัว
- โต๊ะ
- พื้น
- ชั้นวางของ
- วัตถุอื่นที่เห็นชัด

ตัวอย่างความผิดพลาดสำคัญ:

ชั้นวางของที่อ้างว่าใช้เก็บสิ่งของขนาดปกติ
แต่มีสัดส่วนเล็กเหมือนแบบจำลอง
และไม่มีพื้นที่เพียงพอสำหรับการใช้งานนั้น

ถังขยะสำหรับใช้งานทั่วไป
แต่มีขนาดเล็กผิดบริบทเมื่อเทียบกับมือ
และขัดกับคำบรรยาย

อุปกรณ์ครัวที่อ้างว่าใช้กับเครื่องใช้ขนาดปกติ
แต่สินค้าเล็กเกินกว่าจะรองรับเครื่องใช้เหล่านั้น

แม้สินค้าดูเนียน สวย และไม่บิดเบี้ยว
แต่ผิดสเกลการใช้งานอย่างชัดเจน
ให้ FAIL

แม้สินค้าไม่เปลี่ยนขนาดตลอดคลิป
ก็อาจผิดสเกลการใช้งานได้

ต้องอธิบายหลักฐานอย่างชัดเจน

================================
ข้อควรระวังเรื่องสินค้า MINI
================================

สินค้าบางอย่างถูกออกแบบมาให้เล็กจริง เช่น

- ของเล่น
- สินค้ามินิ
- อุปกรณ์ตั้งโต๊ะ
- ชั้นวางขนาดเล็ก
- สินค้าจำลอง
- ห่วงบาสตั้งโต๊ะ

ถ้าเป็นรุ่นเล็กโดยตั้งใจ
และโฆษณาตรงกับชนิดสินค้า
ห้าม FAIL เพียงเพราะเล็ก

ต้องระวังการซูมภาพ
ระยะกล้อง และมุม Perspective

ห้ามสรุปจากการที่สินค้า
กินพื้นที่ในภาพน้อยเพียงอย่างเดียว

ถ้าหลักฐานไม่เพียงพอ
ห้ามเดาขนาดจริงขึ้นเอง

================================
ขั้นที่ 4 ตรวจคลิป HUMAN
================================

หากมีคนยืนขายสินค้า

เน้นตรวจเฉพาะ:

- สินค้าหลักที่คนถือ
- สินค้าที่อยู่ตรงกลาง
- รูปทรงสินค้า
- สินค้าบิดเบี้ยวหรือไม่
- สินค้าเปลี่ยนรูปร่างหรือไม่
- มือที่จับสินค้าผิดรูปหรือไม่
- สินค้าทะลุมือหรือไม่
- การเคลื่อนไหวของสินค้า
- เสียงพูดผิดเพี้ยนหรือไม่
- ปากและเสียงตรงกันหรือไม่เมื่อมองเห็นชัด

อย่าจับผิดฉากหลังหรือรายละเอียดเล็กน้อย
ที่ไม่กระทบคุณภาพโฆษณา

หากสินค้าและเสียงดูดี
และไม่มีความผิดพลาดสำคัญ
ให้ PASS

================================
ขั้นที่ 5 ตรวจคุณภาพเพิ่มเติม
================================

ตรวจสอบ:

- Product Geometry
- Product Scale
- Product Identity
- Product Continuity
- Hand Consistency
- Object Interaction
- Motion Continuity
- Audio Distortion
- Thai Speech Quality
- Advertising Usability

ไม่ต้อง FAIL จากข้อผิดพลาดเล็กน้อย
ที่คนดูทั่วไปไม่สังเกต

================================
ขั้นที่ 6 ตัดสินผล
================================

PASS:
คุณภาพดี ใช้โฆษณาได้
สินค้าและคำพูดสอดคล้องกัน
ไม่มีข้อผิดพลาดสำคัญ

FAIL:
มีข้อผิดพลาดสำคัญ เช่น

1. สินค้าเล็กผิดประเภทอย่างชัดเจน
2. สินค้าขัดกับการใช้งานที่โฆษณา
3. สินค้าเปลี่ยนรูปทรงผิดธรรมชาติ
4. มือผิดรูปอย่างชัดเจน
5. เสียงแตกหรือขาดจนรบกวน
6. การเคลื่อนไหวผิดปกติอย่างรุนแรง

สำหรับการตัดสินผิดสเกล
ต้องมีหลักฐานอย่างน้อยสองด้าน

เช่น:
ขนาดเทียบมือ + วัตถุประสงค์การใช้งาน

หรือ:
ภาพสินค้า + คำพูดที่ขัดแย้ง

อย่าเดาจากมุมกล้องเพียงอย่างเดียว

ห้ามอ้างสิ่งที่ไม่ได้เห็นในเฟรมตัวอย่าง
ห้ามเดา Transcript หากเสียงไม่ชัด

================================
รูปแบบ JSON OUTPUT
================================

ตอบเป็น JSON object เท่านั้น
ไม่มี Markdown

{
  "decision": "PASS",
  "video_type": "HANDS",
  "product_name_th": "ชื่อสินค้าหลัก",
  "reason_th": "เหตุผลสั้นภาษาไทย",
  "issues": [
    {
      "time_sec": 4.0,
      "category": "product",
      "severity": "critical",
      "detail_th": "รายละเอียดข้อผิดพลาด"
    }
  ],
  "speech_summary_th": "สรุปคำพูดที่ได้ยินจริง",
  "scale_review": {
    "size_grade": "normal",
    "category_scale_mismatch": false,
    "spoken_claim_conflict": false,
    "miniature_disclosed": false,
    "visual_evidence_th": "หลักฐานจากภาพ",
    "usage_evidence_th": "หลักฐานด้านการใช้งาน",
    "speech_evidence_th": "หลักฐานจากคำพูด",
    "confidence": 0.0,
    "reason_th": "สรุปเรื่องสเกล"
  }
}

scale_review.size_grade เลือกได้เฉพาะ:

normal
intentional_mini
suspicious
wrong_scale
unknown

wrong_scale:
ใช้เมื่อมีหลักฐานชัดเจน

suspicious:
มีข้อสงสัยแต่ยังไม่มีหลักฐานเพียงพอ

category_scale_mismatch:
เป็น true เฉพาะสเกลที่ขัดกับชนิดสินค้า
และวัตถุประสงค์อย่างชัดเจน

spoken_claim_conflict:
เป็น true เฉพาะเมื่อคำพูดที่ได้ยินจริง
ขัดกับสิ่งที่เห็น

miniature_disclosed:
เป็น true หากระบุชัดว่า
สินค้าถูกออกแบบมาเป็นมินิ
ของเล่น ตั้งโต๊ะ หรือรุ่นเล็ก

confidence:
ตัวเลข 0 ถึง 1
ใช้แสดงความมั่นใจว่าสเกลผิดจริง

สำหรับ HUMAN:
หากไม่ได้ตรวจสเกล
ให้ระบุ size_grade=unknown
"""


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


def command(
    args: list[str],
    timeout: int = 90,
) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise AnalysisError(
            f"FFmpeg/FFprobe ไม่พร้อมหรือหมดเวลา: {exc}"
        ) from exc

    if p.returncode:
        raise AnalysisError(
            (p.stderr or "FFmpeg ผิดพลาด")[-450:]
        )

    return p


def probe(path: Path) -> dict:
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

    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise AnalysisError(
            "อ่านข้อมูลระยะเวลาวิดีโอไม่ได้"
        ) from exc

    streams = data.get("streams", [])

    if not any(
        s.get("codec_type") == "video"
        for s in streams
    ):
        raise AnalysisError(
            "ไม่พบแทร็กวิดีโอ"
        )

    if not 0.5 <= duration <= 25:
        raise AnalysisError(
            "รองรับคลิปประมาณ 10 วินาที "
            f"(ไฟล์นี้ {duration:.1f} วินาที; "
            "จำกัด 0.5–25 วินาที)"
        )

    return {
        "duration": duration,
        "has_audio": any(
            s.get("codec_type") == "audio"
            for s in streams
        ),
    }


def frames_from_video(
    path: Path,
    folder: Path,
    fps: int,
) -> list[Path]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # อ่านวิดีโอรอบเดียว
    # ไม่โหลดทุกเฟรมเข้า RAM
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

    frames = sorted(
        folder.glob("f_*.jpg")
    )

    if not frames:
        raise AnalysisError(
            "ดึงเฟรมไม่ได้"
        )

    return frames[:80]


def motion_peaks(
    frames: list[Path],
    fps: int,
    duration: float,
) -> list[float]:
    changes = []
    previous = None

    for i, file in enumerate(frames):
        with Image.open(file) as im:
            small = ImageOps.grayscale(
                im
            ).resize((48, 48))

        if previous is not None:
            difference = ImageChops.difference(
                small,
                previous,
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


def extra_frame(
    path: Path,
    at: float,
    out: Path,
) -> bool:
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


def contact_sheets(
    frames: list[Path],
    fps: int,
    folder: Path,
) -> list[Path]:
    result = []

    for start in range(
        0,
        len(frames),
        10,
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

            sheet.paste(small, (x, y))

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


def inspect_frames(
    frames: list[Path],
    fps: int,
) -> list[tuple[float, Path]]:
    indexes = sorted(set(
        round(
            x * (len(frames) - 1)
        )
        for x in (
            0.10,
            0.32,
            0.57,
            0.80,
            0.94,
        )
    ))

    return [
        (i / fps, frames[i])
        for i in indexes
    ]


def extract_audio(
    path: Path,
    folder: Path,
    has_audio: bool,
) -> tuple[Path | None, dict]:

    if not has_audio:
        return None, {
            "has_audio": False,
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
        str(audio),
        "rb",
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


def inline_part(
    path: Path,
    mime: str,
) -> dict:
    data = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return {
        "inline_data": {
            "mime_type": mime,
            "data": data,
        }
    }


def _as_bool(value: object) -> bool:
    return (
        value is True
        or (
            isinstance(value, str)
            and value.lower() == "true"
        )
    )


def apply_scale_rule(report: dict) -> dict:
    """
    ใช้ตรวจคำตัดสินอีกชั้น

    ถ้า Gemini ให้ PASS แต่มีหลักฐาน
    ว่าสเกลผิดอย่างชัดเจน ให้เปลี่ยนเป็น FAIL
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

    # คลิป HUMAN ใช้กฎภาพและเสียงทั่วไป
    if category == "HUMAN":
        return result

    # ไม่ตัดตกสินค้า MINI ที่ระบุชัด
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
            size.get("confidence", 0)
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

    # ต้องมีหลักฐานสองด้าน
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
        "detail_th": result["reason_th"],
    })

    result["issues"] = issues[:25]

    return result


def gemini_analyze(
    key: str,
    model: str,
    sheets: list[Path],
    details: list[tuple[float, Path]],
    wav: Path | None,
    duration: float,
    audio_stats: dict,
) -> dict:

    if model not in ALLOWED_MODELS:
        raise AnalysisError(
            "ชื่อโมเดลไม่ได้อยู่ในรายการที่อนุญาต"
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
                "ให้ดูความต่อเนื่องและขนาด"
                "สัมพันธ์กับมือ/สิ่งของ"
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
                "เฟรมเดี่ยวความละเอียดสูงกว่า "
                f"ที่เวลาใกล้ {sec:.1f} วินาที "
                "ตรวจอัตราส่วนเทียบมือและการใช้งาน"
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
                "เสียงจริงจากวิดีโอเดียวกัน "
                "ให้ฟังคำอ้างขนาด/การใช้งาน "
                "และความเพี้ยนของคำพูดภาษาไทย"
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
                "Gemini API ถึงโควตาหรือจำกัดความถี่ "
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
                    "response is not a JSON object"
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

            if (
                decision not in {"PASS", "FAIL"}
                or kind not in {
                    "HUMAN",
                    "HANDS",
                    "PRODUCT",
                    "MIXED",
                }
            ):
                raise ValueError(
                    "decision/video_type ไม่ถูกต้อง"
                )

            if not isinstance(
                report.get("issues", []),
                list,
            ):
                raise ValueError(
                    "issues ต้องเป็น list"
                )

            # HANDS/PRODUCT/MIXED
            # ต้องได้รับผลตรวจสเกล
            if kind in {
                "HANDS",
                "PRODUCT",
                "MIXED",
            }:
                review = report.get(
                    "scale_review"
                )

                if (
                    not isinstance(review, dict)
                    or review.get("size_grade")
                    not in {
                        "normal",
                        "intentional_mini",
                        "suspicious",
                        "wrong_scale",
                        "unknown",
                    }
                ):
                    raise ValueError(
                        "ไม่ได้คืน scale_review ที่จำเป็น"
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
                    if isinstance(x, dict)
                ][:24],
                "speech_summary_th": str(
                    report.get(
                        "speech_summary_th"
                    ) or ""
                )[:800],
                "audio_stats": audio_stats,
                "scale_review": (
                    report.get("scale_review")
                    if isinstance(
                        report.get("scale_review"),
                        dict,
                    )
                    else {}
                ),
            }

            return apply_scale_rule(clean)

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise AnalysisError(
                "Gemini ส่ง JSON ไม่ครบตามที่กำหนด: "
                f"{exc}"
            ) from exc

    raise AnalysisError(
        "Gemini ไม่ตอบกลับ"
    )


def inspect(
    path: Path,
    key: str,
    model: str,
    mode: str,
    thumbnail: Path,
) -> dict:

    start = time.perf_counter()

    meta = probe(path)
    fps = FPS.get(mode, 2)

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
        meta["duration"],
        2,
    )

    result["elapsed"] = round(
        time.perf_counter() - start,
        2,
    )

    return result
