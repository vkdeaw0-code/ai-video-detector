
"""AI Video QC V3 - Product Scale + Thai Speech Guard.

Compatible with V3 Batch Manager and Light ActiveQueue.
One Gemini request per video. No extra transcription API.
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
ENGINE_VERSION = "2026-10-11-v3-scale-thai-audio-guard-v1"

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
คุณเป็นผู้ตรวจคุณภาพคลิปโฆษณาที่สร้างด้วย AI
ความยาวประมาณ 10 วินาที

เป้าหมาย:
ผิดเล็กน้อยให้ผ่าน ผิดรุนแรงให้ตก

ภาพที่มีความเป็น AI บ้างไม่ใช่เหตุผล
ให้ตัดสิน FAIL โดยลำพัง

ต้องตรวจด้วยหลักฐานจากภาพตามเวลา
และเสียงจริงที่ส่งให้เท่านั้น

================================
ขั้นที่ 1 จำแนกประเภทสินค้า
================================

HUMAN:
มีคนยืนพูดหรือเห็นหน้าพรีเซน
ให้โฟกัสคนขาย สินค้าที่ถือ/ชี้ และเสียง
ไม่จับผิดฉากหลังเล็กน้อย

HANDS:
มือหรือแขนโชว์สินค้า
ตรวจมือ สินค้า สัดส่วน การใช้งาน
และเสียงบรรยาย

PRODUCT:
โชว์สินค้าอย่างเดียว
ตรวจรูปทรง ขนาด การใช้งาน
และเสียงบรรยาย

MIXED:
สลับหลายรูปแบบ
ให้ประเมินตามช่วงของคลิป

ต้องระบุสินค้าหลัก
ไม่ใช่สินค้าประกอบฉาก

================================
ขั้นที่ 2 ตรวจภาพและสเกลสินค้า V3
================================

ตรวจ:
- ความบิดเบี้ยวของสินค้า
- มือและนิ้วผิดรูป
- การหยิบจับสินค้า
- สินค้าเปลี่ยนรูป
- สินค้าหายหรือเปลี่ยนชนิด
- ความต่อเนื่องของสินค้า
- ขนาดสินค้าและการใช้งาน

สำหรับ HANDS PRODUCT MIXED:

ต้องตรวจว่าสินค้าเล็กหรือใหญ่
ผิดประเภทหรือไม่

เปรียบเทียบขนาดกับ:
- มือ
- นิ้ว
- โต๊ะ
- สิ่งของรอบข้าง
- วัตถุประสงค์การใช้งาน
- คำพูดที่ได้ยินจริง

หากสินค้าเล็กเหมือนโมเดลย่อส่วน
จนใช้งานตามที่นำเสนอไม่ได้
และมีหลักฐานอย่างน้อย 2 ด้าน
ให้ FAIL

สินค้าประเภท MINI
ของเล่น หรือรุ่นตั้งโต๊ะ
ที่ตั้งใจออกแบบให้เล็ก
ไม่ถือว่าผิดสเกล

มุมกล้อง การซูม และ Perspective
ไม่ใช่หลักฐานของสเกลผิดโดยลำพัง

สำหรับ HUMAN:

เน้นคนขาย
สินค้าที่ถือ
รูปทรงสินค้า
มือที่จับสินค้า
ความต่อเนื่อง
และเสียงพูด

ไม่จับผิดสเกลเกินจำเป็น
หรือข้อผิดพลาดเล็กน้อยของฉากหลัง

ห้ามเดาขนาดจริงเป็นเซนติเมตร
หากไม่มีข้อมูลอ้างอิง

================================
ขั้นที่ 3 THAI SPEECH QC
================================

สำคัญมาก:
ต้องฟังเสียงจริงจากคลิป

อย่าใช้เพียงระดับความดัง
หรือรูปร่างคลื่นเสียง
มาตัดสินว่าคำพูดถูกหรือผิด

ถ้ามีเสียงพูด:
ให้ประเมินว่าผู้ฟังทั่วไป
เข้าใจสาระสำคัญของโฆษณาหรือไม่

A. CLEAR

คำพูดชัดเจน
ฟังเข้าใจ
ไม่มีความเสียหายสำคัญ

ให้ PASS ได้
หากภาพสินค้าไม่มีข้อผิดพลาดสำคัญ

B. MINOR

คำพูดสะดุดเล็กน้อย
พูดผิดเล็กน้อย 1 คำ
ออกเสียงแปลกเล็กน้อย
หรือมีสำเนียง AI เล็กน้อย

แต่ยังเข้าใจความหมายสินค้า

ให้ PASS ได้
หากไม่มีข้อผิดพลาดสำคัญอื่น

ห้าม FAIL เพียงเพราะเสียงไม่สมบูรณ์ 100%

C. CRITICAL — เสียงเพี้ยนหลายคำ

เสียง AI พูดเป็นคำประหลาด
หลายคำต่อเนื่อง

ประโยคผิดเพี้ยนจนฟังไม่รู้เรื่อง
หรือไม่สามารถเข้าใจเนื้อหาโฆษณา

ให้ FAIL

D. CRITICAL — ข้อมูลสำคัญฟังไม่รู้เรื่อง

ถ้าผู้พูดกำลังกล่าวถึง:

- ขนาดสินค้า
- จำนวนชิ้น
- จำนวนชั้น
- ความจุ
- รุ่นสินค้า
- ชื่อสินค้าหลัก
- คุณสมบัติสำคัญ

แต่เสียงเพี้ยนหนักในช่วงนั้น
จนผู้ฟังไม่สามารถเข้าใจ
ข้อมูลสำคัญที่พยายามนำเสนอ

ให้ FAIL

ตัวอย่าง:

ผู้พูดกำลังบอกขนาดล้อแม็ก
แต่เสียงตัวเลขกลายเป็นคำประหลาด
จนฟังไม่ได้ว่าขอบเท่าไร

ถือเป็นข้อผิดพลาดสำคัญ

E. CRITICAL — คำพูดขัดกับสินค้า

หากได้ยินคำพูดชัดเจน
และมีหลักฐานยืนยันว่าคำพูด
ขัดกับสินค้าอย่างสำคัญ

เช่น:

พูดว่ามี 5 ชั้น
แต่ภาพยืนยันชัดว่ามี 3 ชั้น

ให้ FAIL

แต่หากไม่มีข้อมูลยืนยัน
ห้ามเดาว่าคำพูดผิด

F. CRITICAL — เสียงเสียหนัก

เสียงพูดแตกหนัก
เสียงหายบ่อย
เสียงขาดช่วงซ้ำ ๆ
เสียงกลายเป็นเสียงประหลาด

จนสาระสำคัญฟังไม่เข้าใจ

ให้ FAIL

================================
ข้อควรระวังในการตรวจเสียง
================================

1. ห้ามตัดตกเพียงเพราะเสียงฟังคล้าย AI

2. ห้ามตัดตกจากคำสะดุดเล็กน้อย 1 คำ
   ที่ไม่เปลี่ยนความหมายสำคัญ

3. ถ้าถอดเสียงไม่แน่ใจ
   ห้ามแต่งคำพูดขึ้นเอง

4. หากเสียงฟังไม่ชัดเพราะโมเดลไม่แน่ใจ
   แต่ยังไม่มีหลักฐานว่าเสียงเสียจริง
   อย่าฟันธงว่าพูดผิด

5. เสียงเพลงหรือเสียงประกอบ
   ไม่ใช่เสียงพูด

6. ถ้าไม่มีเสียงพูด
   อย่าตัดตกเพียงเพราะไม่มีบทพูด
   เว้นแต่มีเหตุผลสำคัญอื่น

7. ถ้าเสียงเสียหนักต้องอธิบาย
   ว่าเสียลักษณะใด
   และเกิดบริเวณวินาทีใด

8. การตรวจเสียงและสเกลสินค้า
   เป็นคนละเกณฑ์
   ห้ามเอาภาพผิดสเกลมาเป็นหลักฐาน
   ว่าเสียงพูดมีปัญหา

================================
ขั้นที่ 4 คำตัดสิน
================================

PASS:
คลิปใช้โฆษณาได้
สินค้าไม่มีข้อผิดพลาดรุนแรง
เสียงพูดเข้าใจได้
ข้อผิดพลาดเล็กน้อยยอมรับได้

FAIL:
สินค้าผิดสเกลอย่างชัดเจน
สินค้าผิดรูปหนัก
หรือเสียงพูดเสียหนัก
ตามเกณฑ์ CRITICAL

ห้ามเดาสิ่งที่มองไม่เห็น
หรือคำพูดที่ไม่ได้ยินจริง

================================
รูปแบบ JSON OUTPUT
================================

ตอบ JSON object เท่านั้น
ไม่มี Markdown

{
  "decision": "PASS",
  "video_type": "HANDS",
  "product_name_th": "ชื่อสินค้า",
  "reason_th": "เหตุผลภาษาไทย",
  "issues": [
    {
      "time_sec": 4.0,
      "category": "product",
      "severity": "minor",
      "detail_th": "รายละเอียดข้อผิดพลาด"
    }
  ],
  "speech_summary_th": "สรุปคำพูดที่ฟังได้จริง",
  "scale_review": {
    "size_grade": "normal",
    "category_scale_mismatch": false,
    "spoken_claim_conflict": false,
    "miniature_disclosed": false,
    "visual_evidence_th": "",
    "usage_evidence_th": "",
    "speech_evidence_th": "",
    "confidence": 0.0,
    "reason_th": ""
  },
  "audio_review": {
    "status": "clear",
    "speech_present": true,
    "severe_unintelligible": false,
    "key_information_unintelligible": false,
    "meaning_conflict": false,
    "severe_audio_artifact": false,
    "confidence": 0.0,
    "time_sec": 0.0,
    "evidence_th": ""
  }
}

size_grade เลือกได้:
normal
intentional_mini
suspicious
wrong_scale
unknown

audio_review.status เลือกได้:
clear
minor
critical
no_speech
uncertain

confidence ของ audio_review
คือความมั่นใจว่าเสียงเสียรุนแรงจริง
อยู่ระหว่าง 0 ถึง 1

ถ้าเสียงดีให้ confidence ต่ำหรือ 0

critical ต้องมีหลักฐานเสียงชัดเจน
ห้ามตั้ง critical เพียงเพราะคำพูด
ผิดเล็กน้อยหรือฟังไม่แน่ใจ

audio_review.evidence_th
ต้องอธิบายหลักฐานที่ได้ยินจริง
ไม่ใช่ปัญหาด้านภาพ
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

    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise AnalysisError(
            f"FFmpeg/FFprobe ผิดพลาด: {exc}"
        ) from exc

    if p.returncode:
        raise AnalysisError(
            (p.stderr or "FFmpeg ผิดพลาด")[-450:]
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
            data.get(
                "format", {}
            ).get(
                "duration", 0
            ) or 0
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise AnalysisError(
            "อ่านข้อมูลวิดีโอไม่ได้"
        ) from exc

    streams = data.get(
        "streams", []
    )

    if not any(
        s.get("codec_type") == "video"
        for s in streams
    ):
        raise AnalysisError(
            "ไม่พบแทร็กวิดีโอ"
        )

    if not 0.5 <= duration <= 25:
        raise AnalysisError(
            "รองรับคลิปความยาว 0.5-25 วินาที "
            f"ไฟล์นี้ {duration:.1f} วินาที"
        )

    return {
        "duration": duration,
        "has_audio": any(
            s.get("codec_type") == "audio"
            for s in streams
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

    frames = sorted(
        folder.glob("f_*.jpg")
    )

    if not frames:
        raise AnalysisError(
            "ดึงเฟรมวิดีโอไม่ได้"
        )

    return frames[:80]


def motion_peaks(frames, fps, duration):
    changes = []
    previous = None

    for i, file in enumerate(frames):
        with Image.open(file) as im:
            small = ImageOps.grayscale(
                im
            ).resize((48, 48))

        if previous is not None:
            amount = ImageStat.Stat(
                ImageChops.difference(
                    small,
                    previous,
                )
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

            sheet.paste(
                small,
                (x, y),
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
        sum(
            v * v
            for v in samples
        ) / len(samples)
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

    if (
        category == "HUMAN"
        or _as_bool(
            size.get("miniature_disclosed")
        )
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


def apply_audio_rule(report, audio_present):
    """
    บังคับ FAIL เมื่อ Gemini พบเสียงเสียรุนแรง
    พร้อมหลักฐาน โดยไม่เรียก API เพิ่ม
    """

    result = dict(report)

    audio = result.get(
        "audio_review"
    )

    if not isinstance(audio, dict):
        audio = {}

    result["audio_review"] = audio
    result["audio_override"] = False

    if not audio_present:
        audio.update({
            "status": "no_speech",
            "speech_present": False,
        })
        return result

    if not _as_bool(
        audio.get("speech_present")
    ):
        return result

    if str(
        audio.get("status", "")
    ).lower() != "critical":
        return result

    try:
        confidence = float(
            audio.get(
                "confidence", 0
            )
        )
    except (TypeError, ValueError):
        confidence = 0

    if (
        not math.isfinite(confidence)
        or confidence < 0.80
    ):
        return result

    evidence = str(
        audio.get(
            "evidence_th"
        ) or ""
    ).strip()

    flags = (
        "severe_unintelligible",
        "key_information_unintelligible",
        "meaning_conflict",
        "severe_audio_artifact",
    )

    has_critical_issue = any(
        _as_bool(audio.get(k))
        for k in flags
    )

    if not evidence or not has_critical_issue:
        return result

    reason = (
        "เสียงพูดมีปัญหารุนแรง: "
        + evidence[:330]
    )

    result["audio_override"] = (
        result.get("decision") == "PASS"
    )

    result["decision"] = "FAIL"

    old_reason = str(
        result.get(
            "reason_th"
        ) or ""
    ).strip()

    if not old_reason.startswith(
        "เสียงพูดมีปัญหารุนแรง:"
    ):
        result["reason_th"] = (
            reason
            + (
                " | " + old_reason
                if old_reason
                else ""
            )
        )[:800]

    try:
        sec = float(
            audio.get(
                "time_sec", 0
            )
        )

        if (
            not math.isfinite(sec)
            or sec < 0
        ):
            sec = 0

    except (ValueError, TypeError):
        sec = 0

    issues = result.get(
        "issues"
    )

    if not isinstance(issues, list):
        issues = []

    issues = list(issues)

    issues.append({
        "time_sec": round(sec, 1),
        "category": "audio",
        "severity": "critical",
        "detail_th": reason,
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
                f"ประมาณเวลา {sec:.1f} วินาที"
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
                "นี่คือเสียงต้นฉบับจากคลิป "
                "ตรวจการพูดภาษาไทยจริง "
                "ความเพี้ยน คำที่ฟังไม่รู้เรื่อง "
                "และข้อมูลสินค้าที่กล่าวถึง"
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
            "maxOutputTokens": 3000,
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
                    "JSON ต้องเป็น object"
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
                "PASS",
                "FAIL",
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

            issues = report.get(
                "issues", []
            )

            if not isinstance(issues, list):
                raise ValueError(
                    "issues ต้องเป็น list"
                )

            scale_review = report.get(
                "scale_review"
            )

            if kind in {
                "HANDS",
                "PRODUCT",
                "MIXED",
            }:
                valid_scale = (
                    isinstance(scale_review, dict)
                    and scale_review.get(
                        "size_grade"
                    ) in {
                        "normal",
                        "intentional_mini",
                        "suspicious",
                        "wrong_scale",
                        "unknown",
                    }
                )

                if not valid_scale:
                    raise ValueError(
                        "ไม่ได้รับผลตรวจสเกลสินค้า"
                    )

            audio_review = report.get(
                "audio_review"
            )

            if wav is not None:
                valid_audio = (
                    isinstance(audio_review, dict)
                    and audio_review.get(
                        "status"
                    ) in {
                        "clear",
                        "minor",
                        "critical",
                        "no_speech",
                        "uncertain",
                    }
                )

                if not valid_audio:
                    raise ValueError(
                        "ไม่ได้รับ audio_review "
                        "ที่จำเป็นสำหรับตรวจเสียง"
                    )

            if not isinstance(audio_review, dict):
                audio_review = {}

            try:
                audio_confidence = float(
                    audio_review.get(
                        "confidence", 0
                    )
                )
            except (TypeError, ValueError):
                audio_confidence = 0

            if not math.isfinite(
                audio_confidence
            ):
                audio_confidence = 0

            try:
                audio_time = float(
                    audio_review.get(
                        "time_sec", 0
                    )
                )
            except (TypeError, ValueError):
                audio_time = 0

            if not math.isfinite(audio_time):
                audio_time = 0

            safe_audio = {
                "status": str(
                    audio_review.get(
                        "status"
                    ) or "no_speech"
                )[:24],
                "speech_present": _as_bool(
                    audio_review.get(
                        "speech_present"
                    )
                ),
                "severe_unintelligible": _as_bool(
                    audio_review.get(
                        "severe_unintelligible"
                    )
                ),
                "key_information_unintelligible": _as_bool(
                    audio_review.get(
                        "key_information_unintelligible"
                    )
                ),
                "meaning_conflict": _as_bool(
                    audio_review.get(
                        "meaning_conflict"
                    )
                ),
                "severe_audio_artifact": _as_bool(
                    audio_review.get(
                        "severe_audio_artifact"
                    )
                ),
                "confidence": min(
                    1.0,
                    max(0.0, audio_confidence),
                ),
                "time_sec": min(
                    duration,
                    max(0.0, audio_time),
                ),
                "evidence_th": str(
                    audio_review.get(
                        "evidence_th"
                    ) or ""
                )[:500],
            }

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
                    x for x in issues
                    if isinstance(x, dict)
                ][:24],
                "speech_summary_th": str(
                    report.get(
                        "speech_summary_th"
                    ) or ""
                )[:800],
                "audio_stats": audio_stats,
                "audio_review": safe_audio,
                "scale_review": (
                    scale_review
                    if isinstance(
                        scale_review, dict
                    )
                    else {}
                ),
            }

            result = apply_scale_rule(
                clean
            )

            result = apply_audio_rule(
                result,
                audio_present=(
                    wav is not None
                ),
            )

            return result

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise AnalysisError(
                "Gemini ส่ง JSON ไม่ครบหรือไม่ถูกต้อง: "
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

    path = Path(path)
    thumbnail = Path(thumbnail)

    meta = probe(path)

    fps = FPS.get(
        mode, 2
    )

    with tempfile.TemporaryDirectory(
        prefix="qc_audio_v3_"
    ) as temp:
        folder = Path(temp)

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
                output = folder / (
                    f"motion_{i}.jpg"
                )

                if extra_frame(
                    path,
                    max(0, t - 0.15),
                    output,
                ):
                    details.append(
                        (t, output)
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

# =============================================================
# V3 SCALE DECEPTION GUARD v2
# วางโค้ดชุดนี้ท้ายสุดของ qc_core.py
# ไม่เปลี่ยนการตรวจเสียงหรือจำนวนครั้งที่เรียก Gemini
# =============================================================

ENGINE_VERSION = "v3-thai-speech-strict-scale-v2-20261011"

PROMPT += """

[IMPORTANT OVERRIDE — SCALE DECEPTION GUARD V2]
These instructions REPLACE the earlier scale decision rules only.
Keep the previous lenient Thai speech rules unchanged.

ตรวจสเกลสินค้าทุกประเภทคลิป รวม HUMAN, HANDS, PRODUCT, MIXED
อย่ามองว่าภาพคมชัดหรือสินค้าหน้าตาสวย แปลว่าใช้งานจริงได้

ต้องตรวจจากหลักฐานอิสระ 3 ด้าน:
1. HAND: เทียบขนาดสินค้ากับมือ นิ้ว แขน ขณะที่สัมผัสหรือชี้
2. ENVIRONMENT: เทียบกับโต๊ะ ชั้นวาง บรรจุภัณฑ์ คน และสิ่งของรอบตัว
3. FUNCTION: ขนาดเหมาะกับวัตถุประสงค์การใช้งานของสินค้านั้นจริงหรือไม่

ตัวอย่างที่ต้องระวัง ไม่ใช่รายการสำหรับตัดสินแบบอัตโนมัติ:
- แป้นบาสที่นำเสนอสำหรับใช้งานจริงแต่เล็กเพียงของตั้งโต๊ะ
- เครื่องครัว/หม้อ/เครื่องใช้ไฟฟ้าที่ดูคล้ายโมเดลจิ๋วเทียบกับมือ
- ถังขยะมีล้อสำหรับใช้งานบ้าน/อาคารแต่เล็กเกินจะใช้งานตามที่โฆษณา
- ชั้นวางของหรืออุปกรณ์ที่เล็กเกินความจุ/หน้าที่ที่อ้างไว้
- โคมไฟที่นำเสนอเป็นรุ่นติดตั้งจริงแต่แสดงสเกลเหมือนของจำลอง
ใช้แนวคิดเดียวกันกับสินค้าทุกประเภท แม้ไม่ได้อยู่ในตัวอย่างข้างบน

ถ้าเห็นว่าเล็กหรือใหญ่ผิดมากเมื่อเทียบกับมือและสิ่งแวดล้อม
และขัดกับหน้าที่ของสินค้าที่นำเสนอ ให้ size_grade=wrong_scale
ห้ามให้ PASS เพราะรูปร่างดูเนียน หากมีหลักฐานชัดเจนตั้งแต่ 2 ด้านขึ้นไป
ต้องอธิบายความไม่สมเหตุสมผลจากภาพจริง ไม่ใช่เดาความยาวเป็นเซนติเมตร

ตรวจการเคลื่อนไหวหลายช่วง: สินค้าชิ้นเดียวกันหด/ขยายสัดส่วน
หรือเปลี่ยนโครงสร้างแบบเป็นไปไม่ได้หรือไม่ เช่น ชั้นวางใหญ่ขึ้นโดยไม่มี
กลไกกาง/ปรับขนาดที่อธิบายได้ หากพบให้ temporal_scale_change=true
และอธิบาย 2 ช่วงเวลาที่เทียบกัน แต่การพับ/กางชั้นตามกลไกปกติไม่ผิด

ข้อยกเว้นสำคัญ: สินค้าจิ๋วอาจขายเป็นของเล่น โต๊ะจำลอง โมเดล
หรือรุ่น MINI ที่ใช้งานได้จริงตามจุดประสงค์ของมัน
หากมีคำพูดหรือป้ายระบุชัดว่าเป็นแบบจิ๋ว/ของเล่น/โมเดล ให้
miniature_disclosed=true และใส่ข้อความยืนยันใน explicit_mini_evidence_th
ห้ามสมมติว่า 'ตั้งใจจิ๋ว' เพียงเพราะเห็นว่าสินค้าเล็ก
หากอ้างว่าเป็นสินค้าขนาดปกติแล้วเห็นเล็กผิดมากให้ FAIL

หากมีแค่มุมกล้อง ซูม ภาพเบลอ หรือหลักฐานด้านเดียว
ไม่เพียงพอสรุปว่าสเกลหลอก ให้ suspicious / unknown ไม่ใช่ wrong_scale

ต้องกรอก key เหล่านี้ใน scale_review เพิ่มจาก schema เดิมด้วย:
"hand_evidence_th": "ความผิดสเกลที่สังเกตเทียบมือจริง หรือสตริงว่าง",
"environment_evidence_th": "ความผิดสเกลเทียบวัตถุในฉากจริง หรือสตริงว่าง",
"usage_evidence_th": "ความขัดแย้งกับการใช้งานจริง หรือสตริงว่าง",
"explicit_mini_evidence_th": "คำพูด/ข้อความที่ยืนยันว่าเป็นสินค้ามินิ หรือสตริงว่าง",
"temporal_scale_change": false,
"temporal_evidence_th": "หลักฐานเวลาสองช่วงเมื่อสินค้าขยาย/หดผิดธรรมชาติ หรือสตริงว่าง"

ใช้ scale_review.confidence แทนความมั่นใจว่ามีปัญหาสเกลจริง
หากหลักฐานชัดหลายด้านให้รายงานอย่างตรงไปตรงมา
ห้ามแต่งหลักฐานให้ครบเงื่อนไข
"""


def apply_scale_rule(report):
    """Override model PASS only for a well-supported severe scale error.

    Does not modify the Thai audio judgement, FFmpeg, number of frames,
    or number of Gemini requests. Still depends on model evidence.
    """
    result = dict(report)
    scale = result.get("scale_review")
    if not isinstance(scale, dict):
        scale = {}
    result["scale_review"] = scale
    result["scale_override"] = False

    def evidence(field):
        x = scale.get(field, "")
        return x.strip() if isinstance(x, str) else ""

    def strong_evidence(field):
        x = evidence(field)
        if len(x) < 12:
            return ""
        if x.startswith((
            "ไม่พบ", "ไม่มี", "ปกติ", "ไม่แน่ใจ", "ไม่ชัด", "ไม่สามารถยืนยัน"
        )):
            return ""
        return x

    try:
        conf = float(scale.get("confidence", 0) or 0)
    except (ValueError, TypeError):
        conf = 0.0
    if not math.isfinite(conf):
        conf = 0.0

    # Exempt genuine mini products only with explicit evidence.
    confirmed_mini = (
        _as_bool(scale.get("miniature_disclosed"))
        and len(strong_evidence("explicit_mini_evidence_th")) >= 8
    )

    # Independent anchors: hand, environment, intended usage.
    hand = strong_evidence("hand_evidence_th")
    environment = strong_evidence("environment_evidence_th")
    visual = strong_evidence("visual_evidence_th")
    usage = strong_evidence("usage_evidence_th")

    if not hand and not environment and visual:
        hand = visual

    anchors = sum(bool(x) for x in (hand, environment, usage))

    wrong_scale = (
        not confirmed_mini
        and str(scale.get("size_grade", "")).lower() == "wrong_scale"
        and conf >= 0.75
        and anchors >= 2
        and bool(visual or hand or environment)
    )

    # Do not confuse an ordinary folding mechanism with morphing.
    impossible_morph = (
        _as_bool(scale.get("temporal_scale_change"))
        and conf >= 0.84
        and len(strong_evidence("temporal_evidence_th")) >= 16
    )

    if not (wrong_scale or impossible_morph):
        return result

    if impossible_morph:
        explanation = (
            "สินค้าเปลี่ยนสเกลหรือโครงสร้างระหว่างคลิปผิดธรรมชาติ: "
            + evidence("temporal_evidence_th")
        )
    else:
        explanation = (
            "ขนาดสินค้าไม่สมเหตุสมผลเมื่อเทียบกับมือ/ฉาก/การใช้งาน: "
            + (evidence("reason_th") or "; ".join(
                x for x in (hand, environment, usage) if x
            ))
        )

    result["scale_override"] = result.get("decision") == "PASS"
    result["decision"] = "FAIL"

    previous = str(result.get("reason_th") or "").strip()
    if previous and previous != explanation:
        explanation = explanation + " | " + previous

    result["reason_th"] = explanation[:850]

    issues = result.get("issues")
    issues = list(issues) if isinstance(issues, list) else []

    issues.append({
        "time_sec": 0.0,
        "category": "product_scale",
        "severity": "critical",
        "detail_th": explanation[:600],
    })

    result["issues"] = issues[:25]
    return result
