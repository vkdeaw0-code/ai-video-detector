
"""V3.4 Turbo Smart Quality Guard.

Checks product scale, Thai speech, and product/speech consistency.
Performs independent recheck based on risk and safety sampling.

Compatible with existing V3 qc_store.py and Streamlit app.py.
Requires ffmpeg, ffprobe, requests and Pillow.
"""

from __future__ import annotations

import base64
import hashlib
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
from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat


# เปลี่ยนเวอร์ชันเพื่อไม่ใช้ผล Cache เก่า
ENGINE_VERSION = "v3-4-turbo-risk-audit-20261011-r1"

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

# โหมดตรวจยืนยัน:
# smart  = ตรวจซ้ำเมื่อเสี่ยง + ตรวจตัวอย่างคลิปปกติ 50%
# strict = ตรวจซ้ำทุกคลิปที่รอบแรก PASS
# fast   = ตรวจซ้ำเฉพาะคลิปที่มีสัญญาณเสี่ยง
AUDIT_MODE = "smart"

# สัดส่วนการตรวจยืนยันคลิปที่ดูปกติ
# ใช้เฉพาะโหมด smart
CLEAN_PASS_AUDIT_PERCENT = 50

FRAME_SIZE = 768
MAX_FRAMES = 75


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


FIRST_PROMPT = """
คุณเป็นผู้ตรวจคุณภาพวิดีโอโฆษณาที่สร้างด้วย AI
สำหรับการขายสินค้า

ฟังเสียงจริงและตรวจสินค้าจากทุกภาพที่ได้รับ
ห้ามเดาเสียงพูดจากภาพ

เป้าหมาย:
ตรวจข้อผิดพลาดสำคัญให้พบ
แต่ยอมให้ข้อบกพร่องเล็กน้อย PASS

========================================
1. ตรวจภาพสินค้า
========================================

จำแนกประเภทคลิป:
HUMAN, HANDS, PRODUCT, MIXED

ต้องตรวจขนาดสินค้าทุกประเภท
แม้มีคนพูดขายสินค้าก็ตาม

ตรวจ:
- รูปร่างและความต่อเนื่องของสินค้า
- มือและนิ้วที่จับสินค้า
- สินค้าบิดเบี้ยวรุนแรง
- สินค้าหายหรือเปลี่ยนชนิด
- สินค้าเปลี่ยนขนาดระหว่างคลิป
- สัดส่วนสินค้าเทียบกับมือ
- ขนาดสินค้าเทียบสิ่งแวดล้อม
- ความสมเหตุสมผลในการใช้งาน

ห้ามให้ PASS เพียงเพราะภาพสวย
หรือสินค้าไม่มีความผิดปกติด้านสีและพื้นผิว

========================================
2. ตรวจสเกลสินค้า
========================================

เปรียบเทียบขนาดสินค้ากับ:

A. มือและนิ้วที่สัมผัสสินค้า
B. วัตถุในระนาบใกล้เคียงกัน
C. วัตถุประสงค์การใช้งานจริง
D. คำพูดเกี่ยวกับขนาดหรือประเภทสินค้า

ระวัง:
- สินค้าจิ๋วที่ถูกเสนอเป็นสินค้าขนาดปกติ
- สินค้าขนาดใหญ่เกินความเป็นจริง
- ขนาดไม่เหมาะกับการใช้งานที่โฆษณา
- สินค้าหดหรือขยายอย่างผิดธรรมชาติ

ถ้าสเกลผิดอย่างชัดเจน
และมีหลักฐานหลายด้าน
ให้ FAIL

ข้อยกเว้น:
สินค้า MINI ของเล่น หรือสินค้าตั้งโต๊ะ
ที่มีขนาดเล็กจริงและนำเสนอถูกประเภท
สามารถ PASS ได้

ขนาดเล็กเพียงอย่างเดียว
ไม่ใช่หลักฐานว่าสินค้าผิด

มุมกล้อง การซูม หรือการพับกาง
ตามกลไกปกติไม่ใช่หลักฐานผิดสเกล

ห้ามเดาขนาดเป็นนิ้วหรือเซนติเมตร
ถ้าไม่มีหลักฐานยืนยัน

========================================
3. ตรวจเสียงพูดภาษาไทย
========================================

ต้องฟังเสียงจริงตลอดคลิป

PASS:
- เสียงชัดเจน
- คำพูดฟังเข้าใจ
- สำเนียง AI แปลกเล็กน้อย
- ออกเสียงผิดเล็กน้อย 1 คำ
  โดยความหมายไม่เปลี่ยน

FAIL:
- เสียงพูดเพี้ยนจริงตั้งแต่ 2 คำขึ้นไป
- เสียงมั่วหลายคำต่อเนื่อง
- เสียงแตกหรือขาดจนฟังไม่เข้าใจ
- ข้อมูลสำคัญถูกพูดผิดอย่างชัดเจน
- ขนาด จำนวน หรือชนิดสินค้า
  ขัดกับหลักฐานยืนยัน

ห้ามนับสิ่งเหล่านี้เป็นคำผิด:
- สำเนียงท้องถิ่น
- ชื่อแบรนด์
- ศัพท์เทคนิค
- คำทับศัพท์
- การพูดเร็วแต่ฟังเข้าใจ
- ลักษณะเสียงสังเคราะห์เพียงอย่างเดียว

ห้ามแต่งคำผิดเพื่อให้ครบ 2 คำ

========================================
4. ตรวจคำพูดสัมพันธ์กับสินค้า
========================================

ตรวจว่าคนพูดกำลังขายสินค้า
ประเภทเดียวกับสิ่งที่เห็นจริงหรือไม่

ตรวจ:
- ชื่อสินค้า
- ประเภทสินค้า
- ขนาด
- จำนวน
- ความจุ
- คุณสมบัติ
- การใช้งานจริง

หากผู้พูดกล่าวถึงสินค้าอย่างหนึ่ง
แต่ภาพยืนยันว่าเป็นอีกอย่างหนึ่ง
ให้ FAIL

หากมีข้อมูลสินค้าอ้างอิงจากผู้ขาย
ให้ใช้เปรียบเทียบเพิ่มเติม

หากไม่มีข้อมูลยืนยันขนาดจริง
ห้ามเดาว่าตัวเลขที่พูดผิด

========================================
5. ตรวจความเสี่ยงก่อน PASS
========================================

แม้ตัดสิน PASS ก็ต้องระบุ risk_flags
ตามความผิดปกติที่ตรวจพบจริง

ห้ามตั้งค่า normal เพียงเพราะภาพสวย

ตรวจว่ามีหลักฐานด้านสเกล
อย่างน้อย 2 ด้านหรือไม่

หากไม่ได้ตรวจยืนยันครบ
ให้ scale_checks_complete=false

หากไม่แน่ใจว่าเสียงถูกต้อง
ให้ speech_checks_complete=false

หากข้อมูลไม่พอ
ให้ระบุ risk_flags เช่น:
scale_uncertain
audio_uncertain
small_object
visual_scale_anomaly
scale_change
speech_product_mismatch
gibberish
ambiguous_size_claim

ห้ามแต่งหลักฐานหรือความเสี่ยงขึ้นเอง

========================================
6. JSON OUTPUT
========================================

ตอบ JSON object เท่านั้น
ห้ามมี Markdown

{
 "decision":"PASS",
 "video_type":"HANDS",
 "product_name_th":"ชื่อสินค้าหลัก",
 "reason_th":"เหตุผลภาษาไทย",
 "speech_summary_th":"สรุปคำพูดที่ฟังได้จริง",
 "risk_flags":[],
 "scale_checks_complete":false,
 "speech_checks_complete":false,
 "issues":[
   {
     "time_sec":2.0,
     "category":"product_scale",
     "severity":"minor",
     "detail_th":"รายละเอียด"
   }
 ],
 "scale_review":{
   "size_grade":"normal",
   "confidence":0.0,
   "hand_evidence_th":"",
   "environment_evidence_th":"",
   "function_evidence_th":"",
   "visual_evidence_th":"",
   "usage_evidence_th":"",
   "speech_evidence_th":"",
   "miniature_disclosed":false,
   "explicit_mini_evidence_th":"",
   "temporal_scale_change":false,
   "temporal_evidence_th":"",
   "reason_th":"",
   "claim_consistency":"matched"
 },
 "audio_review":{
   "status":"clear",
   "speech_present":true,
   "confidence":0.0,
   "broken_word_count":0,
   "broken_words_evidence_th":"",
   "unintelligible_span_count":0,
   "speech_visual_relation":"matched",
   "severe_unintelligible":false,
   "key_information_unintelligible":false,
   "meaning_conflict":false,
   "severe_audio_artifact":false,
   "time_sec":0.0,
   "evidence_th":""
 }
}

size_grade:
normal, intentional_mini, suspicious,
wrong_scale, unknown

audio_review.status:
clear, minor, critical, no_speech, uncertain

claim_consistency และ speech_visual_relation:
matched, mismatch, uncertain, not_applicable

confidence หมายถึงความมั่นใจ
ว่ามีข้อผิดพลาดร้ายแรงจริง
ถ้าไม่มีปัญหาให้ใช้ 0
"""


# ผู้ตรวจรอบสองไม่ได้รับผล PASS รอบแรก
AUDIT_PROMPT = """
คุณเป็นผู้ตรวจคุณภาพอิสระ
สำหรับภาพและเสียงโฆษณาที่สร้างด้วย AI

คุณไม่ได้รับคำตัดสินจากผู้ตรวจรอบแรก

จงตรวจใหม่โดยไม่สมมติว่าคลิปนี้ดี

========================================
A. ตรวจสเกล
========================================

ตรวจจากหลักฐาน:

1. มือ/นิ้วที่สัมผัสสินค้า
2. วัตถุใกล้เคียงในระนาบเดียวกัน
3. ความสามารถใช้งานตามที่โฆษณา
4. สินค้าเปลี่ยนขนาดผิดธรรมชาติหรือไม่
5. คำพูดสัมพันธ์กับขนาดหรือไม่

หากขนาดสินค้าไม่สอดคล้องกับ
การใช้งานจริงอย่างรุนแรง
และมีหลักฐานอย่างน้อย 2 ด้าน
โดยหนึ่งด้านต้องเกี่ยวกับการใช้งาน
ให้ scale.decision=FAIL

ห้ามตัด FAIL เพียงเพราะสินค้าเล็ก

สินค้า MINI/ของเล่น/ตั้งโต๊ะจริง
ที่นำเสนอถูกประเภท ให้ PASS ได้

อย่าอ้างว่าสินค้าเป็น MINI
โดยไม่มีหลักฐานการนำเสนอ

การซูมกล้องและพับกางตามกลไกปกติ
ไม่ใช่การเปลี่ยนขนาดผิดธรรมชาติ

หากหลักฐานไม่พอ ให้ UNCERTAIN

========================================
B. ตรวจเสียงพูดใหม่อย่างอิสระ
========================================

ฟังเสียงจริงตั้งแต่ต้นจนจบ

PASS:
- คำพูดชัด
- ผิดเล็กน้อยไม่เกิน 1 คำ
- ความหมายยังถูกต้อง

FAIL:
- เสียงเพี้ยนจริงตั้งแต่ 2 คำขึ้นไป
- เสียงมั่วต่อเนื่องหลายคำ
- เสียงเสียจนฟังข้อมูลสำคัญไม่เข้าใจ
- คำพูดขัดกับสินค้าอย่างยืนยันได้

ต้องระบุเวลาและหลักฐานคำเพี้ยน
แต่ละคำที่ได้ยินจริง

ห้ามนับคำเดิมซ้ำ
ห้ามแต่งคำผิดให้ครบจำนวน
ห้ามนับสำเนียงหรือชื่อแบรนด์เป็นคำผิด

ถ้าไม่มีเสียงพูด:
speech_present=false
speech.decision=PASS
events=[]

ถ้าหลายคำมั่วติดกัน
จนแยกคำไม่ได้จริง
ใช้ continuous_gibberish=true

รูปภาพช่วยระบุชนิดสินค้า
แต่ใช้แทนการฟังเสียงไม่ได้

========================================
C. JSON OUTPUT
========================================

ตอบ JSON object เท่านั้น

{
 "scale":{
   "decision":"PASS",
   "confidence":0.0,
   "intentional_mini":false,
   "mini_evidence_th":"",
   "hand_evidence_th":"",
   "environment_evidence_th":"",
   "function_evidence_th":"",
   "impossible_size_change":false,
   "time_a":0.0,
   "time_b":0.0,
   "temporal_evidence_th":"",
   "reason_th":""
 },
 "speech":{
   "decision":"PASS",
   "speech_present":true,
   "confidence":0.0,
   "events":[
     {
       "time_sec":2.2,
       "heard_fragment_th":"คำที่ได้ยินจริง",
       "why_problem_th":"เหตุผลที่เสียงเพี้ยน",
       "confidence":0.0
     }
   ],
   "continuous_gibberish":false,
   "continuous_start_sec":0.0,
   "continuous_end_sec":0.0,
   "continuous_broken_word_count":0,
   "continuous_evidence_th":"",
   "critical_claim_verified":false,
   "spoken_claim_th":"",
   "visual_or_reference_evidence_th":"",
   "speech_summary_th":"",
   "reason_th":""
 }
}

decision:
PASS, FAIL, UNCERTAIN

confidence:
ความมั่นใจว่ามีข้อผิดพลาดรุนแรงจริง

ถ้าไม่มีคำผิด ต้องให้ events=[]

ห้ามสร้างหลักฐานให้ครบเกณฑ์
"""


def _run(args, timeout=120):
    try:
        process = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AnalysisError(
            f"FFmpeg/FFprobe ทำงานไม่ได้: {exc}"
        ) from exc

    if process.returncode:
        raise AnalysisError(
            "วิดีโอ/เสียงอ่านไม่ได้: "
            + (process.stderr or "")[-400:]
        )

    return process.stdout


def probe(path):
    raw = _run([
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of", "json",
        str(path),
    ])

    try:
        data = json.loads(raw)
        duration = float(
            data.get("format", {}).get("duration", 0)
        )
        streams = data.get("streams", [])
    except (TypeError, ValueError, KeyError) as exc:
        raise AnalysisError(
            "ไม่สามารถอ่านข้อมูลวิดีโอ"
        ) from exc

    if not math.isfinite(duration) or not 0.5 <= duration <= 25:
        raise AnalysisError(
            f"รองรับวิดีโอ 0.5–25 วินาที "
            f"(ไฟล์นี้ {duration:.1f}s)"
        )

    if not any(
        stream.get("codec_type") == "video"
        for stream in streams
    ):
        raise AnalysisError("ไม่พบภาพในไฟล์วิดีโอ")

    return {
        "duration": duration,
        "has_audio": any(
            stream.get("codec_type") == "audio"
            for stream in streams
        ),
    }


def _frames(path, folder, fps):
    folder.mkdir(parents=True, exist_ok=True)

    video_filter = (
        f"fps={fps},"
        f"scale={FRAME_SIZE}:{FRAME_SIZE}:"
        "force_original_aspect_ratio=decrease:"
        "force_divisible_by=2"
    )

    _run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin", "-y",
        "-threads", "2",
        "-i", str(path),
        "-vf", video_filter,
        "-q:v", "5",
        str(folder / "frame_%03d.jpg"),
    ], 140)

    frames = sorted(
        folder.glob("frame_*.jpg")
    )[:MAX_FRAMES]

    if not frames:
        raise AnalysisError(
            "ดึงภาพจากวิดีโอไม่สำเร็จ"
        )

    return frames


def _sheets(frames, fps, folder):
    outputs = []

    for offset in range(0, len(frames), 12):
        canvas = Image.new(
            "RGB",
            (4 * 260, 3 * 242),
            (24, 28, 39),
        )

        draw = ImageDraw.Draw(canvas)

        for n, path in enumerate(
            frames[offset:offset + 12]
        ):
            with Image.open(path) as image:
                thumbnail = ImageOps.contain(
                    image.convert("RGB"),
                    (250, 208),
                )

            x = (
                (n % 4) * 260
                + (260 - thumbnail.width) // 2
            )
            y = (
                (n // 4) * 242
                + 22
                + (208 - thumbnail.height) // 2
            )

            canvas.paste(thumbnail, (x, y))

            draw.text(
                (
                    (n % 4) * 260 + 8,
                    (n // 4) * 242 + 4,
                ),
                f"{(offset + n) / fps:.1f}s",
                fill="white",
            )

        destination = folder / (
            f"sheet_{len(outputs):02d}.jpg"
        )

        canvas.save(
            destination,
            "JPEG",
            quality=80,
            optimize=True,
        )

        outputs.append(destination)

    return outputs


def _motion_samples(path, frames, fps, duration, folder):
    changes = []
    previous = None

    for index, file in enumerate(frames):
        with Image.open(file) as image:
            current = ImageOps.grayscale(
                image
            ).resize((48, 48))

        if previous is not None:
            score = ImageStat.Stat(
                ImageChops.difference(
                    current,
                    previous,
                )
            ).mean[0]

            changes.append((score, index / fps))

        previous = current

    changes.sort(reverse=True)
    picks = []

    for score, seconds in changes:
        if (
            score < 15
            or seconds > duration - 0.1
            or seconds < 0.15
        ):
            continue

        if all(
            abs(seconds - picked) > 0.8
            for picked in picks
        ):
            picks.append(seconds)

        if len(picks) == 2:
            break

    output = []

    for index, seconds in enumerate(sorted(picks)):
        destination = folder / (
            f"motion_{index}.jpg"
        )

        try:
            _run([
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-nostdin", "-y",
                "-ss", f"{max(0, seconds - 0.14):.2f}",
                "-i", str(path),
                "-frames:v", "1",
                "-vf",
                f"scale={FRAME_SIZE}:{FRAME_SIZE}:"
                "force_original_aspect_ratio=decrease:"
                "force_divisible_by=2",
                str(destination),
            ], 30)

            if destination.is_file():
                output.append(
                    (seconds, destination)
                )

        except AnalysisError:
            pass

    return output


def _audio(path, folder, has_audio):
    if not has_audio:
        return None, {"has_audio": False}

    destination = folder / "audio.wav"

    _run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin", "-y",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(destination),
    ], 75)

    with wave.open(str(destination), "rb") as sound:
        samples = array("h")
        samples.frombytes(
            sound.readframes(
                sound.getnframes()
            )
        )

    if not samples:
        return None, {
            "has_audio": True,
            "silent": True,
        }

    rms = math.sqrt(
        sum(value * value for value in samples)
        / len(samples)
    ) / 32768.0

    clipping = sum(
        abs(value) > 32300
        for value in samples
    ) / len(samples)

    stats = {
        "has_audio": True,
        "rms": round(rms, 4),
        "clipped_ratio": round(clipping, 4),
    }

    if rms < 0.001:
        return None, {
            **stats,
            "silent": True,
        }

    return destination, stats


def _part(path, mime):
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(
                Path(path).read_bytes()
            ).decode("ascii"),
        }
    }


def _json_api(key, model, parts, max_tokens):
    if not key:
        raise AnalysisError("ไม่พบ GEMINI_API_KEY")

    if model not in ALLOWED_MODELS:
        raise AnalysisError(
            f"โมเดล {model} ไม่อยู่ในรายการที่อนุญาต"
        )

    url = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/{model}:generateContent"
    )

    body = {
        "contents": [{
            "role": "user",
            "parts": parts,
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": key,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=(15, 110),
            )

        except requests.RequestException as exc:
            if attempt == 2:
                raise AnalysisError(
                    f"เชื่อมต่อ Gemini ไม่สำเร็จ: {exc}"
                ) from exc

            time.sleep(2 * (attempt + 1))
            continue

        if response.status_code == 429:
            raise QuotaError(
                "Gemini ถึงโควตา API "
                "กรุณารอแล้วเริ่มคิวต่อ"
            )

        if (
            response.status_code
            in (500, 502, 503, 504)
            and attempt < 2
        ):
            time.sleep(2 * (attempt + 1))
            continue

        if response.status_code >= 400:
            try:
                message = response.json().get(
                    "error", {}
                ).get("message", "")
            except ValueError:
                message = response.text[:300]

            raise AnalysisError(
                f"Gemini HTTP {response.status_code}: "
                f"{str(message)[:350]}"
            )

        try:
            obj = response.json()

            chunks = obj["candidates"][0]["content"]["parts"]

            text = "".join(
                str(item.get("text", ""))
                for item in chunks
            ).strip()

            text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text,
            )

            data = json.loads(text)

            if not isinstance(data, dict):
                raise ValueError(
                    "ผลไม่ใช่ JSON object"
                )

            return data

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise AnalysisError(
                "ผล JSON จาก Gemini ไม่สมบูรณ์: "
                f"{exc}"
            ) from exc

    raise AnalysisError("Gemini ไม่ตอบกลับ")


def _number(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (ValueError, TypeError):
        return default


def _bool(value):
    return (
        value is True
        or (
            isinstance(value, str)
            and value.lower() == "true"
        )
    )


def _text(value, max_len=400):
    return str(value or "").strip()[:max_len]


def _evidence(value):
    text = _text(value)

    if (
        len(text) < 14
        or text.startswith((
            "ไม่พบ",
            "ไม่มี",
            "ปกติ",
            "ไม่ชัด",
            "ไม่แน่ใจ",
            "ยืนยันไม่ได้",
        ))
    ):
        return ""

    return text


def _issue(result, category, reason, sec=0):
    issues = result.get("issues")

    if not isinstance(issues, list):
        issues = []

    issues = [
        item
        for item in issues
        if isinstance(item, dict)
    ][:22]

    issues.append({
        "time_sec": round(
            max(0.0, _number(sec)),
            2,
        ),
        "category": category,
        "severity": "critical",
        "detail_th": _text(reason, 570),
    })

    result["issues"] = issues


def _fail(result, category, reason, sec=0):
    previous = _text(
        result.get("reason_th"),
        300,
    )

    result["decision"] = "FAIL"

    result["reason_th"] = (
        _text(reason, 520)
        + (
            " | " + previous
            if previous and previous != reason
            else ""
        )
    )[:850]

    _issue(result, category, reason, sec)


def _first_pass(
    key,
    model,
    sheets,
    details,
    wav,
    duration,
    stats,
    reference,
):
    instruction = (
        FIRST_PROMPT
        + f"\nคลิปยาว {duration:.2f} วินาที"
    )

    if reference:
        instruction += (
            "\nข้อมูลสินค้าที่ผู้ขายให้มา (อ้างอิง): "
            + reference[:900]
        )

    instruction += (
        "\nสถิติเสียง (ไม่ใช่ผลถอดคำพูด): "
        + json.dumps(stats)
    )

    parts = [{"text": instruction}]

    for path in sheets:
        parts.extend([
            {"text": "ภาพรวมหลายช่วงเวลา"},
            _part(path, "image/jpeg"),
        ])

    for sec, path in details:
        parts.extend([
            {
                "text": (
                    "ภาพเดี่ยวที่เวลาประมาณ "
                    f"{sec:.2f} วินาที"
                )
            },
            _part(path, "image/jpeg"),
        ])

    if wav is not None:
        parts.extend([
            {
                "text": (
                    "เสียงจริงทั้งคลิป "
                    "ตรวจคำพูดภาษาไทย"
                )
            },
            _part(wav, "audio/wav"),
        ])

    data = _json_api(
        key,
        model,
        parts,
        3800,
    )

    decision = _text(
        data.get("decision")
    ).upper()

    kind = _text(
        data.get("video_type")
    ).upper()

    if (
        decision not in ("PASS", "FAIL")
        or kind not in (
            "HUMAN", "HANDS", "PRODUCT", "MIXED"
        )
    ):
        raise AnalysisError(
            "Gemini ส่งคำตัดสินหรือประเภทคลิปไม่ถูกต้อง"
        )

    scale = data.get("scale_review")
    audio = data.get("audio_review")

    if (
        not isinstance(scale, dict)
        or (
            wav is not None
            and not isinstance(audio, dict)
        )
    ):
        raise AnalysisError(
            "Gemini ส่งข้อมูลตรวจสเกลหรือเสียงไม่ครบ"
        )

    if not isinstance(audio, dict):
        audio = {
            "status": "no_speech",
            "speech_present": False,
        }

    issues = data.get("issues", [])

    if not isinstance(issues, list):
        issues = []

    return {
        "decision": decision,
        "video_type": kind,
        "product_name_th": _text(
            data.get("product_name_th"),
            180,
        ),
        "reason_th": _text(
            data.get("reason_th"),
            850,
        ),
        "speech_summary_th": _text(
            data.get("speech_summary_th"),
            800,
        ),
        "issues": [
            item for item in issues
            if isinstance(item, dict)
        ][:22],
        "scale_review": scale,
        "audio_review": audio,
        "risk_flags": data.get("risk_flags", []),
        "scale_checks_complete": _bool(
            data.get("scale_checks_complete")
        ),
        "speech_checks_complete": _bool(
            data.get("speech_checks_complete")
        ),
        "audio_stats": stats,
        "scale_override": False,
        "audio_override": False,
        "scale_audit_override": False,
        "speech_audit_override": False,
    }


def _apply_first_safety(result):
    scale = result.get("scale_review", {})

    if (
        _text(scale.get("size_grade")).lower()
        == "wrong_scale"
        and _number(scale.get("confidence")) >= 0.82
        and not _bool(
            scale.get("miniature_disclosed")
        )
        and _evidence(
            scale.get("function_evidence_th")
            or scale.get("usage_evidence_th")
        )
        and (
            _evidence(
                scale.get("hand_evidence_th")
            )
            or _evidence(
                scale.get("environment_evidence_th")
            )
        )
    ):
        was_pass = result["decision"] == "PASS"

        _fail(
            result,
            "product_scale",
            "รอบแรกพบสินค้าผิดสเกลชัดเจน: "
            + _text(
                scale.get("reason_th")
                or scale.get("function_evidence_th")
            ),
        )

        result["scale_override"] = was_pass

    audio = result.get("audio_review", {})

    if (
        _number(audio.get("confidence")) >= 0.82
        and (
            _evidence(audio.get("evidence_th"))
            or _evidence(
                audio.get("broken_words_evidence_th")
            )
        )
        and (
            _bool(audio.get("severe_unintelligible"))
            or _bool(
                audio.get("key_information_unintelligible")
            )
            or _bool(audio.get("meaning_conflict"))
            or _number(
                audio.get("broken_word_count")
            ) >= 2
        )
    ):
        was_pass = result["decision"] == "PASS"

        _fail(
            result,
            "audio",
            "รอบแรกพบเสียงพูดเสียรุนแรง: "
            + _text(
                audio.get("evidence_th")
                or audio.get("broken_words_evidence_th")
            ),
            audio.get("time_sec"),
        )

        result["audio_override"] = was_pass

    return result


def _independent_audit(
    key,
    model,
    sheets,
    details,
    wav,
    duration,
    reference,
):
    prompt = (
        AUDIT_PROMPT
        + f"\nคลิปยาว {duration:.2f} วินาที"
    )

    if reference:
        prompt += (
            "\nข้อมูลสินค้าที่ผู้ขายให้มา (อ้างอิง): "
            + reference[:900]
        )

    parts = [{"text": prompt}]

    for path in sheets[:5]:
        parts.extend([
            {
                "text": (
                    "ภาพตามลำดับเวลาจากคลิป"
                )
            },
            _part(path, "image/jpeg"),
        ])

    for sec, path in details[:8]:
        parts.extend([
            {
                "text": (
                    "ดูขนาดสินค้าและสิ่งที่มือจับ "
                    f"ณ {sec:.2f}s"
                )
            },
            _part(path, "image/jpeg"),
        ])

    if wav is not None:
        parts.extend([
            {
                "text": (
                    "เสียงพูดจริงทั้งคลิป: "
                    "ต้องฟังเพื่อหาคำเพี้ยน"
                )
            },
            _part(wav, "audio/wav"),
        ])

    data = _json_api(
        key,
        model,
        parts,
        3100,
    )

    scale = data.get("scale")
    speech = data.get("speech")

    if (
        not isinstance(scale, dict)
        or not isinstance(speech, dict)
    ):
        raise AnalysisError(
            "ผลตรวจยืนยันไม่มี scale/speech"
        )

    if _text(
        scale.get("decision")
    ).upper() not in (
        "PASS", "FAIL", "UNCERTAIN"
    ):
        raise AnalysisError(
            "ผลตรวจสเกลยืนยันไม่ถูกต้อง"
        )

    if _text(
        speech.get("decision")
    ).upper() not in (
        "PASS", "FAIL", "UNCERTAIN"
    ):
        raise AnalysisError(
            "ผลตรวจเสียงยืนยันไม่ถูกต้อง"
        )

    return scale, speech


def _apply_audit(result, scale, speech, duration):
    result["scale_audit"] = scale
    result["speech_audit"] = speech
    result["audit_done"] = True

    confidence = _number(
        scale.get("confidence")
    )

    hand = _evidence(
        scale.get("hand_evidence_th")
    )
    environment = _evidence(
        scale.get("environment_evidence_th")
    )
    function = _evidence(
        scale.get("function_evidence_th")
    )

    confirmed_mini = (
        _bool(scale.get("intentional_mini"))
        and bool(
            _evidence(
                scale.get("mini_evidence_th")
            )
        )
    )

    confirmed_scale = (
        _text(
            scale.get("decision")
        ).upper() == "FAIL"
        and confidence >= 0.77
        and bool(function)
        and bool(hand or environment)
        and not confirmed_mini
    )

    time_a = _number(
        scale.get("time_a"),
        -1,
    )
    time_b = _number(
        scale.get("time_b"),
        -1,
    )

    morph = (
        _text(
            scale.get("decision")
        ).upper() == "FAIL"
        and _bool(
            scale.get("impossible_size_change")
        )
        and confidence >= 0.86
        and 0 <= time_a < time_b <= duration
        and time_b - time_a >= 0.3
        and bool(
            _evidence(
                scale.get("temporal_evidence_th")
            )
        )
    )

    if confirmed_scale or morph:
        detail = (
            _text(scale.get("reason_th"))
            or "; ".join(
                text
                for text in (
                    hand,
                    environment,
                    function,
                )
                if text
            )
        )

        reason = (
            "สเกลสินค้าผิดรุนแรงจากการตรวจยืนยัน: "
            + detail
        )

        result["scale_audit_override"] = (
            result["decision"] == "PASS"
        )

        _fail(
            result,
            "product_scale",
            reason,
            time_a if morph else 0,
        )

    score = _number(
        speech.get("confidence")
    )

    valid_events = []
    stamps = []

    events = speech.get("events", [])

    if not isinstance(events, list):
        events = []

    for event in events[:20]:
        if not isinstance(event, dict):
            continue

        sec = _number(
            event.get("time_sec"),
            -1,
        )

        fragment = _text(
            event.get("heard_fragment_th"),
            100,
        )

        why = _evidence(
            event.get("why_problem_th")
        )

        event_confidence = _number(
            event.get("confidence")
        )

        if (
            0 <= sec <= duration
            and fragment
            and why
            and event_confidence >= 0.75
            and all(
                abs(sec - old) >= 0.10
                for old in stamps
            )
        ):
            valid_events.append(
                (sec, fragment, why)
            )
            stamps.append(sec)

    start = _number(
        speech.get("continuous_start_sec"),
        -1,
    )
    end = _number(
        speech.get("continuous_end_sec"),
        -1,
    )

    continuous = (
        _bool(
            speech.get("continuous_gibberish")
        )
        and score >= 0.82
        and _number(
            speech.get("continuous_broken_word_count")
        ) >= 2
        and 0 <= start < end <= duration
        and end - start >= 0.35
        and bool(
            _evidence(
                speech.get("continuous_evidence_th")
            )
        )
    )

    broken_words = (
        len(valid_events) >= 2
        and score >= 0.77
    )

    spoken = _evidence(
        speech.get("spoken_claim_th")
    )

    visual = _evidence(
        speech.get(
            "visual_or_reference_evidence_th"
        )
    )

    mismatch = (
        _bool(
            speech.get("critical_claim_verified")
        )
        and score >= 0.87
        and bool(spoken and visual)
    )

    severe_audio = (
        _text(
            speech.get("decision")
        ).upper() == "FAIL"
        and _bool(
            speech.get("speech_present")
        )
        and (
            continuous
            or broken_words
            or mismatch
        )
    )

    if severe_audio:
        if broken_words:
            explanation = "; ".join(
                (
                    f"{sec:.1f}s {fragment}: "
                    f"{why}"
                )
                for sec, fragment, why
                in valid_events[:3]
            )

            at = valid_events[0][0]

            cause = (
                "เสียงพูดเพี้ยนอย่างน้อย 2 คำ: "
                + explanation
            )

        elif continuous:
            at = start

            cause = (
                "เสียงพูดมั่วหลายคำต่อเนื่อง: "
                + _text(
                    speech.get(
                        "continuous_evidence_th"
                    )
                )
            )

        else:
            at = 0.0

            cause = (
                "คำพูดขัดกับสินค้าที่ตรวจยืนยันได้: "
                + spoken
                + " / "
                + visual
            )

        result["speech_audit_override"] = (
            result["decision"] == "PASS"
        )

        _fail(
            result,
            "audio",
            "ตรวจเสียงยืนยันพบปัญหารุนแรง: "
            + cause,
            at,
        )

        audio = dict(
            result.get("audio_review") or {}
        )

        audio.update({
            "status": "critical",
            "speech_present": True,
            "confidence": score,
            "time_sec": at,
            "evidence_th": cause[:500],
            "severe_unintelligible": bool(
                continuous or broken_words
            ),
            "meaning_conflict": bool(mismatch),
        })

        result["audio_review"] = audio

    if _text(
        speech.get("speech_summary_th")
    ):
        result["speech_summary_th"] = _text(
            speech.get("speech_summary_th"),
            800,
        )

    if (
        result["decision"] == "PASS"
        and (
            _text(
                scale.get("decision")
            ).upper() == "UNCERTAIN"
            or _text(
                speech.get("decision")
            ).upper() == "UNCERTAIN"
        )
    ):
        result["audit_warning_th"] = (
            "มีข้อสงสัยแต่หลักฐานไม่พอ "
            "โปรดตรวจด้วยคนถ้าสำคัญ"
        )

        result["reason_th"] = (
            _text(
                result.get("reason_th"),
                620,
            )
            + " | ⚠️ สเกลหรือเสียงยังมีข้อสงสัย"
            " ควรตรวจด้วยคน"
        )[:850]

    return result


def _should_audit(
    result,
    meta,
    audio_stats,
    clip_path,
    reference="",
):
    """
    Decide whether a PASS clip needs independent verification.

    Do not rely only on Gemini's 'normal' label.
    A clean sample is audited even when Gemini reports no risk.
    """

    if result.get("decision") != "PASS":
        return False, "first_pass_fail"

    if AUDIT_MODE == "strict":
        return True, "strict_all_pass"

    reasons = []

    kind = _text(
        result.get("video_type")
    ).upper()

    scale = result.get("scale_review") or {}
    audio = result.get("audio_review") or {}

    if not isinstance(scale, dict):
        scale = {}
        reasons.append("missing_scale_review")

    if not isinstance(audio, dict):
        audio = {}
        reasons.append("missing_audio_review")

    # ตรวจซ้ำคลิปที่มีการโชว์สินค้ากับมือ
    # แม้รอบแรกบอกว่าสเกลปกติ
    if kind in {"HANDS", "MIXED"}:
        reasons.append("hand_scale_crosscheck")

    if not result.get("scale_checks_complete"):
        reasons.append("scale_not_verified")

    if (
        meta.get("has_audio")
        and not result.get("speech_checks_complete")
    ):
        reasons.append("speech_not_verified")

    grade = _text(
        scale.get("size_grade")
    ).lower()

    if grade in {
        "unknown",
        "suspicious",
        "wrong_scale",
        "",
    }:
        reasons.append("scale_risk")

    if _bool(
        scale.get("temporal_scale_change")
    ):
        reasons.append("scale_morph")

    if _text(
        scale.get("claim_consistency")
    ).lower() in {
        "mismatch",
        "uncertain",
        "",
    }:
        reasons.append("scale_claim_unclear")

    audio_status = _text(
        audio.get("status")
    ).lower()

    if meta.get("has_audio"):
        if audio_status in {
            "minor",
            "uncertain",
            "critical",
            "",
        }:
            reasons.append("audio_risk")

        if _number(
            audio.get("broken_word_count")
        ) >= 1:
            reasons.append("possible_broken_words")

        if _number(
            audio.get("unintelligible_span_count")
        ) >= 1:
            reasons.append("unintelligible_voice")

        relation = _text(
            audio.get("speech_visual_relation")
        ).lower()

        if relation in {
            "mismatch",
            "uncertain",
            "",
        }:
            reasons.append("speech_product_unclear")

        if _number(
            audio_stats.get("clipped_ratio")
        ) >= 0.03:
            reasons.append("heavy_audio_clipping")

    flags = result.get("risk_flags")

    if not isinstance(flags, list):
        reasons.append("risk_flags_missing")

    elif any(
        isinstance(flag, str) and flag.strip()
        for flag in flags
    ):
        reasons.append("model_risk_flag")

    if (
        reference
        and _text(
            scale.get("claim_consistency")
        ).lower() != "matched"
    ):
        reasons.append("reference_unverified")

    if reasons:
        unique_reasons = list(
            dict.fromkeys(reasons)
        )

        return True, ",".join(unique_reasons)

    if AUDIT_MODE == "fast":
        return False, "clean_fast"

    # ตรวจยืนยันเพิ่มเติมบางส่วน
    # แม้โมเดลแรกบอกว่าไม่มีความเสี่ยง
    # เลือกด้วย Hash ของข้อมูลคลิป
    # เพื่อให้ได้ผลการเลือกที่คงที่

    with open(clip_path, "rb") as handle:
        first_block = handle.read(
            256 * 1024
        )

    digest = hashlib.sha256(
        first_block
    ).digest()

    slot = int.from_bytes(
        digest[:4],
        "big",
    ) % 100

    if slot < CLEAN_PASS_AUDIT_PERCENT:
        return True, "clean_safety_sample"

    return False, "clean_not_sampled"


def inspect(
    path,
    key,
    model,
    mode,
    thumbnail,
    reference="",
):
    """
    Main entry point.

    Compatible with existing qc_store.py:
    inspect(path, key, model, mode, thumbnail)

    Optional:
    inspect(path, key, model, mode, thumbnail, reference)
    """

    started = time.perf_counter()

    path = Path(path)
    thumbnail = Path(thumbnail)

    metadata = probe(path)

    fps = FPS.get(mode, 2)

    with tempfile.TemporaryDirectory(
        prefix="qc_v34_"
    ) as temporary:
        folder = Path(temporary)

        frames = _frames(
            path,
            folder / "frames",
            fps,
        )

        thumbnail.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not thumbnail.is_file():
            with Image.open(frames[0]) as image:
                small = ImageOps.contain(
                    image.convert("RGB"),
                    (300, 300),
                )

                small.save(
                    thumbnail,
                    "JPEG",
                    quality=75,
                )

        sheets = _sheets(
            frames,
            fps,
            folder,
        )

        positions = (
            0.06,
            0.23,
            0.40,
            0.57,
            0.74,
            0.93,
        )

        selection = sorted(set(
            round(
                (len(frames) - 1) * pos
            )
            for pos in positions
        ))

        details = [
            (index / fps, frames[index])
            for index in selection
        ]

        if mode != "Fast":
            details += _motion_samples(
                path,
                frames,
                fps,
                metadata["duration"],
                folder,
            )

        wav, audio_stats = _audio(
            path,
            folder,
            metadata["has_audio"],
        )

        # ตรวจครั้งแรก: ภาพ + สเกล + เสียง
        result = _first_pass(
            key,
            model,
            sheets,
            details,
            wav,
            metadata["duration"],
            audio_stats,
            str(reference or ""),
        )

        # ถ้ารอบแรกมีหลักฐานรุนแรง
        # แต่ Gemini ให้ PASS ให้บังคับ FAIL
        result = _apply_first_safety(result)

        result["audit_done"] = False

        # เลือกตรวจยืนยันตามความเสี่ยง
        should_audit, audit_reason = _should_audit(
            result,
            metadata,
            audio_stats,
            path,
            str(reference or ""),
        )

        result["audit_trigger"] = audit_reason
        result["audit_mode"] = AUDIT_MODE

        if should_audit:
            # ตรวจยืนยันภาพและเสียงร่วมกัน
            # ไม่ส่งคำตัดสินรอบแรกให้ผู้ตรวจรอบสอง
            scale, speech = _independent_audit(
                key,
                model,
                sheets,
                details,
                wav,
                metadata["duration"],
                str(reference or ""),
            )

            result = _apply_audit(
                result,
                scale,
                speech,
                metadata["duration"],
            )

        elif result["decision"] == "PASS":
            result["audit_done"] = False

            result["audit_warning_th"] = (
                "ตรวจรอบเดียว; ยังมีโอกาสพลาด"
                "ข้อผิดพลาดที่ AI มองไม่เห็น"
            )

    result["duration"] = round(
        metadata["duration"],
        2,
    )

    result["elapsed"] = round(
        time.perf_counter() - started,
        2,
    )

    return result
