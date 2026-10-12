
"""V3.10 Field Ready Video QC — MP4 only, V3.9.1 drop-in.

Requires requests, Pillow, ffmpeg and ffprobe. No embedded API key.
Return PASS/FAIL/REVIEW with evidence. This is a screening tool,
not a guarantee of detecting every AI video artifact.
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
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageOps,
    ImageStat,
)

ENGINE_VERSION = "3.10-field-ready-mp4-20261012-r1"

MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
)

DEFAULT_MODEL = MODELS[0]
ALLOWED_MODELS = set(MODELS)

FPS = {
    "Fast": 2,
    "Balanced": 4,
    "Deep": 5,
}

MAX_DURATION = 35
FRAME_EDGE = 768

# balanced = ตรวจซ้ำคลิปเสี่ยง + สุ่มตรวจคลิป PASS
# strict = ตรวจยืนยันทุกคลิปที่รอบแรกยังไม่ FAIL
# fast = ตรวจซ้ำเฉพาะคลิปที่มีสัญญาณเสี่ยง
AUDIT_POLICY = "balanced"

CLEAN_PASS_SAMPLE_PERCENT = 20
HANDS_PASS_SAMPLE_PERCENT = 50

AUDIT_FPS = 5
AUDIT_SECONDS = 2.8


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


# ==========================================
# MODEL DISCOVERY
# ==========================================

def available_models(api_key: str) -> list[str]:
    if not api_key:
        return []

    found = set()
    token = None

    for _ in range(5):
        params = {"pageSize": 100}

        if token:
            params["pageToken"] = token

        try:
            r = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={
                    "x-goog-api-key": api_key
                },
                params=params,
                timeout=(8, 15),
            )

        except requests.RequestException as exc:
            raise AnalysisError(
                f"ตรวจสอบโมเดลไม่สำเร็จ: {exc}"
            ) from exc

        if r.status_code != 200:
            raise AnalysisError(
                f"ตรวจโมเดล HTTP {r.status_code}: "
                f"{r.text[:180]}"
            )

        try:
            data = r.json()

        except ValueError as exc:
            raise AnalysisError(
                "Google ส่งรายชื่อโมเดลผิดรูปแบบ"
            ) from exc

        for entry in data.get("models", []):
            name = str(
                entry.get("name", "")
            ).removeprefix("models/")

            methods = entry.get(
                "supportedGenerationMethods", []
            )

            if (
                name in MODELS
                and "generateContent" in methods
            ):
                found.add(name)

        token = data.get("nextPageToken")

        if not token:
            break

    return [
        m for m in MODELS if m in found
    ]


# ==========================================
# BASIC HELPERS
# ==========================================

def _s(value, n=450):
    return str(value or "").strip()[:n]


def _number(value, fallback=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else fallback

    except (TypeError, ValueError):
        return fallback


def _true(value):
    return (
        value is True
        or (
            isinstance(value, str)
            and value.lower() == "true"
        )
    )


def _state(section):
    if not isinstance(section, dict):
        return "REVIEW"

    result = _s(
        section.get("status"), 12
    ).upper()

    if result in {"PASS", "FAIL", "REVIEW"}:
        return result

    return "REVIEW"


def _proof(value, minimum=8):
    text = _s(value).lower()

    invalid = (
        "ไม่มี",
        "ไม่พบ",
        "ไม่แน่",
        "ไม่ชัด",
        "ไม่สามารถ",
        "ปกติทั้งหมด",
        "ตรวจแล้วปกติ",
        "unknown",
        "n/a",
        "none",
        "no evidence",
        "not visible",
    )

    return (
        len(text) >= minimum
        and not text.startswith(invalid)
    )


# ==========================================
# FFMPEG
# ==========================================

def _command(command, timeout=100):
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    except (
        subprocess.TimeoutExpired,
        OSError,
    ) as exc:
        raise AnalysisError(
            f"FFmpeg/ffprobe ผิดพลาด: {exc}"
        ) from exc

    if proc.returncode:
        raise AnalysisError(
            "ไม่สามารถอ่าน MP4: "
            + (proc.stderr or "")[-450:]
        )

    return proc.stdout


def _probe(video):
    data = _command([
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of", "json",
        str(video),
    ], 25)

    try:
        obj = json.loads(data)

        length = float(
            obj["format"]["duration"]
        )

        kinds = {
            x.get("codec_type")
            for x in obj["streams"]
        }

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise AnalysisError(
            "ไม่สามารถอ่านระยะเวลาหรือรูปแบบ MP4"
        ) from exc

    if (
        not math.isfinite(length)
        or not 0.5 <= length <= MAX_DURATION
    ):
        raise AnalysisError(
            f"รองรับวิดีโอ 0.5–{MAX_DURATION} วินาที"
        )

    if "video" not in kinds:
        raise AnalysisError(
            "ไม่มีภาพวิดีโอ"
        )

    return length, "audio" in kinds


# ==========================================
# FRAME EXTRACTION
# ==========================================

def _frames(
    video,
    folder,
    fps,
    start=0.0,
    seconds=None,
):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-threads", "2",
    ]

    if start > 0:
        cmd += [
            "-ss",
            f"{start:.3f}",
        ]

    cmd += [
        "-i",
        str(video),
    ]

    if seconds is not None:
        cmd += [
            "-t",
            f"{seconds:.3f}",
        ]

    cmd += [
        "-vf",
        (
            f"fps={fps},"
            f"scale={FRAME_EDGE}:{FRAME_EDGE}:"
            "force_original_aspect_ratio=decrease:"
            "force_divisible_by=2"
        ),
        "-q:v", "6",
        str(folder / "f_%03d.jpg"),
    ]

    _command(cmd, 130)

    files = sorted(
        folder.glob("f_*.jpg")
    )

    if not files:
        raise AnalysisError(
            "ดึงภาพจาก MP4 ไม่สำเร็จ"
        )

    return files


# ==========================================
# MOTION / FRAME SELECTION
# ==========================================

def _motion(frames):
    scores = [0.0] * len(frames)
    previous = None

    for i, file in enumerate(frames):
        with Image.open(file) as image:
            current = ImageOps.grayscale(
                image
            ).resize((64, 64))

        if previous is not None:
            difference = ImageChops.difference(
                previous,
                current,
            )

            scores[i] = ImageStat.Stat(
                difference
            ).mean[0]

        previous = current

    return scores


def _select_frames(
    frames,
    fps,
    scores,
    count=6,
):
    n = len(frames)

    indices = {
        round((n - 1) * p)
        for p in (
            0.08,
            0.31,
            0.55,
            0.78,
            0.96,
        )
    }

    ranking = sorted(
        range(1, n),
        key=lambda j: scores[j],
        reverse=True,
    )

    for i in ranking:
        if len(indices) >= count:
            break

        if all(
            abs(i - j) >= max(2, fps // 2)
            for j in indices
        ):
            indices.add(i)

    return [
        (i / fps, frames[i])
        for i in sorted(indices)
    ]


# ==========================================
# CONTACT SHEETS
# ==========================================

def _contact_sheets(
    frames,
    fps,
    folder,
    start=0.0,
    prefix="sheet",
):
    result = []

    for offset in range(
        0,
        len(frames),
        6,
    ):
        canvas = Image.new(
            "RGB",
            (960, 522),
            (25, 30, 38),
        )

        draw = ImageDraw.Draw(canvas)

        for j, path in enumerate(
            frames[offset:offset + 6]
        ):
            with Image.open(path) as source:
                thumb = ImageOps.contain(
                    source.convert("RGB"),
                    (306, 218),
                )

            x = (
                j % 3 * 320
                + (320 - thumb.width) // 2
            )

            y = (
                j // 3 * 261
                + 22
                + (218 - thumb.height) // 2
            )

            canvas.paste(
                thumb,
                (x, y),
            )

            draw.text(
                (
                    j % 3 * 320 + 6,
                    j // 3 * 261 + 5,
                ),
                f"{start + (offset + j) / fps:.2f}s",
                fill="white",
            )

        dst = folder / (
            f"{prefix}_{len(result):02d}.jpg"
        )

        canvas.save(
            dst,
            "JPEG",
            quality=76,
            optimize=True,
        )

        canvas.close()
        result.append(dst)

    return result


# ==========================================
# AUDIO EXTRACTION
# ==========================================

def _audio(
    video,
    folder,
    has_audio,
):
    if not has_audio:
        return None

    target = folder / "speech.wav"

    _command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(target),
    ], 70)

    with wave.open(
        str(target), "rb"
    ) as sound:
        pcm = sound.readframes(
            sound.getnframes()
        )

    sample = array("h")
    sample.frombytes(pcm)

    if not sample:
        return None

    stride = max(
        1,
        len(sample) // 6000,
    )

    values = sample[::stride]

    rms = math.sqrt(
        sum(x * x for x in values)
        / len(values)
    ) / 32768

    return (
        target
        if rms >= 0.0008
        else None
    )


# ==========================================
# INLINE MEDIA FOR GEMINI
# ==========================================

def _inline(path, mime):
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(
                Path(path).read_bytes()
            ).decode("ascii"),
        }
    }


# ==========================================
# STRUCTURED OUTPUT
# ==========================================

OUTPUT_EXAMPLE = r'''
{
 "product_type":"ชื่อสินค้าเฉพาะ",
 "video_type":"HANDS",
 "visible_product":"สิ่งที่เห็น",
 "spoken_product":"สิ่งที่พูดถึง หรือไม่มีเสียง",
 "product_use":"วิธีใช้งานสินค้าตามสิ่งที่เห็นหรือได้ยิน",
 "scale":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "observation":"เห็นขนาดเทียบกับอะไรและสัดส่วนเป็นอย่างไร",
   "advertised_use":"การใช้งานที่ได้ยินหรือเห็น",
   "miniature_risk":false,
   "miniature_intended":false,
   "events":[]
 },
 "audio":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "speech_present":true,
   "heard":"วลีที่ได้ยินจริง",
   "bad_words":[],
   "gibberish":false,
   "gibberish_proof":""
 },
 "match":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "spoken":"คำพูดอ้างสิ่งใด",
   "seen":"สิ่งที่แสดงช่วงเดียวกัน",
   "comparison":"ตรงหรือขัดแย้งอย่างไร"
 },
 "people":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "visible":true,
   "observation":"มือและคนปฏิสัมพันธ์กับสินค้าอย่างไร",
   "events":[]
 },
 "product_visual":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "observation":"รูปทรงสินค้าเปลี่ยนหรือคงที่อย่างไร",
   "events":[]
 },
 "reflection":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "applies":false,
   "angle_uncertain":false,
   "reflection_expected":false,
   "observation":"มุมสะท้อนที่เกี่ยวข้องหรือไม่เกี่ยวข้อง",
   "events":[]
 },
 "risk_flags":[],
 "risk_times":[],
 "summary":"สรุปสั้นตามหลักฐาน"
}
'''


# ==========================================
# FIRST GEMINI INSPECTION PROMPT
# ==========================================

FIRST_PROMPT = r'''
ผู้ตรวจคุณภาพวิดีโอโฆษณาสินค้า AI ความยาวสั้น
ไม่มี JSON สินค้า

จงเป็นผู้ตรวจจับปัญหาอย่างจริงจัง
ไม่ใช่ผู้ให้คะแนนความสวย

กฎหลัก:
ปัญหารุนแรงมีหลักฐาน = FAIL
เสี่ยงจริงแต่พิสูจน์ไม่ได้ = REVIEW
ภาพเสียงและการขายสัมพันธ์กัน
ไม่มีข้อผิดพลาดสำคัญ = PASS

อย่ากำหนดสัดส่วน FAIL
และห้ามสร้างหลักฐาน/คำพูด/ขนาดจริงที่ไม่มีให้

ขั้นแรก:
บอกว่าขายสินค้าอะไรจากวัตถุหลักและเสียง
เป็นสินค้าแบบไหน และใช้งานอย่างไร

แบ่ง video_type:
HUMAN, HANDS, PRODUCT, MIXED

ตรวจสินค้าทุกประเภทแม้เป็น HUMAN

==================================
SCALE
==================================

ดูมือ คน ชั้น โต๊ะ พื้น
ในระนาบใกล้กัน

เปรียบเทียบขนาดสินค้ากับหน้าที่ใช้จริง
ดูการเปลี่ยนสเกลเอง

ตรวจของจิ๋วที่ถูกใช้โฆษณาแทนของ
ขนาดใช้งานจริง

ของมินิที่โฆษณาตรงประเภทไม่ผิด
สินค้าพับหรือปรับลดระดับยังไม่ผิด

อย่าสมมติสูงเป็นเซนติเมตร
ถ้าคลิปไม่ได้พูดถึง

==================================
AUDIO
==================================

ฟังเสียงไทยจริงตั้งแต่ต้นจนจบ
ระบุวลีที่ได้ยิน

ผิดชัดตั้งแต่ 2 คำคนละช่วง
หรือพูดมั่วต่อเนื่อง = FAIL
พร้อมเวลาจริง

ผิดเล็กน้อย 1 คำ สำเนียง
ชื่อยี่ห้อ ไม่มีเสียงพูด
ไม่ผิดโดยอัตโนมัติ

==================================
MATCH
==================================

เทียบคำพูดกับสิ่งที่ปรากฏ
ในช่วงเดียวกัน

ตรวจ:
- สินค้าชนิดเดียวกันหรือไม่
- สี
- จำนวน
- วิธีใช้
- คุณสมบัติที่มองเห็นจริง

ภาพสวยแต่พูดขายของอีกชนิด = FAIL
ถ้ามีหลักฐานชัด

ถ้าข้อเท็จจริงมองไม่เห็น
ให้ REVIEW เฉพาะเมื่อสำคัญ

==================================
PEOPLE
==================================

ตรวจมือ นิ้ว การสัมผัสสินค้า
มือทะลุ มือหาย แขนเพิ่ม
ผิดกายวิภาครุนแรง

มือออกนอกขอบภาพตามมุมกล้อง
ไม่ถือว่ามือหายผิดธรรมชาติ

==================================
PRODUCT VISUAL
==================================

ตรวจสินค้าหลัก รูปทรง ฐาน ล้อ
ขา จำนวนชั้น ขอบและการใช้งาน

ดูว่าของหาย เปลี่ยนชนิด บิดละลาย
เปลี่ยนทรงอย่างเป็นไปไม่ได้หรือไม่

==================================
REFLECTION
==================================

ถ้าสินค้าคือกระจกเงาหลัก
ตรวจมุมที่มือควรสะท้อน
และเงาต่อเนื่อง

ถ้าอ้างมือหายเกิน 2 วินาที
ให้ระบุหลายช่วงเวลาที่เห็นจริง

หากมุมไม่ชัด REVIEW

==================================
กฎการรายงาน
==================================

การตรวจต้องอธิบายข้อสังเกตจริง
ไม่ใช่คำว่า "ปกติดี" เฉย ๆ

หากด้านนั้นไม่เกี่ยวข้อง เช่น
ไม่มีคน ไม่มีเสียง ไม่ใช่กระจก
ให้ระบุอย่างตรงไปตรงมา

ถ้าพบปัญหา เติม events ของ
scale/people/product_visual
ตามรูปแบบ:

[{"t":2.0,"seen":"ภาพที่เห็น","why":"ข้อขัดแย้ง"}]

ถ้ามีคำผิด เติม audio.bad_words:

[{"t":2.0,"heard":"คำจริงที่ได้ยิน","why":"เหตุผล"}]

ถ้ากระจกผิด เติม reflection.events:

[{"type":"missing_hand_reflection",
  "from":1.0,"to":3.2,
  "seen":"ภาพที่เห็น",
  "geometry":"มุมสะท้อน",
  "samples":[1.0,2.0,3.2]}]

ข้อมูล events/bad_words ต้องเป็น []
เมื่อไม่พบอะไร

ห้ามคัดลอกเหตุการณ์ตัวอย่าง
เป็นหลักฐานจริง

สถานะ FAIL ต้องกรอก confidence
ให้สะท้อนความมั่นใจที่มีหลักฐานจริง

confidence เป็นความเห็นของโมเดล
ไม่ใช่หลักฐานยืนยัน

ตอบ JSON object เท่านั้น ตามรูปแบบนี้:
''' + OUTPUT_EXAMPLE


# ==========================================
# INDEPENDENT SECOND INSPECTION
# ==========================================

SECOND_PROMPT = r'''
คุณคือผู้ตรวจยืนยันอิสระ
ตรวจค้นหาข้อผิดพลาดที่ผู้ตรวจทั่วไปพลาด

ไม่ทราบผลรอบแรก
ตรวจเฉพาะหัวข้อที่ระบุตอนท้าย
พร้อมหลักฐานและเวลา

ถ้าข้อมูลภาพเป็นเพียงบางช่วง
อย่าอ้างว่าได้ตรวจทั้งคลิป

ห้ามแต่งเหตุการณ์ คำพูด หรือขนาด

มือออกนอกขอบภาพไม่ใช่มือหาย

แป้นบาส รถเข็น ชั้นวางขนาดเล็ก
อาจเป็นสินค้ามินิจริง

ถ้าขาดข้ออ้างขนาดใช้งานจริง
อย่าสรุปว่าโฆษณาหลอก

เสียงเพี้ยนต้องมีอย่างน้อย 2 คำ
ที่ได้ยินจริง หรือพูดมั่วต่อเนื่อง

กระจก:
ดูเรขาคณิต มุม สิ่งบัง
และหลักฐานหลายเฟรม

ตอบ JSON object ตาม schema ด้านล่าง
ทั้ง 6 หัวข้อ

หัวข้ออื่นที่ไม่ตรวจ
ให้ checked=false และ events=[]

''' + OUTPUT_EXAMPLE


# ==========================================
# BUILD GEMINI REQUEST
# ==========================================

def _parts(
    prompt,
    duration,
    sheets,
    details,
    wav,
):
    parts = [{
        "text": (
            prompt
            + f"\nความยาววิดีโอ {duration:.2f} วินาที"
        )
    }]

    for picture in sheets:
        parts += [
            {
                "text": (
                    "ภาพตามลำดับเวลา "
                    "ป้ายเวลาแสดงบนแต่ละช่อง"
                )
            },
            _inline(
                picture,
                "image/jpeg",
            ),
        ]

    for sec, picture in details:
        parts += [
            {
                "text": (
                    f"ภาพเดี่ยวขยาย ณ {sec:.2f}s"
                )
            },
            _inline(
                picture,
                "image/jpeg",
            ),
        ]

    if wav:
        parts += [
            {
                "text": (
                    "เสียงจริงของวิดีโอ "
                    "ต้องฟังคำไทยเอง"
                )
            },
            _inline(
                wav,
                "audio/wav",
            ),
        ]

    else:
        parts.append({
            "text": (
                "ไม่มีแทร็กเสียงที่ได้ยินชัด "
                "ห้ามสร้างคำผิด"
            )
        })

    return parts


# ==========================================
# GEMINI API
# ==========================================

def _gemini(
    api_key,
    selected_model,
    parts,
    max_tokens,
):
    if not api_key:
        raise AnalysisError(
            "ไม่ได้ตั้ง GEMINI_API_KEY"
        )

    model = (
        selected_model
        if selected_model in MODELS
        else DEFAULT_MODEL
    )

    choices = [model]
    checked = 0

    while checked < len(choices):
        model = choices[checked]
        checked += 1

        payload = {
            "contents": [{
                "role": "user",
                "parts": parts,
            }],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "maxOutputTokens": max_tokens,
            },
        }

        endpoint = (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models/"
            + model
            + ":generateContent"
        )

        for attempt in range(2):
            try:
                r = requests.post(
                    endpoint,
                    headers={
                        "x-goog-api-key": api_key
                    },
                    json=payload,
                    timeout=(12, 105),
                )

            except requests.RequestException as exc:
                if attempt == 0:
                    time.sleep(1)
                    continue

                raise AnalysisError(
                    f"เชื่อมต่อ Gemini ไม่ได้: {exc}"
                ) from exc

            if r.status_code == 429:
                raise QuotaError(
                    "Gemini API ถึงโควตา 429 "
                    "กรุณารอสักครู่"
                )

            if (
                r.status_code in (
                    500,
                    502,
                    503,
                    504,
                )
                and attempt == 0
            ):
                time.sleep(1)
                continue

            if r.status_code >= 400:
                try:
                    msg = r.json().get(
                        "error", {}
                    ).get("message", "")

                except ValueError:
                    msg = r.text[:400]

                if r.status_code == 404:
                    try:
                        known_models = available_models(
                            api_key
                        )

                        for known in known_models:
                            if known not in choices:
                                choices.append(known)

                    except AnalysisError:
                        pass

                    if checked < len(choices):
                        break

                raise AnalysisError(
                    f"Gemini HTTP {r.status_code} "
                    f"({model}): {_s(msg, 350)}"
                )

            try:
                choices_data = r.json()[
                    "candidates"
                ]

                text = "".join(
                    p.get("text", "")
                    for p in choices_data[0][
                        "content"
                    ]["parts"]
                )

                text = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    text.strip(),
                )

                obj = json.loads(text)

                required = (
                    "scale",
                    "audio",
                    "match",
                    "people",
                    "product_visual",
                    "reflection",
                )

                if (
                    not isinstance(obj, dict)
                    or any(
                        not isinstance(
                            obj.get(section),
                            dict,
                        )
                        for section in required
                    )
                ):
                    raise ValueError(
                        "ข้อมูลการตรวจไม่ครบ 6 หัวข้อ"
                    )

                return obj

            except (
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                raise AnalysisError(
                    f"Gemini ส่งผลไม่สมบูรณ์: {exc}"
                ) from exc

    raise AnalysisError(
        "ไม่พบโมเดล Gemini ที่สามารถใช้ได้"
    )


# ==========================================
# IMAGE EVIDENCE
# ==========================================

def _events(block, duration):
    records = block.get(
        "events", []
    )

    if not isinstance(records, list):
        return []

    out = []

    for item in records[:12]:
        if not isinstance(item, dict):
            continue

        t = _number(
            item.get("t"), -1
        )

        seen = _s(
            item.get("seen"), 250
        )

        why = _s(
            item.get("why"), 300
        )

        if (
            0 <= t <= duration
            and _proof(seen)
            and _proof(why)
        ):
            out.append((
                t, seen, why
            ))

    return out


# ==========================================
# THAI SPEECH EVIDENCE
# ==========================================

def _bad_words(block, duration):
    words = block.get(
        "bad_words", []
    )

    if not isinstance(words, list):
        return []

    out = []

    for item in words[:14]:
        if not isinstance(item, dict):
            continue

        t = _number(
            item.get("t"), -1
        )

        heard = _s(
            item.get("heard"), 70
        )

        why = _s(
            item.get("why"), 170
        )

        distinct = all(
            abs(t - other[0]) >= 0.18
            for other in out
        )

        if (
            0 <= t <= duration
            and heard
            and _proof(why)
            and distinct
        ):
            out.append((
                t, heard, why
            ))

    return out


# ==========================================
# MIRROR EVIDENCE
# ==========================================

def _mirror_events(block, duration):
    raw = block.get(
        "events", []
    )

    if not isinstance(raw, list):
        return []

    valid_types = {
        "missing_hand_reflection",
        "severe_warp",
        "impossible_reflection",
        "unrelated_motion",
    }

    out = []

    for item in raw[:10]:
        if (
            not isinstance(item, dict)
            or item.get("type") not in valid_types
        ):
            continue

        a = _number(
            item.get("from"), -1
        )

        b = _number(
            item.get("to"), -1
        )

        if not 0 <= a < b <= duration:
            continue

        seen = _s(
            item.get("seen"), 250
        )

        geo = _s(
            item.get("geometry"), 250
        )

        if not (
            _proof(seen)
            and _proof(geo)
        ):
            continue

        samples = item.get(
            "samples", []
        )

        if not isinstance(samples, list):
            samples = []

        valid = sorted({
            _number(x, -1)
            for x in samples
            if (
                a - 0.5
                <= _number(x, -1)
                <= b + 0.5
            )
        })

        sustained = (
            len(valid) >= 3
            and valid[-1] - valid[0] >= 1.8
        )

        out.append((
            a,
            b,
            item["type"],
            seen,
            geo,
            sustained,
        ))

    return out


def _reflection_applies(data):
    text = _s(
        data.get("product_type"),
        200,
    )

    return (
        _true(
            data["reflection"].get("applies")
        )
        or bool(re.search(
            r"กระจกเงา|กระจกส่อง|"
            r"กระจกแต่งหน้า|\bmirror\b",
            text,
            re.I,
        ))
    )


# ==========================================
# CHECK EVIDENCE COVERAGE
# ==========================================

def _coverage(data, audible, scope):
    missing = set()

    if (
        not _proof(
            data.get("product_type"), 4
        )
        or not _proof(
            data.get("visible_product"), 4
        )
    ):
        missing.update({
            "scale",
            "product_visual",
        })

    for area in scope:
        part = data[area]

        if (
            area in {"audio", "match"}
            and not audible
        ):
            continue

        if (
            area == "reflection"
            and not _reflection_applies(data)
        ):
            continue

        if not _true(
            part.get("checked")
        ):
            missing.add(area)

    if (
        "scale" in scope
        and _state(data["scale"]) == "PASS"
    ):
        if not _proof(
            data["scale"].get("observation")
        ):
            missing.add("scale")

    if (
        "audio" in scope
        and audible
        and _true(
            data["audio"].get("speech_present")
        )
    ):
        if not _proof(
            data["audio"].get("heard"), 4
        ):
            missing.add("audio")

    if (
        "match" in scope
        and audible
        and _true(
            data["audio"].get("speech_present")
        )
    ):
        if not _proof(
            data["match"].get("comparison")
        ):
            missing.add("match")

    if (
        "people" in scope
        and _true(
            data["people"].get("visible")
        )
    ):
        if not _proof(
            data["people"].get("observation")
        ):
            missing.add("people")

    if (
        "product_visual" in scope
        and not _proof(
            data["product_visual"].get("observation")
        )
    ):
        missing.add("product_visual")

    if (
        "reflection" in scope
        and _reflection_applies(data)
    ):
        if not _proof(
            data["reflection"].get("observation")
        ):
            missing.add("reflection")

    return missing & set(scope)


# ==========================================
# DECISION ENGINE
# ==========================================

def _judge(
    data,
    duration,
    audible,
    scope=None,
):
    if scope is None:
        scope = {
            "scale",
            "audio",
            "match",
            "people",
            "product_visual",
            "reflection",
        }

    else:
        scope = set(scope)

    failures = []
    concerns = set()

    missing = _coverage(
        data,
        audible,
        scope,
    )

    for area in scope:
        b = data[area]

        state = _state(b)

        conf = _number(
            b.get("confidence")
        )

        # Product scale, people, shape

        if area in {
            "scale",
            "people",
            "product_visual",
        }:
            ev = _events(
                b, duration
            )

            threshold = (
                0.71
                if area == "scale"
                else 0.76
            )

            confident = (
                conf >= threshold
                and bool(ev)
            )

            if area == "scale":
                has_context = (
                    _proof(
                        b.get("observation")
                    )
                    or len(ev) > 1
                )

                confident = (
                    confident
                    and has_context
                    and not _true(
                        b.get("miniature_intended")
                    )
                )

            if (
                state == "FAIL"
                and confident
            ):
                failures.append((
                    area,
                    ev[0][0],
                    (
                        f"{ev[0][1]} — "
                        f"{ev[0][2]}"
                    ),
                ))

            elif (
                state != "PASS"
                or ev
            ):
                concerns.add(area)

            if (
                area == "scale"
                and _true(
                    b.get("miniature_risk")
                )
                and not _true(
                    b.get("miniature_intended")
                )
            ):
                concerns.add("scale")

        # Audio

        elif area == "audio" and audible:
            ev = _bad_words(
                b, duration
            )

            if (
                state == "FAIL"
                and conf >= 0.72
                and len(ev) >= 2
            ):
                msg = "; ".join(
                    (
                        f"{t:.1f}s {word} "
                        f"({why})"
                    )
                    for t, word, why
                    in ev[:3]
                )

                failures.append((
                    "audio",
                    ev[0][0],
                    "เสียงเพี้ยน >=2 คำ: " + msg,
                ))

            elif (
                state == "FAIL"
                and conf >= 0.80
                and _true(
                    b.get("gibberish")
                )
                and _proof(
                    b.get("gibberish_proof")
                )
            ):
                failures.append((
                    "audio",
                    0,
                    (
                        "พูดมั่วต่อเนื่อง: "
                        + _s(
                            b.get("gibberish_proof")
                        )
                    ),
                ))

            elif (
                state != "PASS"
                or len(ev) >= 2
            ):
                concerns.add("audio")

        # Speech vs product

        elif area == "match" and audible:
            confirmed = (
                state == "FAIL"
                and conf >= 0.75
                and _proof(
                    b.get("spoken")
                )
                and _proof(
                    b.get("seen")
                )
                and _proof(
                    b.get("comparison")
                )
            )

            if confirmed:
                failures.append((
                    "match",
                    0,
                    (
                        "คำพูดขัดกับภาพ: "
                        + _s(
                            b.get("comparison")
                        )
                    ),
                ))

            elif state != "PASS":
                concerns.add("match")

        # Mirror

        elif (
            area == "reflection"
            and _reflection_applies(data)
        ):
            events = _mirror_events(
                b,
                duration,
            )

            confirmed = []

            for event in events:
                (
                    t1,
                    t2,
                    kind,
                    seen,
                    geo,
                    samples_ok,
                ) = event

                if (
                    not samples_ok
                    or _true(
                        b.get("angle_uncertain")
                    )
                ):
                    continue

                if (
                    kind == "missing_hand_reflection"
                    and (
                        t2 - t1 <= 2.0
                        or not _true(
                            b.get(
                                "reflection_expected"
                            )
                        )
                    )
                ):
                    continue

                confirmed.append(event)

            if (
                state == "FAIL"
                and conf >= 0.82
                and confirmed
            ):
                e = confirmed[0]

                failures.append((
                    "reflection",
                    e[0],
                    (
                        "เงากระจกผิด: "
                        + e[3]
                        + "; "
                        + e[4]
                    ),
                ))

            elif (
                state != "PASS"
                or events
            ):
                concerns.add("reflection")

    # Additional risk flags

    flags = data.get(
        "risk_flags", []
    )

    if isinstance(flags, list):
        for item in flags[:8]:
            tag = _s(item).lower()

            if not _proof(tag):
                continue

            if any(
                word in tag
                for word in (
                    "scale",
                    "สเกล",
                    "ขนาด",
                    "mini",
                )
            ):
                concerns.add("scale")

            elif any(
                word in tag
                for word in (
                    "เสียง",
                    "พูด",
                    "speech",
                    "audio",
                )
            ):
                if audible:
                    concerns.add("audio")

            elif any(
                word in tag
                for word in (
                    "hand",
                    "นิ้ว",
                    "มือ",
                )
            ):
                concerns.add("people")

            elif any(
                word in tag
                for word in (
                    "mirror",
                    "กระจก",
                    "เงา",
                )
            ):
                concerns.add("reflection")

            elif any(
                word in tag
                for word in (
                    "mismatch",
                    "ไม่ตรง",
                )
            ):
                if audible:
                    concerns.add("match")

            else:
                concerns.add("product_visual")

    return (
        failures,
        (concerns | missing) & scope,
        missing,
    )


# ==========================================
# SMART AUDIT SELECTION
# ==========================================

def _sampling(video, data):
    if AUDIT_POLICY == "strict":
        return True

    if AUDIT_POLICY == "fast":
        return False

    with open(video, "rb") as handle:
        seed = int.from_bytes(
            hashlib.sha256(
                handle.read(65536)
            ).digest()[:4],
            "big",
        ) % 100

    interactive = (
        _true(
            data["people"].get("visible")
        )
        or _reflection_applies(data)
        or _true(
            data["scale"].get("miniature_risk")
        )
    )

    percentage = (
        HANDS_PASS_SAMPLE_PERCENT
        if interactive
        else CLEAN_PASS_SAMPLE_PERCENT
    )

    return seed < percentage


# ==========================================
# FIND RISK TIMESTAMP
# ==========================================

def _focus_time(
    data,
    areas,
    duration,
):
    points = []

    for a in areas:
        if a in {
            "scale",
            "people",
            "product_visual",
        }:
            points += [
                t
                for t, *_ in _events(
                    data[a], duration
                )
            ]

        elif a == "audio":
            points += [
                t
                for t, *_ in _bad_words(
                    data[a], duration
                )
            ]

        elif a == "reflection":
            points += [
                (start + end) / 2
                for start, end, *_
                in _mirror_events(
                    data[a], duration
                )
            ]

    risk_times = data.get(
        "risk_times", []
    )

    if isinstance(risk_times, list):
        for t in risk_times:
            value = _number(t, -1)

            if 0 <= value <= duration:
                points.append(value)

    if points:
        return min(points)

    return duration / 2


# ==========================================
# SELECTIVE AUDIT MEDIA
# ==========================================

def _audit_media(
    video,
    folder,
    duration,
    focus,
    original,
    sheets,
    details,
    wav,
):
    need_img = bool(
        focus & {
            "scale",
            "match",
            "people",
            "product_visual",
            "reflection",
        }
    )

    need_audio = bool(
        focus & {
            "audio",
            "match",
        }
    )

    if not need_img:
        return (
            [],
            [],
            wav if need_audio else None,
            None,
        )

    dense = bool(
        focus & {
            "scale",
            "people",
            "product_visual",
            "reflection",
        }
    )

    if dense:
        center = _focus_time(
            original,
            focus,
            duration,
        )

        length = min(
            AUDIT_SECONDS,
            duration,
        )

        begin = max(
            0,
            min(
                duration - length,
                center - length / 2,
            ),
        )

        snap = _frames(
            video,
            folder / "focused",
            AUDIT_FPS,
            begin,
            length,
        )

        frames2 = _contact_sheets(
            snap,
            AUDIT_FPS,
            folder,
            begin,
            "focused",
        )

        indices = sorted({
            0,
            len(snap) // 2,
            len(snap) - 1,
        })

        detailed = [
            (
                begin + i / AUDIT_FPS,
                snap[i],
            )
            for i in indices
        ]

        selected_sheets = frames2[:3]

        if "match" in focus:
            selected_sheets += [
                sheets[0],
                sheets[-1],
            ]

        return (
            selected_sheets,
            detailed,
            wav if need_audio else None,
            [
                round(begin, 2),
                round(begin + length, 2),
            ],
        )

    step = max(
        1,
        len(sheets) // 3,
    )

    return (
        sheets[::step][:3],
        details[:2],
        wav if need_audio else None,
        None,
    )


# ==========================================
# MAIN INSPECTION
# ==========================================

def inspect(
    path,
    key,
    model,
    mode,
    thumbnail,
    reference="",
):
    """
    Compatible with V3.9.1 ActiveQueue.

    MP4-only:
    reference is intentionally ignored.
    """

    begun = time.perf_counter()

    video = Path(path)
    thumb = Path(thumbnail)

    duration, has_audio = _probe(video)

    fps = FPS.get(
        mode,
        FPS["Balanced"],
    )

    with tempfile.TemporaryDirectory(
        prefix="qc310_"
    ) as tmp:
        folder = Path(tmp)

        # Full timeline

        full_frames = _frames(
            video,
            folder / "frames",
            fps,
        )

        motion = _motion(
            full_frames
        )

        sheets = _contact_sheets(
            full_frames,
            fps,
            folder,
        )

        details = _select_frames(
            full_frames,
            fps,
            motion,
            (
                8
                if mode == "Deep"
                else 6
            ),
        )

        # Thumbnail

        thumb.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not thumb.exists():
            with Image.open(
                full_frames[0]
            ) as source:
                small = ImageOps.contain(
                    source.convert("RGB"),
                    (320, 320),
                )

                small.save(
                    thumb,
                    "JPEG",
                    quality=72,
                )

        # Actual audio

        wav = _audio(
            video,
            folder,
            has_audio,
        )

        prepared = time.perf_counter()

        # First Gemini inspection

        first = _gemini(
            key,
            model,
            _parts(
                FIRST_PROMPT,
                duration,
                sheets,
                details,
                wav,
            ),
            2700,
        )

        first_end = time.perf_counter()

        (
            failures,
            worries,
            coverage,
        ) = _judge(
            first,
            duration,
            wav is not None,
        )

        final_fails = list(failures)
        pending = set(worries)

        second = None
        focus = set()
        span = None

        # Second independent inspection

        if not failures:
            focus = set(worries)

            if (
                not focus
                and _sampling(video, first)
            ):
                focus = {
                    "scale",
                    "people",
                    "product_visual",
                }

                if wav is not None:
                    focus |= {
                        "audio",
                        "match",
                    }

                if _reflection_applies(first):
                    focus.add("reflection")

            if focus:
                (
                    images,
                    stills,
                    audio2,
                    span,
                ) = _audit_media(
                    video,
                    folder,
                    duration,
                    focus,
                    first,
                    sheets,
                    details,
                    wav,
                )

                note = (
                    "\nตรวจเฉพาะหัวข้อ: "
                    + ", ".join(sorted(focus))
                )

                if span:
                    note += (
                        "\nภาพเพิ่มเติมเป็นช่วง "
                        f"{span[0]} ถึง {span[1]} วินาที "
                        "ไม่ใช่ทั้งวิดีโอ"
                    )

                second = _gemini(
                    key,
                    model,
                    _parts(
                        SECOND_PROMPT + note,
                        duration,
                        images,
                        stills,
                        audio2,
                    ),
                    2400,
                )

                (
                    secondary_fails,
                    secondary_risks,
                    _,
                ) = _judge(
                    second,
                    duration,
                    wav is not None,
                    scope=focus,
                )

                final_fails = secondary_fails
                pending = set(secondary_risks)

                # Preserve evidence from first inspection

                for area in worries & focus:
                    if area == "reflection":
                        events = _mirror_events(
                            first[area],
                            duration,
                        )

                    elif area == "audio":
                        events = _bad_words(
                            first[area],
                            duration,
                        )

                    elif area in {
                        "scale",
                        "people",
                        "product_visual",
                    }:
                        events = _events(
                            first[area],
                            duration,
                        )

                    else:
                        events = []

                    already_failed = any(
                        f[0] == area
                        for f in final_fails
                    )

                    if (
                        events
                        and not already_failed
                    ):
                        pending.add(area)

                    if _state(
                        second[area]
                    ) == "REVIEW":
                        pending.add(area)

        # Final decision

        if final_fails:
            outcome = "FAIL"

        elif pending:
            outcome = "REVIEW"

        else:
            outcome = "PASS"

        if final_fails:
            reason = "; ".join(
                e[2]
                for e in final_fails[:3]
            )

        elif pending:
            reason = (
                "รอตรวจด้วยคน: "
                "พบข้อสงสัย/หลักฐานยังไม่พอด้าน "
                + ", ".join(sorted(pending))
            )

        else:
            reason = (
                _s(
                    first.get("summary"), 750
                )
                or (
                    "ไม่พบข้อผิดพลาดสำคัญ"
                    "จากวิดีโอ"
                )
            )

        # Evidence for UI

        issues = [
            {
                "category": k,
                "time_sec": round(t, 2),
                "detail_th": why,
                "severity": "critical",
            }
            for k, t, why
            in final_fails
        ]

        if outcome == "REVIEW":
            issues.append({
                "category": "human_review",
                "time_sec": 0,
                "detail_th": reason,
                "severity": "warning",
            })

        chosen = (
            second
            if (
                second is not None
                and final_fails
            )
            else first
        )

        ended = time.perf_counter()

        return {
            "decision": outcome,

            "reason_th": reason[:850],

            "product_type_th": _s(
                chosen.get("product_type"),
                180,
            ),

            "product_name_th": _s(
                chosen.get("visible_product"),
                180,
            ),

            "spoken_product_th": _s(
                chosen.get("spoken_product"),
                180,
            ),

            "speech_summary_th": _s(
                first["audio"].get("heard"),
                600,
            ),

            "video_type": (
                _s(
                    first.get("video_type"),
                    30,
                )
                or "PRODUCT"
            ),

            "scale_review": chosen["scale"],
            "audio_review": chosen["audio"],
            "match_review": chosen["match"],
            "people_review": chosen["people"],

            "visual_review": chosen[
                "product_visual"
            ],

            "mirror_review": chosen[
                "reflection"
            ],

            "scale_audit": (
                second.get("scale")
                if second else None
            ),

            "speech_audit": (
                second.get("audio")
                if second else None
            ),

            "match_audit": (
                second.get("match")
                if second else None
            ),

            "people_audit": (
                second.get("people")
                if second else None
            ),

            "visual_audit": (
                second.get("product_visual")
                if second else None
            ),

            "mirror_audit": (
                second.get("reflection")
                if second else None
            ),

            "audit_done": (
                second is not None
            ),

            "audit_trigger": (
                ", ".join(sorted(focus))
                if focus
                else "none"
            ),

            "audit_dense_window_sec": span,

            "evidence_coverage_missing": sorted(
                coverage
            ),

            "issues": issues,

            "review_required": (
                outcome == "REVIEW"
            ),

            "product_reference_used": False,

            "duration": round(duration, 2),

            "elapsed": round(
                ended - begun, 2
            ),

            "engine": ENGINE_VERSION,

            "timings": {
                "prepare_seconds": round(
                    prepared - begun, 2
                ),

                "first_api_seconds": round(
                    first_end - prepared, 2
                ),

                "audit_api_seconds": (
                    round(
                        ended - first_end,
                        2,
                    )
                    if second else 0.0
                ),
            },
        }
