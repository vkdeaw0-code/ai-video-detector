
"""V3.9.2 Precision & Speed Engine - MP4 Only.

Drop-in replacement for V3.9.1 qc_core.py.

Evidence-based first pass:
  Balanced = 4 FPS
  Fast     = 2 FPS
  Deep     = 5 FPS

Optional targeted 5 FPS recheck.
Results: PASS, FAIL, REVIEW.

Other project files remain unchanged.
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
from PIL import Image, ImageDraw, ImageOps


ENGINE_VERSION = "3.9.2-precision-speed-mp4-20261012-r1"

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

FRAME_EDGE = 896
MAX_SECONDS = 35

AUDIT_FPS = 5
AUDIT_WINDOW_SEC = 2.8


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


# ==========================================
# CHECK GEMINI MODEL ACCESS
# ==========================================

def available_models(api_key: str) -> list[str]:
    if not api_key:
        return []

    found = set()
    token = None

    try:
        for _ in range(6):
            params = {
                "pageSize": 100
            }

            if token:
                params["pageToken"] = token

            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={
                    "x-goog-api-key": api_key
                },
                params=params,
                timeout=(8, 16),
            )

            if response.status_code >= 400:
                raise AnalysisError(
                    f"ตรวจรายชื่อโมเดล HTTP "
                    f"{response.status_code}: "
                    + response.text[:240]
                )

            data = response.json()

            for item in data.get("models", []):
                model = str(
                    item.get("name", "")
                ).removeprefix("models/")

                methods = item.get(
                    "supportedGenerationMethods",
                    [],
                )

                if (
                    model in MODELS
                    and "generateContent" in methods
                ):
                    found.add(model)

            token = data.get("nextPageToken")

            if not token:
                break

    except (
        requests.RequestException,
        ValueError,
    ) as exc:
        raise AnalysisError(
            f"ตรวจสอบโมเดลไม่ได้: {exc}"
        ) from exc

    return [
        name for name in MODELS
        if name in found
    ]


# ==========================================
# HELPERS
# ==========================================

def _text(x, limit=350):
    return str(x or "").strip()[:limit]


def _number(x, default=0.0):
    try:
        result = float(x)

        if math.isfinite(result):
            return result

        return default

    except (
        ValueError,
        TypeError,
    ):
        return default


def _yes(value):
    return (
        value is True
        or (
            isinstance(value, str)
            and value.lower() == "true"
        )
    )


def _status(section):
    if not isinstance(section, dict):
        return "REVIEW"

    value = _text(
        section.get("status"),
        20,
    ).upper()

    if value in {
        "PASS",
        "FAIL",
        "REVIEW",
    }:
        return value

    return "REVIEW"


def _proof(x, size=9):
    text = _text(x).lower()

    invalid_prefixes = (
        "ไม่มี",
        "ไม่พบ",
        "ไม่แน่",
        "ไม่ชัด",
        "ปกติ",
        "ยังไม่",
        "n/a",
        "unknown",
        "none",
        "no evidence",
        "not visible",
        "uncertain",
    )

    return (
        len(text) >= size
        and not text.startswith(
            invalid_prefixes
        )
    )


# ==========================================
# FFMPEG
# ==========================================

def _run(args, timeout=115):
    try:
        process = subprocess.run(
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
            f"FFmpeg ทำงานไม่ได้: {exc}"
        ) from exc

    if process.returncode:
        raise AnalysisError(
            "เปิดไฟล์ไม่สำเร็จ: "
            + (
                process.stderr or ""
            )[-450:]
        )

    return process.stdout


# ==========================================
# PROBE VIDEO
# ==========================================

def _probe(file):
    raw = _run([
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of", "json",
        str(file),
    ])

    try:
        data = json.loads(raw)

        duration = float(
            data["format"]["duration"]
        )

        streams = {
            entry.get("codec_type")
            for entry in data["streams"]
        }

    except (
        TypeError,
        KeyError,
        ValueError,
    ) as exc:
        raise AnalysisError(
            "อ่านรายละเอียด MP4 ไม่ได้"
        ) from exc

    if (
        not math.isfinite(duration)
        or not 0.5 <= duration <= MAX_SECONDS
    ):
        raise AnalysisError(
            f"รองรับคลิปยาว "
            f"0.5–{MAX_SECONDS} วินาที"
        )

    if "video" not in streams:
        raise AnalysisError(
            "ไฟล์ไม่มีภาพวิดีโอ"
        )

    return duration, "audio" in streams


# ==========================================
# EXTRACT FRAMES
# ==========================================

def _frames(
    video,
    folder,
    fps,
    start=0.0,
    length=None,
):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_filter = (
        f"fps={fps},"
        f"scale={FRAME_EDGE}:{FRAME_EDGE}:"
        "force_original_aspect_ratio=decrease:"
        "force_divisible_by=2"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-threads", "2",
    ]

    if start > 0:
        command.extend([
            "-ss",
            f"{start:.3f}",
        ])

    command.extend([
        "-i",
        str(video),
    ])

    if length is not None:
        command.extend([
            "-t",
            f"{length:.3f}",
        ])

    command.extend([
        "-vf",
        video_filter,
        "-q:v", "6",
        str(folder / "f_%03d.jpg"),
    ])

    _run(
        command,
        timeout=135,
    )

    paths = sorted(
        folder.glob("f_*.jpg")
    )

    if not paths:
        raise AnalysisError(
            "ดึงเฟรมภาพไม่สำเร็จ"
        )

    return paths


# ==========================================
# MAKE CONTACT SHEETS
# ==========================================

def _sheets(
    files,
    fps,
    folder,
    *,
    start=0.0,
    prefix="sheet",
):
    sheets = []

    for offset in range(
        0,
        len(files),
        6,
    ):
        canvas = Image.new(
            "RGB",
            (990, 560),
            (22, 27, 35),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        selected = files[
            offset:offset + 6
        ]

        for j, file in enumerate(selected):
            with Image.open(file) as opened:
                thumb = ImageOps.contain(
                    opened.convert("RGB"),
                    (320, 236),
                )

            x = (
                (j % 3) * 330
                + (330 - thumb.width) // 2
            )

            y = (
                (j // 3) * 280
                + 26
                + (236 - thumb.height) // 2
            )

            canvas.paste(
                thumb,
                (x, y),
            )

            sec = start + (
                offset + j
            ) / fps

            draw.text(
                (
                    (j % 3) * 330 + 7,
                    (j // 3) * 280 + 6,
                ),
                f"{sec:.2f}s",
                fill="white",
            )

        target = folder / (
            f"{prefix}_{len(sheets):02d}.jpg"
        )

        canvas.save(
            target,
            "JPEG",
            quality=76,
            optimize=True,
        )

        canvas.close()
        sheets.append(target)

    return sheets


# ==========================================
# AUDIO
# ==========================================

def _audio(
    video,
    folder,
    has_audio,
):
    if not has_audio:
        return None

    dest = folder / "audio.wav"

    _run([
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
        str(dest),
    ], timeout=90)

    with wave.open(
        str(dest),
        "rb",
    ) as sound:
        raw = sound.readframes(
            sound.getnframes()
        )

    samples = array("h")
    samples.frombytes(raw)

    if not samples:
        return None

    stride = max(
        1,
        len(samples) // 12000,
    )

    sampled = samples[::stride]

    rms = math.sqrt(
        sum(
            value * value
            for value in sampled
        ) / len(sampled)
    ) / 32768

    if rms < 0.0008:
        return None

    return dest


# ==========================================
# IMAGE / AUDIO INLINE
# ==========================================

def _inline(path, mime):
    encoded = base64.b64encode(
        Path(path).read_bytes()
    ).decode("ascii")

    return {
        "inline_data": {
            "mime_type": mime,
            "data": encoded,
        }
    }


# ==========================================
# STRUCTURED QC SCHEMA
# ==========================================

SCHEMA = r'''
{
 "product_type_th":"ชื่อหมวดสินค้าอย่างเฉพาะเจาะจง",
 "visible_product_th":"สินค้าที่เห็นจริง",
 "spoken_product_th":"สินค้าที่เสียงพูดกำลังขาย หรือไม่มีเสียงพูด",
 "usage_th":"หน้าที่/วิธีใช้งานของสินค้าที่เห็น",
 "product_understanding_checked":true,
 "scale":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "hand_or_person_visible":false,
   "anchor_th":"อธิบายจุดเทียบขนาดหรือว่าไม่มีจุดเทียบ",
   "normal_reason_th":"สิ่งที่ทำให้สเกลสมเหตุสมผล",
   "mini_correct":false,
   "risk_of_miniature":false,
   "events":[
     {
       "time_sec":1.0,
       "seen_th":"",
       "problem_th":""
     }
   ]
 },
 "speech":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "speech_present":true,
   "heard_excerpt_th":"วลีที่ได้ยินจริงสั้นๆ",
   "wrong_words":[
     {
       "time_sec":1.0,
       "heard_th":"",
       "problem_th":""
     }
   ],
   "gibberish":false,
   "gibberish_evidence_th":""
 },
 "match":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "spoken_claim_th":"คำอ้างในเสียงเกี่ยวกับสินค้า",
   "visual_fact_th":"สิ่งที่ภาพแสดง",
   "comparison_th":"อธิบายว่าคำพูดตรงหรือไม่ตรงกับภาพ"
 },
 "people":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "person_or_hand_visible":true,
   "interaction_th":"มือ/คนสัมผัสหรือใช้งานสินค้าอย่างไร",
   "events":[
     {
       "time_sec":1.0,
       "seen_th":"",
       "problem_th":""
     }
   ]
 },
 "visual":{
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "continuity_th":"รูปทรง/จำนวน/ตำแหน่งสินค้าตลอดคลิปเป็นอย่างไร",
   "events":[
     {
       "time_sec":1.0,
       "seen_th":"",
       "problem_th":""
     }
   ]
 },
 "mirror":{
   "applicable":false,
   "status":"PASS",
   "checked":true,
   "confidence":0.0,
   "reflection_expected":false,
   "angle_uncertain":false,
   "reason_th":"อธิบายเรขาคณิตของมุมสะท้อนเมื่อเกี่ยวข้อง",
   "events":[
     {
       "type":"missing_hand_reflection",
       "start_sec":1.0,
       "end_sec":3.5,
       "observed_th":"",
       "geometry_th":"",
       "sample_times_sec":[1.0,2.0,3.5]
     }
   ]
 },
 "risk_flags":[],
 "uncertain_time_sec":null,
 "summary_th":"สรุปตามหลักฐานจริง"
}
'''


# ==========================================
# FIRST PASS PROMPT
# ==========================================

FIRST_PROMPT = r'''
คุณเป็นผู้ตรวจ QC โฆษณาสินค้าจากวิดีโอ AI
โดยไม่มี JSON อ้างอิง

ต้องตรวจทุกด้านที่เกี่ยวข้องอย่างเป็นอิสระก่อนให้ PASS
อย่าให้ PASS เพียงเพราะภาพสวยหรือผู้บรรยายฟังคล่อง

ห้ามตั้งโควตา FAIL
ห้ามสร้างหลักฐานหรือคำพูดที่ไม่มีจริง

ขั้น 1
จำแนกว่าสินค้าหลักที่ขายคืออะไร
จากสิ่งที่เห็นและได้ยิน

ระบุชนิดอย่างเฉพาะเจาะจง เช่น
แป้นบาสเด็ก ชั้นวางครัว ถังขยะ

ระบุจุดใช้งานที่ควรเป็นของสินค้านั้น
โดยไม่สมมติสเปกที่ไม่ได้เห็น

ขั้น 2
ตรวจก่อนสรุป PASS ทีละหัวข้อ

SCALE:
- เทียบสินค้าในมือ กับนิ้ว คน โต๊ะ ชั้น พื้น
- เทียบกับวัตถุในระนาบเดียวกัน
- ตรวจของจิ๋วที่นำเสนอเหมือนของใช้งานจริง
- ตรวจการหดหรือขยายขนาดแบบไร้เหตุผล
- ตรวจความเหมาะสมของวิธีใช้งาน
- ของ MINI ที่ขายเป็น MINI จริงไม่ผิด
- ของพับหรือปรับระดับต่ำไม่ผิดอัตโนมัติ
- ไม่มีจุดอ้างอิงห้ามเดาว่าสูงกี่เซนติเมตร
- หากไม่มีหลักฐานเพียงพอให้ระบุข้อจำกัด

SPEECH:
- ฟังเสียงภาษาไทยจริงทั้งหมด
- คำออกเสียงผิดชัดตั้งแต่ 2 ตำแหน่งให้ FAIL
  เมื่อมีเวลาและคำที่ได้ยินจริงประกอบ
- พูดมั่วต่อเนื่องฟังไม่เข้าใจให้ FAIL
- เพี้ยนเล็กน้อยเพียง 1 คำ ไม่ต้อง FAIL
- สำเนียง ชื่อแบรนด์ ไม่ถือว่าผิดโดยอัตโนมัติ
- ถ้าไม่มีเสียงพูด อย่าสร้างข้อความถอดเสียง

MATCH:
- แยกสินค้าที่เห็นกับสินค้าที่พูดถึง
- ตรวจชนิด สี จำนวน รูปทรง วิธีใช้
- ตรวจคุณสมบัติที่เห็นได้จากภาพ
- พูดสินค้า A แต่ภาพขาย B จริงให้ FAIL
- คำพูดขัดกับภาพชัดเจนให้ FAIL
- ห้ามเดาคุณสมบัติที่ภาพไม่ได้พิสูจน์

PEOPLE:
- ตรวจมือและนิ้วที่จับหรือสัมผัสสินค้า
- นิ้วผิดรูป มือหาย
- มือทะลุวัตถุอย่างผิดธรรมชาติ
- มือเปลี่ยนจำนวนกะทันหัน
- สัดส่วนมือกับสินค้าไม่สมจริง
- ปฏิสัมพันธ์ของคนและสินค้าผิดธรรมชาติ

VISUAL:
- ตรวจรูปร่างสินค้า
- ตรวจชั้น ฐาน ล้อ และขอบที่ควรต่อเนื่อง
- ตรวจสินค้าเปลี่ยนรูปทรงระหว่างเฟรม
- สินค้าละลายหรือหาย
- สินค้าเปลี่ยนชนิด
- วางของขัดฟิสิกส์อย่างรุนแรง

MIRROR:
ตรวจเฉพาะกระจกเงาที่เป็นสินค้าหลัก
- เงาที่ควรปรากฏกลับหาย
- เงาบิดเบี้ยวรุนแรง
- เงาผิดตำแหน่ง
- เงาเคลื่อนผิดจากวัตถุจริง
- มือหายต่อเนื่องเกิน 2 วินาที
  ต้องมีหลักฐานหลาย timestamp จริง
- พิจารณามุมกล้องและการบดบัง
- บางมุมอาจไม่ควรเห็นเงามืออยู่แล้ว

ทุกหัวข้อที่เกี่ยวข้องต้อง:
1. กรอก checked=true
2. บอกข้อสังเกตเฉพาะจริง
3. ตัดสินตามหลักฐาน

คำว่า "ตรวจแล้วปกติ" เฉย ๆ
ไม่เพียงพอสำหรับฉากที่มีมือสัมผัสสินค้า

หากไม่มีคนหรือมือ
PEOPLE.checked=true และ
person_or_hand_visible=false ได้

หากไม่ได้ขายกระจกเงา
mirror.applicable=false

ผลแต่ละด้าน:
status=PASS/FAIL/REVIEW

ผล REVIEW เมื่อมีข้อสงสัยสำคัญ
ที่ยังยืนยันไม่ได้

ไม่ใช่เพราะไม่มี JSON

ค่าความมั่นใจเป็นเพียงความเห็นของ AI
ไม่ใช่หลักฐานอิสระ

ถ้าไม่มีปัญหาจริง:
events=[]
wrong_words=[]

อย่านำค่าตัวอย่างใน schema ไปเป็นผลจริง

ตอบ JSON object เท่านั้นตามโครงสร้างนี้:
''' + SCHEMA


# ==========================================
# INDEPENDENT SECOND PASS PROMPT
# ==========================================

SECOND_PROMPT = r'''
คุณเป็นผู้ตรวจซ้ำอิสระของโฆษณาสินค้า MP4

ตรวจเฉพาะหัวข้อที่ท้ายข้อความกำหนด
อย่ารับเอาคำตัดสินรอบแรก

มีภาพเฉพาะช่วงเสี่ยง
และอาจมีเสียงจริง

ห้ามสร้างเหตุการณ์หรือคำผิด

ถ้าข้อมูลที่ได้รับไม่ครอบคลุมทั้งคลิป
ห้ามสรุปด้านที่ไม่ได้ส่งมา

ถ้าสเกล:
เทียบมือและการใช้งาน
ระวังการพับ/ยืดตามปกติ

ถ้าเสียง:
ฟังและให้ตำแหน่งคำผิดจริง
สองตำแหน่งแยกจากกัน

ถ้าคำพูดเทียบภาพ:
อธิบายชัดว่าสิ่งที่พูดและภาพ
เหมือนหรือต่างกันอย่างไร

ถ้ามือหรือสินค้า:
ตรวจช่วงติดต่อสัมผัส
และการเปลี่ยนรูปตามเวลา

ถ้ากระจก:
ตรวจว่ามุมควรเห็นเงาจริงหรือไม่
และหลักฐานหลายช่วงเวลา

สถานะ PASS ต้องมีเหตุผลเฉพาะ
REVIEW เมื่อตรวจยืนยันไม่ได้

ตอบ JSON object ตาม schema นี้

หัวข้อที่ไม่ได้ตรวจ:
checked=false และ status=PASS
''' + SCHEMA


# ==========================================
# BUILD GEMINI CONTENT
# ==========================================

def _parts(
    prompt,
    duration,
    sheets,
    details,
    audio,
):
    items = [{
        "text": (
            f"{prompt}\n"
            f"ความยาวคลิปจริง "
            f"{duration:.2f} วินาที"
        )
    }]

    for image in sheets:
        items.append({
            "text": (
                "ภาพตามลำดับเวลา "
                "พร้อมเวลาที่มุมภาพ"
            )
        })

        items.append(
            _inline(
                image,
                "image/jpeg",
            )
        )

    for sec, image in details:
        items.append({
            "text": (
                f"ภาพขยายความคมชัด "
                f"ณ {sec:.2f}s"
            )
        })

        items.append(
            _inline(
                image,
                "image/jpeg",
            )
        )

    if audio is not None:
        items.append({
            "text": (
                "ไฟล์เสียงจริงของวิดีโอทั้งหมด"
            )
        })

        items.append(
            _inline(
                audio,
                "audio/wav",
            )
        )

    else:
        items.append({
            "text": (
                "ไม่มีเสียงที่ฟังได้ "
                "ห้ามสมมติว่ามีการพูด"
            )
        })

    return items


# ==========================================
# GEMINI API
# ==========================================

def _api(
    key,
    model,
    parts,
    max_tokens,
):
    if not key:
        raise AnalysisError(
            "ไม่ได้ตั้ง GEMINI_API_KEY"
        )

    chosen = (
        model
        if model in MODELS
        else DEFAULT_MODEL
    )

    options = [chosen]
    index = 0

    while index < len(options):
        selected = options[index]
        index += 1

        url = (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models/"
            f"{selected}:generateContent"
        )

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

        for attempt in range(2):
            try:
                response = requests.post(
                    url,
                    headers={
                        "x-goog-api-key": key
                    },
                    json=payload,
                    timeout=(12, 105),
                )

            except requests.RequestException as exc:
                if attempt == 0:
                    time.sleep(1)
                    continue

                raise AnalysisError(
                    f"ติดต่อ Gemini ไม่ได้: {exc}"
                ) from exc

            if response.status_code == 429:
                raise QuotaError(
                    "Gemini API ถึงโควตา 429 "
                    "กรุณารอแล้วลองใหม่"
                )

            if (
                response.status_code
                in (500, 502, 503, 504)
                and attempt == 0
            ):
                time.sleep(1)
                continue

            if response.status_code >= 400:
                try:
                    reason = response.json().get(
                        "error", {}
                    ).get(
                        "message", ""
                    )

                except ValueError:
                    reason = response.text[:280]

                if response.status_code == 404:
                    # Only retry models visible
                    # to this API key.
                    for available in available_models(
                        key
                    ):
                        if available not in options:
                            options.append(
                                available
                            )

                    if index < len(options):
                        break

                raise AnalysisError(
                    f"Gemini HTTP "
                    f"{response.status_code} "
                    f"({selected}): "
                    f"{_text(reason, 340)}"
                )

            try:
                candidates = response.json()[
                    "candidates"
                ]

                raw = "".join(
                    piece.get("text", "")
                    for piece in candidates[0][
                        "content"
                    ]["parts"]
                ).strip()

                raw = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    raw,
                )

                obj = json.loads(raw)

                required = (
                    "scale",
                    "speech",
                    "match",
                    "people",
                    "visual",
                    "mirror",
                )

                if (
                    not isinstance(obj, dict)
                    or not all(
                        isinstance(
                            obj.get(name),
                            dict,
                        )
                        for name in required
                    )
                ):
                    raise ValueError(
                        "ข้อมูล QC ไม่ครบ 6 ด้าน"
                    )

                return obj

            except (
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                raise AnalysisError(
                    f"Gemini ส่งผล JSON ไม่ครบ: "
                    f"{exc}"
                ) from exc

    raise AnalysisError(
        "Gemini API ไม่สามารถตรวจวิดีโอได้"
    )


# ==========================================
# VALIDATE VISUAL EVENTS
# ==========================================

def _events(
    section,
    duration,
):
    found = []

    raw = section.get(
        "events", []
    )

    if not isinstance(raw, list):
        return found

    for item in raw[:14]:
        if not isinstance(item, dict):
            continue

        t = _number(
            item.get("time_sec"),
            -1,
        )

        seen = _text(
            item.get("seen_th"),
            250,
        )

        why = _text(
            item.get("problem_th"),
            330,
        )

        if (
            0 <= t <= duration
            and _proof(seen)
            and _proof(why)
        ):
            found.append(
                (t, seen, why)
            )

    return found


# ==========================================
# VALIDATE SPEECH ERRORS
# ==========================================

def _words(
    section,
    duration,
):
    found = []

    raw = section.get(
        "wrong_words", []
    )

    if not isinstance(raw, list):
        return found

    for item in raw[:16]:
        if not isinstance(item, dict):
            continue

        t = _number(
            item.get("time_sec"),
            -1,
        )

        heard = _text(
            item.get("heard_th"),
            95,
        )

        why = _text(
            item.get("problem_th"),
            200,
        )

        if (
            0 <= t <= duration
            and heard
            and _proof(why)
            and all(
                abs(t - old[0]) > 0.16
                for old in found
            )
        ):
            found.append(
                (t, heard, why)
            )

    return found


# ==========================================
# VALIDATE MIRROR ERRORS
# ==========================================

def _mirror_events(
    section,
    duration,
):
    found = []

    raw = section.get(
        "events", []
    )

    if not isinstance(raw, list):
        return found

    valid = {
        "missing_hand_reflection",
        "severe_warp",
        "impossible_reflection",
        "unrelated_motion",
    }

    for event in raw[:12]:
        if (
            not isinstance(event, dict)
            or event.get("type")
            not in valid
        ):
            continue

        start = _number(
            event.get("start_sec"),
            -1,
        )

        end = _number(
            event.get("end_sec"),
            -1,
        )

        if not 0 <= start < end <= duration:
            continue

        evidence = _text(
            event.get("observed_th"),
            250,
        )

        geometry = _text(
            event.get("geometry_th"),
            250,
        )

        if (
            not _proof(evidence)
            or not _proof(geometry)
        ):
            continue

        times = event.get(
            "sample_times_sec", []
        )

        if not isinstance(times, list):
            times = []

        unique = sorted({
            round(
                _number(t, -1), 2
            )
            for t in times
            if (
                start - 0.5
                <= _number(t, -1)
                <= end + 0.5
            )
        })

        sustained = (
            len(unique) >= 3
            and unique[-1] - unique[0] >= 1.8
        )

        found.append((
            start,
            end,
            event["type"],
            evidence,
            geometry,
            sustained,
        ))

    return found


def _mirror_expected(data):
    title = _text(
        data.get("product_type_th"),
        180,
    )

    inferred = bool(
        re.search(
            r"กระจกเงา|กระจกส่อง|"
            r"กระจกแต่งหน้า|mirror",
            title,
            re.I,
        )
    )

    applicable = _yes(
        data["mirror"].get("applicable")
    )

    return inferred or applicable


# ==========================================
# EVIDENCE COVERAGE
# ==========================================

def _check_coverage(
    data,
    audible,
    scope,
):
    """
    Missing inspection evidence requires
    a focused audit or REVIEW.

    Missing evidence is NOT proof
    that the video is defective.
    """

    required = set(scope)
    missing = set()

    understood = _yes(
        data.get(
            "product_understanding_checked"
        )
    )

    named = _proof(
        data.get("visible_product_th"),
        4,
    )

    if not understood or not named:
        if "scale" in required:
            missing.add("scale")

        if (
            "match" in required
            and audible
        ):
            missing.add("match")

    for area in required:
        block = data[area]

        if (
            area in {"speech", "match"}
            and not audible
        ):
            continue

        if (
            area == "mirror"
            and not _mirror_expected(data)
        ):
            continue

        if not _yes(
            block.get("checked")
        ):
            missing.add(area)

    if "scale" in required:
        scale = data["scale"]

        if (
            _status(scale) == "PASS"
            and _yes(
                scale.get(
                    "hand_or_person_visible"
                )
            )
            and not _proof(
                scale.get("anchor_th")
            )
            and not _proof(
                scale.get("normal_reason_th")
            )
        ):
            missing.add("scale")

    if (
        "speech" in required
        and audible
    ):
        speech = data["speech"]

        if (
            _yes(
                speech.get("speech_present")
            )
            and _status(speech) == "PASS"
            and not _proof(
                speech.get(
                    "heard_excerpt_th"
                ),
                4,
            )
        ):
            missing.add("speech")

    if (
        "match" in required
        and audible
        and _yes(
            data["speech"].get(
                "speech_present"
            )
        )
    ):
        match = data["match"]

        if (
            _status(match) == "PASS"
            and not _proof(
                match.get("comparison_th")
            )
        ):
            missing.add("match")

    if (
        "people" in required
        and _yes(
            data["people"].get(
                "person_or_hand_visible"
            )
        )
    ):
        people = data["people"]

        if (
            _status(people) == "PASS"
            and not _proof(
                people.get("interaction_th")
            )
        ):
            missing.add("people")

    if (
        "visual" in required
        and _status(
            data["visual"]
        ) == "PASS"
    ):
        if not _proof(
            data["visual"].get(
                "continuity_th"
            )
        ):
            missing.add("visual")

    if (
        "mirror" in required
        and _mirror_expected(data)
    ):
        mirror = data["mirror"]

        if (
            _status(mirror) == "PASS"
            and not _proof(
                mirror.get("reason_th")
            )
        ):
            missing.add("mirror")

    return missing


# ==========================================
# QUALITY DECISION
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
            "speech",
            "match",
            "people",
            "visual",
            "mirror",
        }

    else:
        scope = set(scope)

    failures = []
    concerns = set()

    coverage = _check_coverage(
        data,
        audible,
        scope,
    )

    for area in scope:
        block = data[area]

        confidence = _number(
            block.get("confidence")
        )

        status = _status(block)

        # Scale, people, product appearance

        if area in {
            "scale",
            "people",
            "visual",
        }:
            events = _events(
                block,
                duration,
            )

            threshold = (
                0.73
                if area == "scale"
                else 0.78
            )

            strong = (
                bool(events)
                and confidence >= threshold
            )

            if area == "scale":
                mini_correct = _yes(
                    block.get("mini_correct")
                )

                anchor = _proof(
                    block.get("anchor_th")
                )

                strong = (
                    strong
                    and not mini_correct
                    and (
                        anchor
                        or len(events) >= 2
                    )
                )

            if (
                status == "FAIL"
                and strong
            ):
                t, seen, why = events[0]

                failures.append((
                    area,
                    t,
                    f"{area}: {seen} — {why}",
                ))

            elif (
                status != "PASS"
                or events
                or (
                    area == "scale"
                    and _yes(
                        block.get(
                            "risk_of_miniature"
                        )
                    )
                    and not _yes(
                        block.get(
                            "mini_correct"
                        )
                    )
                )
            ):
                concerns.add(area)

        # Thai speech

        elif area == "speech":
            words = _words(
                block,
                duration,
            )

            if (
                audible
                and status == "FAIL"
                and confidence >= 0.74
                and len(words) >= 2
            ):
                explanation = "; ".join(
                    (
                        f"{t:.1f}s '{word}' "
                        f"({why})"
                    )
                    for t, word, why
                    in words[:3]
                )

                failures.append((
                    area,
                    words[0][0],
                    (
                        "เสียงพูดผิด >=2 ตำแหน่ง: "
                        + explanation
                    ),
                ))

            elif (
                audible
                and status == "FAIL"
                and confidence >= 0.82
                and _yes(
                    block.get("gibberish")
                )
                and _proof(
                    block.get(
                        "gibberish_evidence_th"
                    )
                )
            ):
                failures.append((
                    area,
                    0,
                    (
                        "พูดมั่วต่อเนื่อง: "
                        + _text(
                            block.get(
                                "gibberish_evidence_th"
                            )
                        )
                    ),
                ))

            elif (
                audible
                and (
                    status != "PASS"
                    or len(words) >= 2
                )
            ):
                concerns.add(area)

        # Speech vs image

        elif area == "match":
            spoken = _proof(
                block.get("spoken_claim_th")
            )

            visual_fact = _proof(
                block.get("visual_fact_th")
            )

            if (
                audible
                and status == "FAIL"
                and confidence >= 0.78
                and spoken
                and visual_fact
            ):
                failures.append((
                    area,
                    0,
                    (
                        "เสียงกับภาพขัดกัน: "
                        + _text(
                            block.get(
                                "comparison_th"
                            ),
                            350,
                        )
                    ),
                ))

            elif (
                audible
                and status != "PASS"
            ):
                concerns.add(area)

        # Mirror / reflections

        elif area == "mirror":
            if not _mirror_expected(data):
                continue

            events = _mirror_events(
                block,
                duration,
            )

            valid = []

            for event in events:
                (
                    start,
                    end,
                    kind,
                    seen,
                    geometry,
                    sustained,
                ) = event

                if (
                    not sustained
                    or _yes(
                        block.get(
                            "angle_uncertain"
                        )
                    )
                ):
                    continue

                if (
                    kind == "missing_hand_reflection"
                    and (
                        end - start <= 2
                        or not _yes(
                            block.get(
                                "reflection_expected"
                            )
                        )
                    )
                ):
                    continue

                valid.append(event)

            if (
                status == "FAIL"
                and confidence >= 0.82
                and valid
            ):
                (
                    start,
                    end,
                    kind,
                    seen,
                    geometry,
                    _,
                ) = valid[0]

                failures.append((
                    area,
                    start,
                    (
                        f"เงากระจกผิด ({kind}): "
                        f"{seen}; {geometry}"
                    ),
                ))

            elif (
                status != "PASS"
                or events
            ):
                concerns.add(area)

    # Additional AI risk flags

    flags = data.get(
        "risk_flags", []
    )

    if isinstance(flags, list):
        for flag in flags[:8]:
            if not _proof(flag):
                continue

            text = _text(
                flag
            ).lower()

            if any(
                word in text
                for word in (
                    "เสียง",
                    "พูด",
                    "audio",
                    "speech",
                    "voice",
                )
            ):
                concerns.add(
                    "speech" if audible
                    else "visual"
                )

            elif any(
                word in text
                for word in (
                    "สเกล",
                    "ขนาด",
                    "scale",
                    "mini",
                )
            ):
                concerns.add("scale")

            elif any(
                word in text
                for word in (
                    "เงา",
                    "กระจก",
                    "reflect",
                    "mirror",
                )
            ):
                concerns.add("mirror")

            elif any(
                word in text
                for word in (
                    "มือ",
                    "นิ้ว",
                    "hand",
                    "finger",
                )
            ):
                concerns.add("people")

            elif any(
                word in text
                for word in (
                    "ความตรง",
                    "สินค้าไม่ตรง",
                    "mismatch",
                )
            ):
                concerns.add(
                    "match" if audible
                    else "visual"
                )

            else:
                concerns.add("visual")

    return (
        failures,
        concerns | coverage,
        coverage,
    )


# ==========================================
# AUDIT RISK FOCUS
# ==========================================

def _audit_focus(risks):
    accepted = {
        "scale",
        "speech",
        "match",
        "people",
        "visual",
        "mirror",
    }

    return set(risks) & accepted


def _focus_times(
    result,
    duration,
    focus,
):
    times = []

    for area in focus:
        section = result[area]

        if area == "mirror":
            for (
                start,
                end,
                *_
            ) in _mirror_events(
                section,
                duration,
            ):
                times.append(
                    (start + end) / 2
                )

        elif area == "speech":
            times.extend(
                t
                for t, *_ in _words(
                    section,
                    duration,
                )
            )

        elif area in {
            "scale",
            "people",
            "visual",
        }:
            times.extend(
                t
                for t, *_ in _events(
                    section,
                    duration,
                )
            )

    uncertain = _number(
        result.get(
            "uncertain_time_sec"
        ),
        -1,
    )

    if 0 <= uncertain <= duration:
        times.append(uncertain)

    if times:
        return sorted(times)[0]

    return duration * 0.50


# ==========================================
# TARGETED 5 FPS AUDIT
# ==========================================

def _make_audit_media(
    video,
    folder,
    duration,
    focus,
    first,
    fps,
    wav,
    sheets,
    details,
):
    image_areas = {
        "scale",
        "match",
        "people",
        "visual",
        "mirror",
    }

    audio_areas = {
        "speech",
        "match",
    }

    want_images = bool(
        focus & image_areas
    )

    want_audio = bool(
        focus & audio_areas
    )

    window = None

    dense_areas = {
        "scale",
        "people",
        "visual",
        "mirror",
    }

    if (
        want_images
        and focus & dense_areas
    ):
        moment = _focus_times(
            first,
            duration,
            focus,
        )

        start = max(
            0.0,
            min(
                max(
                    0.0,
                    duration - AUDIT_WINDOW_SEC,
                ),
                moment - AUDIT_WINDOW_SEC / 2,
            ),
        )

        length = min(
            AUDIT_WINDOW_SEC,
            duration - start,
        )

        if length >= 0.5:
            dense = _frames(
                video,
                folder / "dense",
                AUDIT_FPS,
                start=start,
                length=length,
            )

            dense_sheets = _sheets(
                dense,
                AUDIT_FPS,
                folder,
                start=start,
                prefix="dense",
            )

            indices = sorted({
                0,
                len(dense) // 2,
                len(dense) - 1,
            })

            samples = [
                (
                    start + i / AUDIT_FPS,
                    dense[i],
                )
                for i in indices
            ]

            window = (
                dense_sheets[:3],
                samples[::2][:2],
                start,
                length,
            )

    if want_images:
        if window:
            image_sheets = window[0]
            image_details = window[1]

        else:
            step = max(
                1,
                len(sheets) // 3,
            )

            image_sheets = (
                sheets[::step][:3]
            )

            image_details = (
                details[:2]
            )

    else:
        image_sheets = []
        image_details = []

    return (
        image_sheets,
        image_details,
        wav if want_audio else None,
        window,
    )


# ==========================================
# MAIN INSPECT FUNCTION
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
    Compatible with V3.9.1 qc_store.py.

    MP4-only mode:
    reference is intentionally ignored.
    """

    started = time.perf_counter()

    video = Path(path)
    thumbnail = Path(thumbnail)

    duration, has_audio = _probe(
        video
    )

    fps = FPS.get(
        mode, 4
    )

    with tempfile.TemporaryDirectory(
        prefix="qc392_"
    ) as temp:
        folder = Path(temp)

        # Prepare full-video 4 FPS sequence
        frames = _frames(
            video,
            folder / "frames",
            fps,
        )

        sheets = _sheets(
            frames,
            fps,
            folder,
        )

        # Larger standalone frames
        indices = sorted({
            round(
                (len(frames) - 1) * value
            )
            for value in (
                0.13,
                0.38,
                0.64,
                0.87,
            )
        })

        details = [
            (
                index / fps,
                frames[index],
            )
            for index in indices
        ]

        # Thumbnail
        thumbnail.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not thumbnail.exists():
            with Image.open(
                frames[0]
            ) as image:
                small = ImageOps.contain(
                    image.convert("RGB"),
                    (320, 320),
                )

                small.save(
                    thumbnail,
                    "JPEG",
                    quality=72,
                )

        # Audio
        wav = _audio(
            video,
            folder,
            has_audio,
        )

        prepared = time.perf_counter()

        # ==================================
        # FIRST PASS
        # ==================================

        first = _api(
            key,
            model,
            _parts(
                FIRST_PROMPT,
                duration,
                sheets,
                details,
                wav,
            ),
            3600,
        )

        first_done = time.perf_counter()

        (
            initial_fail,
            risks,
            coverage,
        ) = _judge(
            first,
            duration,
            wav is not None,
        )

        second = None
        focus = set()
        dense_window = None

        final_fail = list(
            initial_fail
        )

        pending = set(
            risks
        )

        # ==================================
        # SECOND PASS IF NEEDED
        # ==================================

        if (
            not initial_fail
            and risks
        ):
            focus = _audit_focus(
                risks
            )

            if focus:
                (
                    audit_sheets,
                    audit_details,
                    audit_wav,
                    dense_window,
                ) = _make_audit_media(
                    video,
                    folder,
                    duration,
                    focus,
                    first,
                    fps,
                    wav,
                    sheets,
                    details,
                )

                instruction = (
                    SECOND_PROMPT
                    + "\nประเด็นตรวจซ้ำ: "
                    + ", ".join(
                        sorted(focus)
                    )
                )

                if dense_window:
                    begin = dense_window[2]
                    end = (
                        begin
                        + dense_window[3]
                    )

                    instruction += (
                        "\nภาพที่สองเป็นช่วงเวลา"
                        f"ประมาณ {begin:.2f}–"
                        f"{end:.2f} วินาที "
                        "ไม่ใช่ภาพทั้งหมดของคลิป"
                    )

                second = _api(
                    key,
                    model,
                    _parts(
                        instruction,
                        duration,
                        audit_sheets,
                        audit_details,
                        audit_wav,
                    ),
                    3200,
                )

                (
                    checked_fails,
                    newer_risks,
                    second_missing,
                ) = _judge(
                    second,
                    duration,
                    wav is not None,
                    scope=focus,
                )

                final_fail = checked_fails
                pending = set(newer_risks)

                # Do not erase concrete
                # first-round evidence silently.

                for area in focus:
                    prior = first[area]
                    later = second[area]

                    if _status(prior) in (
                        "REVIEW",
                        "FAIL",
                    ):
                        if (
                            _status(later) == "PASS"
                            and not _yes(
                                later.get("checked")
                            )
                        ):
                            pending.add(area)

                        if area in {
                            "scale",
                            "people",
                            "visual",
                        }:
                            prior_events = _events(
                                prior,
                                duration,
                            )

                            has_final_failure = any(
                                item[0] == area
                                for item in final_fail
                            )

                            if (
                                prior_events
                                and not has_final_failure
                            ):
                                pending.add(area)

                    if area == "mirror":
                        prior_mirror_events = (
                            _mirror_events(
                                prior,
                                duration,
                            )
                        )

                        has_mirror_fail = any(
                            item[0] == area
                            for item in final_fail
                        )

                        if (
                            prior_mirror_events
                            and not has_mirror_fail
                        ):
                            pending.add(area)

                    if (
                        area in coverage
                        and area in second_missing
                    ):
                        pending.add(area)

            else:
                pending.add(
                    "unresolved"
                )

        # ==================================
        # FINAL DECISION
        # ==================================

        if final_fail:
            decision = "FAIL"

        elif pending:
            decision = "REVIEW"

        else:
            decision = "PASS"

        if final_fail:
            reason = "; ".join(
                item[2]
                for item in final_fail[:3]
            )

        elif pending:
            reason = (
                "รอตรวจด้วยคน: "
                "ต้องยืนยันด้าน "
                + ", ".join(
                    sorted(pending)
                )
            )

        else:
            reason = (
                _text(
                    first.get("summary_th"),
                    750,
                )
                or (
                    "ตรวจแล้วไม่พบข้อผิดพลาดสำคัญ"
                )
            )

        issues = [
            {
                "category": area,
                "time_sec": round(t, 2),
                "detail_th": why,
                "severity": "critical",
            }
            for area, t, why
            in final_fail
        ]

        if decision == "REVIEW":
            issues.append({
                "category": "human_review",
                "time_sec": 0,
                "detail_th": reason,
                "severity": "warning",
            })

        source = (
            second
            if (
                second is not None
                and final_fail
            )
            else first
        )

        ended = time.perf_counter()

        return {
            "decision": decision,

            "reason_th": reason[:850],

            "product_type_th": _text(
                source.get(
                    "product_type_th"
                ),
                180,
            ),

            "product_name_th": _text(
                source.get(
                    "visible_product_th"
                ),
                180,
            ),

            "spoken_product_th": _text(
                source.get(
                    "spoken_product_th"
                ),
                180,
            ),

            "speech_summary_th": _text(
                first["speech"].get(
                    "heard_excerpt_th"
                ),
                700,
            ),

            "video_type": "PRODUCT",

            "scale_review": source["scale"],
            "audio_review": source["speech"],
            "match_review": source["match"],
            "people_review": source["people"],
            "visual_review": source["visual"],
            "mirror_review": source["mirror"],

            "scale_audit": (
                second["scale"]
                if second else None
            ),

            "speech_audit": (
                second["speech"]
                if second else None
            ),

            "match_audit": (
                second["match"]
                if second else None
            ),

            "people_audit": (
                second["people"]
                if second else None
            ),

            "visual_audit": (
                second["visual"]
                if second else None
            ),

            "mirror_audit": (
                second["mirror"]
                if second else None
            ),

            "first_scale_review": (
                first["scale"]
            ),

            "first_audio_review": (
                first["speech"]
            ),

            "evidence_coverage_missing": (
                sorted(coverage)
            ),

            "issues": issues,

            "review_required": (
                decision == "REVIEW"
            ),

            "audit_done": (
                second is not None
            ),

            "audit_trigger": (
                ", ".join(
                    sorted(focus)
                )
                if focus else "none"
            ),

            "audit_dense_window_sec": (
                [
                    round(
                        dense_window[2],
                        2,
                    ),
                    round(
                        dense_window[2]
                        + dense_window[3],
                        2,
                    ),
                ]
                if dense_window
                else None
            ),

            "product_reference_used": False,

            "duration": round(
                duration,
                2,
            ),

            "elapsed": round(
                ended - started,
                2,
            ),

            "engine": ENGINE_VERSION,

            "timings": {
                "prepare_seconds": round(
                    prepared - started,
                    2,
                ),

                "first_api_seconds": round(
                    first_done - prepared,
                    2,
                ),

                "audit_api_seconds": (
                    round(
                        ended - first_done,
                        2,
                    )
                    if second else 0
                ),
            },
        }
