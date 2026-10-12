
"""V3.9 Hybrid Quality Engine.

MP4 alone OR MP4 + optional product JSON.
PASS / FAIL / REVIEW.

Uses one Gemini request for clearly good/bad clips
and a targeted second request for uncertain clips.
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
from PIL import Image, ImageDraw, ImageOps


ENGINE_VERSION = "v3.9-hybrid-quality-20261012-r1"

ALLOWED_MODELS = {
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
}

FPS = {
    "Fast": 1,
    "Balanced": 2,
    "Deep": 3,
}

FRAME_EDGE = 640
MAX_SECONDS = 30
RECHECK_MODE = "smart"
CLEAN_SAMPLE_PERCENT = 5


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


SCHEMA = r'''
{
 "product_type_th":"ประเภทสินค้าจากภาพ",
 "video_type":"HANDS",
 "visible_product_th":"สิ่งที่เห็นจริงในคลิป",
 "spoken_product_th":"สินค้าที่พูดถึงในเสียง",
 "speech_summary_th":"สรุปคำพูดจริง",
 "scale":{
  "result":"PASS",
  "confidence":0.0,
  "reference_fit":"consistent",
  "observed_surface":"unknown",
  "hand_anchor_th":"",
  "environment_anchor_th":"",
  "product_use_th":"",
  "potential_miniature":false,
  "mini_is_correct_product":false,
  "extension_evidence_th":"",
  "events":[
   {
    "time_sec":1.0,
    "seen_th":"",
    "conflict_th":""
   }
  ],
  "reason_th":""
 },
 "speech":{
  "result":"PASS",
  "confidence":0.0,
  "speech_present":true,
  "wrong_words":[
   {
    "time_sec":1.0,
    "heard_th":"",
    "why_wrong_th":""
   }
  ],
  "gibberish":false,
  "gibberish_evidence_th":"",
  "reason_th":""
 },
 "match":{
  "result":"PASS",
  "confidence":0.0,
  "spoken_claim_th":"",
  "video_fact_th":"",
  "json_fact_th":"",
  "reason_th":""
 },
 "visual":{
  "result":"PASS",
  "confidence":0.0,
  "events":[
   {
    "time_sec":1.0,
    "seen_th":"",
    "wrong_th":""
   }
  ],
  "reason_th":""
 },
 "mirror":{
  "applicable":false,
  "result":"PASS",
  "confidence":0.0,
  "reflection_expected":false,
  "angle_uncertain":false,
  "events":[
   {
    "type":"missing_hand_reflection",
    "start_sec":1.0,
    "end_sec":3.5,
    "seen_th":"",
    "geometry_th":"",
    "sample_times_sec":[1.0,2.0,3.5]
   }
  ],
  "reason_th":""
 },
 "risk_flags":[],
 "summary_th":""
}
'''


FIRST_PROMPT = r'''
คุณเป็นผู้ตรวจ QC วิดีโอโฆษณาสินค้า AI
ตรวจภาพตามลำดับเวลาและฟังเสียงจริง

เป้าหมาย:
- คลิปดีจริงต้อง PASS
- ผิดรุนแรงที่มีหลักฐานต้อง FAIL
- ยังไม่แน่ใจเรื่องสำคัญต้อง REVIEW
- ห้ามพยายามให้ FAIL ครบจำนวนที่กำหนด
- ห้ามสร้างหลักฐานขึ้นเอง

ขั้นแรก จำแนกประเภทสินค้าอย่างเฉพาะเจาะจง
จากภาพ ไม่จำกัดหมวดหมู่ เช่น
ลูกฟุตบอล แป้นบาส กระจก รองเท้าหรือเครื่องใช้ไฟฟ้า

ถ้ามี JSON ให้ใช้เป็นข้อมูลสินค้าอ้างอิง
ถ้าไม่มี JSON ให้ตรวจจากภาพและคำพูด
อย่าลงโทษคลิปเพราะไม่มี JSON

ด้านที่ 1: สเกลและการใช้งาน
- เทียบขนาดกับมือ นิ้ว คน โต๊ะ พื้น ชั้นวาง
- เทียบกับวัตถุในระนาบเดียวกัน
- ตรวจว่าขนาดเปลี่ยนผิดธรรมชาติหรือไม่
- ตรวจของจิ๋วที่นำเสนอเป็นสินค้าขนาดจริง
- ตรวจว่าสินค้าใช้งานได้ตามที่โฆษณาหรือไม่
- สินค้าอาจพับเก็บหรือปรับอยู่ระดับต่ำได้
- สินค้าวางบนชั้นไม่ได้แปลว่าขนาดผิดทันที
- หากมีข้ออ้างขนาดใน JSON ให้เทียบกับภาพ
- ห้ามเดาขนาดเป็นเซนติเมตรจากภาพล้วน
- หลักฐานไม่พอให้ REVIEW ไม่ใช่ FAIL

ด้านที่ 2: เสียงภาษาไทย
- ฟังคำพูดจริงทุกช่วง
- ออกเสียงผิดอย่างชัดเจนตั้งแต่ 2 ตำแหน่งคำ
  ให้ FAIL เมื่อระบุคำและเวลาได้
- เสียงมั่วต่อเนื่องฟังไม่เข้าใจ ให้ FAIL
- เพี้ยนเล็กน้อย 1 คำยัง PASS ได้
- สำเนียง AI ชื่อแบรนด์ ศัพท์เทคนิค
  ไม่ผิดโดยอัตโนมัติ
- ห้ามสร้างคำผิดที่ไม่ได้ยินจริง

ด้านที่ 3: ความสัมพันธ์ของคำพูดและสินค้า
- เปรียบเทียบชนิดสินค้าที่พูดกับสิ่งที่เห็น
- ตรวจจำนวน ขนาด คุณสมบัติ วิธีใช้
- หากมี JSON ให้ตรวจเทียบกับข้อมูลสินค้า
- พูดขายสินค้าอีกชนิดหนึ่งอย่างชัดเจนให้ FAIL
- ไม่มีหลักฐานตัวเลข ห้ามเดาว่าผิด
- ความขัดแย้งที่ยังพิสูจน์ไม่ได้ให้ REVIEW

ด้านที่ 4: ความสมจริง
- มือทะลุสินค้าผิดธรรมชาติ
- นิ้วหายขณะสัมผัสสินค้า
- สินค้าละลายหรือบิดเบี้ยวรุนแรง
- สินค้าเปลี่ยนชนิดระหว่างคลิป
- สินค้าหายโดยไม่มีเหตุสมควร
- รูปทรงผิดจนใช้งานไม่ได้
สิ่งเหล่านี้ให้ FAIL เมื่อเห็นหลักฐานชัด
ข้อผิดพลาดเล็กน้อยที่ไม่กระทบสินค้า PASS ได้

ด้านที่ 5: กระจกและเงาสะท้อน
ถ้าสินค้าหลักเป็นกระจกเงาหรือผิวสะท้อน:
- ตรวจมือที่ควรสะท้อนแต่เงาหายไป
- เงามือเคลื่อนไหวไม่สัมพันธ์กับมือ
- ภาพสะท้อนบิดเบี้ยวอย่างรุนแรง
- มือทะลุหรือถูกกลืนเข้าไปในกระจก
- ตรวจมุมกล้องและสิ่งบดบังประกอบ
- การอ้างว่ามือหายเกิน 2 วินาที
  ต้องมีหลักฐานจากหลายเฟรมตามเวลา
- ถ้ามุมสะท้อนไม่แน่ชัดให้ REVIEW
ถ้าไม่ใช่กระจกให้ mirror.applicable=false

การตอบ:
ใช้ PASS / FAIL / REVIEW แยกทุกด้าน
confidence คือความมั่นใจของ AI
ไม่ใช่หลักฐานยืนยันในตัวเอง

ทุกเหตุการณ์ต้องอธิบายสิ่งที่เห็นจริงพร้อมเวลา
ไม่มีเหตุการณ์ให้ events=[]
ไม่มีคำผิดให้ wrong_words=[]

reference_fit=consistent|contradicts|unverified
observed_surface=floor|shelf|table|handheld|wall|unknown
mirror event type:
missing_hand_reflection
severe_warp
impossible_reflection
unrelated_motion

ตอบเป็น JSON object อย่างเดียวตามโครงสร้าง:
''' + SCHEMA


AUDIT_PROMPT = r'''
คุณเป็นผู้ตรวจ QC อิสระรอบที่สอง
ตรวจเฉพาะด้านที่ระบุท้ายคำสั่ง

ห้ามอาศัยคำตัดสินของรอบแรก
และห้ามสร้างเหตุการณ์ขึ้นมาเอง

ตรวจชนิดสินค้า สัดส่วนกับมือ คน สภาพแวดล้อม
และความเป็นไปได้ของการใช้งาน

ถ้าเป็นสินค้าพับหรือปรับระดับได้
ต้องระวังการตีความผิดสเกล

ตรวจเสียงที่ได้ยินจริง
ตรวจคำพูดสัมพันธ์กับภาพและ JSON เมื่อมี

สำหรับกระจก:
ตรวจเงามือ ภาพสะท้อน และมุมสะท้อน
หากอ้างว่ามือหายเกิน 2 วินาที
ต้องมีหลักฐานหลายช่วงเวลา

ถ้ายืนยันผิดได้จริงให้ FAIL
ถ้ายืนยันไม่ได้ให้ REVIEW
หากไม่พบข้อผิดพลาดสำคัญให้ PASS

ตอบ JSON object ตามโครงสร้างต่อไปนี้
ด้านที่ไม่ได้ตรวจให้ PASS และ events=[]:
''' + SCHEMA


# ==========================================
# HELPERS
# ==========================================

def _text(x, limit=450):
    return str(x or "").strip()[:limit]


def _num(x, fallback=0.0):
    try:
        n = float(x)
        return n if math.isfinite(n) else fallback
    except (ValueError, TypeError):
        return fallback


def _boolean(x):
    return (
        x is True
        or (
            isinstance(x, str)
            and x.lower() == "true"
        )
    )


def _status(block):
    status = _text(
        block.get("result")
        if isinstance(block, dict)
        else None
    ).upper()

    return (
        status
        if status in ("PASS", "FAIL", "REVIEW")
        else "REVIEW"
    )


def _evidence(x):
    s = _text(x).lower()

    return (
        len(s) >= 8
        and not s.startswith((
            "ไม่มี",
            "ไม่พบ",
            "ไม่แน่ใจ",
            "ยืนยันไม่ได้",
            "ปกติ",
            "none",
            "unknown",
            "no evidence",
        ))
    )


def _run(cmd, timeout=110):
    try:
        process = subprocess.run(
            cmd,
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
            f"FFmpeg/FFprobe ทำงานไม่ได้: {exc}"
        ) from exc

    if process.returncode:
        raise AnalysisError(
            "อ่านวิดีโอไม่ได้: "
            + (process.stderr or "")[-400:]
        )

    return process.stdout


# ==========================================
# VIDEO PREPARATION
# ==========================================

def _probe(path):
    raw = _run([
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of", "json",
        str(path),
    ])

    try:
        obj = json.loads(raw)

        duration = float(
            obj["format"]["duration"]
        )

        types = {
            s.get("codec_type")
            for s in obj.get("streams", [])
        }

    except (
        KeyError,
        ValueError,
        TypeError,
    ) as exc:
        raise AnalysisError(
            "อ่านความยาววิดีโอไม่ได้"
        ) from exc

    if (
        not math.isfinite(duration)
        or not 0.5 <= duration <= MAX_SECONDS
    ):
        raise AnalysisError(
            f"รองรับคลิป 0.5–{MAX_SECONDS} วินาที"
        )

    if "video" not in types:
        raise AnalysisError(
            "ไม่มีภาพวิดีโอ"
        )

    return duration, "audio" in types


def _extract(video, folder, fps):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    filt = (
        f"fps={fps},"
        f"scale={FRAME_EDGE}:{FRAME_EDGE}:"
        "force_original_aspect_ratio=decrease:"
        "force_divisible_by=2"
    )

    _run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin", "-y",
        "-threads", "2",
        "-i", str(video),
        "-vf", filt,
        "-q:v", "5",
        str(folder / "f_%03d.jpg"),
    ], 140)

    frames = sorted(
        folder.glob("f_*.jpg")
    )

    if not frames:
        raise AnalysisError(
            "ดึงภาพจากคลิปไม่ได้"
        )

    return frames


def _sheets(frames, fps, folder):
    result = []

    for start in range(0, len(frames), 6):
        canvas = Image.new(
            "RGB",
            (930, 516),
            (25, 29, 35),
        )

        draw = ImageDraw.Draw(canvas)

        for i, src in enumerate(
            frames[start:start + 6]
        ):
            with Image.open(src) as im:
                thumb = ImageOps.contain(
                    im.convert("RGB"),
                    (300, 226),
                )

            x = (
                (i % 3) * 310
                + (310 - thumb.width) // 2
            )

            y = (
                (i // 3) * 258
                + 24
                + (226 - thumb.height) // 2
            )

            canvas.paste(thumb, (x, y))

            draw.text(
                (
                    (i % 3) * 310 + 8,
                    (i // 3) * 258 + 5,
                ),
                f"{(start + i) / fps:.1f}s",
                fill="white",
            )

        dst = folder / (
            f"sheet_{len(result):02d}.jpg"
        )

        canvas.save(
            dst,
            "JPEG",
            quality=75,
            optimize=True,
        )

        canvas.close()
        result.append(dst)

    return result


def _audio(video, folder, has_audio):
    if not has_audio:
        return None

    dst = folder / "speech.wav"

    _run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin", "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst),
    ], 75)

    with wave.open(
        str(dst), "rb"
    ) as w:
        data = w.readframes(
            w.getnframes()
        )

    samples = array("h")
    samples.frombytes(data)

    if not samples:
        return None

    rms = math.sqrt(
        sum(v * v for v in samples)
        / len(samples)
    ) / 32768

    return (
        dst
        if rms >= 0.0008
        else None
    )


# ==========================================
# GEMINI REQUEST
# ==========================================

def _inline(file, mime):
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(
                Path(file).read_bytes()
            ).decode("ascii"),
        }
    }


def _parts(
    prompt,
    reference,
    duration,
    sheets,
    extra,
    wav,
):
    if reference:
        context = (
            "ผู้ขายแนบข้อมูลสินค้า JSON "
            "ให้ใช้เปรียบเทียบความถูกต้อง: "
            + _text(reference, 1250)
        )

    else:
        context = (
            "ไม่มี JSON: ให้จำแนกประเภทสินค้า "
            "และตรวจจากภาพและเสียงเอง "
            "อย่าเดาตัวเลข/รุ่นสินค้า "
            "หรือให้ REVIEW เพียงเพราะไม่มี JSON"
        )

    parts = [{
        "text": (
            f"{prompt}\n"
            f"ความยาวคลิป {duration:.2f}s\n"
            f"{context}"
        )
    }]

    for file in sheets:
        parts.extend([
            {
                "text": "ภาพเรียงตามเวลาจากต้นไปท้าย"
            },
            _inline(file, "image/jpeg"),
        ])

    for t, file in extra:
        parts.extend([
            {
                "text": (
                    f"รายละเอียดเฟรม {t:.1f}s"
                )
            },
            _inline(file, "image/jpeg"),
        ])

    if wav is not None:
        parts.extend([
            {
                "text": (
                    "เสียงพูดและเสียงประกอบจริงทั้งคลิป"
                )
            },
            _inline(wav, "audio/wav"),
        ])

    else:
        parts.append({
            "text": (
                "ไม่มีเสียงที่ได้ยินชัด "
                "ห้ามสร้างคำผิดขึ้นเอง"
            )
        })

    return parts


def _api(key, model, parts, max_tokens):
    if not key:
        raise AnalysisError(
            "ไม่พบ GEMINI_API_KEY"
        )

    if model not in ALLOWED_MODELS:
        raise AnalysisError(
            f"โมเดล {model} ไม่อยู่ในรายการที่รองรับ"
        )

    endpoint = (
        "https://generativelanguage.googleapis.com"
        "/v1beta/models/"
        + model
        + ":generateContent"
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

    for retry in range(2):
        try:
            resp = requests.post(
                endpoint,
                headers={
                    "x-goog-api-key": key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(12, 100),
            )

        except requests.RequestException as exc:
            if retry == 0:
                time.sleep(1)
                continue

            raise AnalysisError(
                f"ติดต่อ Gemini ไม่สำเร็จ: {exc}"
            ) from exc

        if resp.status_code == 429:
            raise QuotaError(
                "Gemini API ถึงโควตา (429) "
                "กรุณารอแล้วเริ่มคิวต่อ"
            )

        if (
            resp.status_code
            in (500, 502, 503, 504)
            and retry == 0
        ):
            time.sleep(1)
            continue

        if resp.status_code >= 400:
            try:
                error = resp.json().get(
                    "error", {}
                ).get("message", "")

            except ValueError:
                error = resp.text

            raise AnalysisError(
                f"Gemini HTTP {resp.status_code}: "
                f"{_text(error, 350)}"
            )

        try:
            arr = resp.json()[
                "candidates"
            ][0]["content"]["parts"]

            raw = "".join(
                x.get("text", "")
                for x in arr
            ).strip()

            raw = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                raw,
            )

            result = json.loads(raw)

            required = (
                "scale",
                "speech",
                "match",
                "visual",
                "mirror",
            )

            if (
                not isinstance(result, dict)
                or any(
                    not isinstance(result.get(k), dict)
                    for k in required
                )
            ):
                raise ValueError(
                    "Gemini ส่ง JSON ตรวจไม่ครบ"
                )

            return result

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise AnalysisError(
                f"คำตอบ Gemini JSON ไม่สมบูรณ์: {exc}"
            ) from exc

    raise AnalysisError(
        "Gemini ไม่ตอบกลับ"
    )


# ==========================================
# VALIDATE EVIDENCE
# ==========================================

def _events(block, duration, kind="scale"):
    if (
        not isinstance(block, dict)
        or not isinstance(
            block.get("events"),
            list,
        )
    ):
        return []

    found = []

    for item in block["events"][:12]:
        if not isinstance(item, dict):
            continue

        t = _num(
            item.get("time_sec"), -1
        )

        seen = _text(
            item.get("seen_th")
        )

        why = _text(
            item.get(
                "conflict_th"
                if kind == "scale"
                else "wrong_th"
            )
        )

        if (
            0 <= t <= duration
            and _evidence(seen)
            and _evidence(why)
        ):
            found.append(
                (t, seen, why)
            )

    return found


def _words(speech, duration):
    events = speech.get(
        "wrong_words", []
    )

    if not isinstance(events, list):
        return []

    found = []

    for item in events[:16]:
        if not isinstance(item, dict):
            continue

        t = _num(
            item.get("time_sec"), -1
        )

        heard = _text(
            item.get("heard_th")
        )

        why = _text(
            item.get("why_wrong_th")
        )

        if (
            0 <= t <= duration
            and heard
            and _evidence(why)
            and all(
                abs(t - x[0]) > 0.18
                for x in found
            )
        ):
            found.append(
                (t, heard, why)
            )

    return found


def _mirrors(mirror, duration):
    events = mirror.get(
        "events", []
    )

    if not isinstance(events, list):
        return []

    items = []

    valid_kinds = {
        "missing_hand_reflection",
        "severe_warp",
        "impossible_reflection",
        "unrelated_motion",
    }

    for item in events[:10]:
        if not isinstance(item, dict):
            continue

        start = _num(
            item.get("start_sec"), -1
        )

        end = _num(
            item.get("end_sec"), -1
        )

        times = item.get(
            "sample_times_sec", []
        )

        if not isinstance(times, list):
            times = []

        samples = sorted({
            round(_num(t, -1), 2)
            for t in times
            if (
                start - 0.5
                <= _num(t, -1)
                <= end + 0.5
            )
        })

        if (
            item.get("type") in valid_kinds
            and 0 <= start < end <= duration
            and _evidence(
                item.get("seen_th")
            )
            and _evidence(
                item.get("geometry_th")
            )
        ):
            multiple = (
                len(samples) >= 4
                and samples[-1] - samples[0] >= 2.0
            )

            items.append((
                start,
                end,
                multiple,
                _text(item.get("seen_th")),
                _text(item.get("geometry_th")),
                item.get("type"),
            ))

    return items


# ==========================================
# PRODUCT RISK
# ==========================================

def _dimension_claim(reference):
    if not reference:
        return False

    matches = re.finditer(
        r"(?<!\d)(\d{1,4}(?:[.,]\d+)?)\s*"
        r"(ซม\.?|เซนติเมตร|cm\b|เมตร|\bm\b|"
        r"นิ้ว|inch(?:es)?\b)",
        reference,
        re.I,
    )

    for item in matches:
        number = _num(
            item.group(1).replace(",", ".")
        )

        unit = item.group(2).lower()

        if unit in ("เมตร", "m"):
            cm = number * 100

        elif unit in (
            "นิ้ว",
            "inch",
            "inches",
        ):
            cm = number * 2.54

        else:
            cm = number

        if 75 <= cm <= 1000:
            return True

    return False


def _mirror_claim(reference):
    return bool(re.search(
        r"กระจกเงา|กระจกแต่งหน้า|กระจกส่อง|"
        r"กระจกติดผนัง|กระจกเต็มตัว|\bmirror\b",
        reference or "",
        re.I,
    ))


# ==========================================
# DECISION RULES
# ==========================================

def _judge(
    data,
    duration,
    reference,
    scope=None,
):
    scope = set(
        scope or (
            "scale",
            "speech",
            "match",
            "visual",
            "mirror",
        )
    )

    fails = []
    needs = []

    sc = data["scale"]
    sp = data["speech"]
    ma = data["match"]
    vi = data["visual"]
    mi = data["mirror"]

    # SCALE
    if "scale" in scope:
        evidence = _events(
            sc, duration
        )

        fit = _text(
            sc.get("reference_fit")
        ).lower()

        confident = (
            _num(
                sc.get("confidence")
            ) >= 0.75
        )

        anchor = (
            _evidence(
                sc.get("hand_anchor_th")
            )
            or _evidence(
                sc.get(
                    "environment_anchor_th"
                )
            )
        )

        mini_ok = _boolean(
            sc.get("mini_is_correct_product")
        )

        if (
            (
                _status(sc) == "FAIL"
                or fit == "contradicts"
            )
            and confident
            and evidence
            and (anchor or len(evidence) >= 2)
            and not mini_ok
        ):
            sec, seen, why = evidence[0]

            fails.append((
                "scale",
                sec,
                f"สเกลผิด: {seen}; {why}",
            ))

        elif (
            _status(sc) != "PASS"
            or fit == "contradicts"
            or (
                _boolean(
                    sc.get("potential_miniature")
                )
                and not mini_ok
            )
            or (
                _dimension_claim(reference)
                and _text(
                    sc.get("observed_surface")
                ).lower() in (
                    "table",
                    "shelf",
                )
                and not mini_ok
                and not _evidence(
                    sc.get(
                        "extension_evidence_th"
                    )
                )
                and not _evidence(
                    sc.get(
                        "environment_anchor_th"
                    )
                )
            )
        ):
            needs.append("scale")

    # SPEECH
    if "speech" in scope:
        words = _words(
            sp, duration
        )

        if (
            _status(sp) == "FAIL"
            and _num(
                sp.get("confidence")
            ) >= 0.75
            and len(words) >= 2
        ):
            sec = words[0][0]

            explanation = "; ".join(
                (
                    f"{t:.1f}s '{word}' ({why})"
                )
                for t, word, why
                in words[:3]
            )

            fails.append((
                "audio",
                sec,
                "เสียงเพี้ยน >=2 คำ: "
                + explanation,
            ))

        elif (
            _status(sp) == "FAIL"
            and _num(
                sp.get("confidence")
            ) >= 0.80
            and _boolean(
                sp.get("gibberish")
            )
            and _evidence(
                sp.get(
                    "gibberish_evidence_th"
                )
            )
        ):
            fails.append((
                "audio",
                0,
                "เสียงมั่วต่อเนื่อง: "
                + _text(
                    sp.get(
                        "gibberish_evidence_th"
                    )
                ),
            ))

        elif (
            _status(sp) != "PASS"
            or len(words) >= 2
        ):
            needs.append("speech")

    # SPEECH / PRODUCT MATCH
    if "match" in scope:
        if (
            _status(ma) == "FAIL"
            and _num(
                ma.get("confidence")
            ) >= 0.78
            and _evidence(
                ma.get("spoken_claim_th")
            )
            and (
                _evidence(
                    ma.get("video_fact_th")
                )
                or (
                    reference
                    and _evidence(
                        ma.get("json_fact_th")
                    )
                )
            )
        ):
            fails.append((
                "match",
                0,
                "คำพูดไม่ตรงภาพ/สินค้า: "
                + _text(
                    ma.get("reason_th")
                    or ma.get("spoken_claim_th")
                ),
            ))

        elif _status(ma) != "PASS":
            needs.append("match")

    # VISUAL / HAND
    if "visual" in scope:
        events = _events(
            vi,
            duration,
            "visual",
        )

        if (
            _status(vi) == "FAIL"
            and _num(
                vi.get("confidence")
            ) >= 0.78
            and events
        ):
            sec, seen, why = events[0]

            fails.append((
                "visual",
                sec,
                f"ภาพผิดธรรมชาติ: {seen}; {why}",
            ))

        elif _status(vi) != "PASS":
            needs.append("visual")

    # MIRROR
    inferred_mirror = bool(re.search(
        r"กระจกเงา|กระจกส่อง|กระจกแต่งหน้า|\bmirror\b",
        _text(
            data.get("product_type_th"),
            180,
        ),
        re.I,
    ))

    mirror_applicable = (
        _boolean(
            mi.get("applicable")
        )
        or _mirror_claim(reference)
        or inferred_mirror
    )

    if (
        "mirror" in scope
        and mirror_applicable
    ):
        events = _mirrors(
            mi, duration
        )

        confirmed = [
            item
            for item in events
            if (
                item[2]
                and not _boolean(
                    mi.get("angle_uncertain")
                )
                and (
                    item[5] != "missing_hand_reflection"
                    or (
                        _boolean(
                            mi.get(
                                "reflection_expected"
                            )
                        )
                        and item[1] - item[0] > 2.0
                    )
                )
            )
        ]

        if (
            _status(mi) == "FAIL"
            and _num(
                mi.get("confidence")
            ) >= 0.82
            and confirmed
        ):
            item = confirmed[0]

            fails.append((
                "mirror",
                item[0],
                (
                    "เงาสะท้อนผิดธรรมชาติ: "
                    f"{item[3]}; {item[4]}"
                ),
            ))

        elif (
            _status(mi) != "PASS"
            or events
            or (
                inferred_mirror
                and not _boolean(
                    mi.get("applicable")
                )
            )
        ):
            needs.append("mirror")

    return (
        fails,
        list(dict.fromkeys(needs)),
    )


# ==========================================
# SMART RECHECK
# ==========================================

def _audit_focus(needs, path):
    if RECHECK_MODE == "strict":
        return {
            "scale",
            "speech",
            "match",
            "visual",
            "mirror",
        }

    if needs:
        return set(needs)

    if RECHECK_MODE == "smart":
        with open(path, "rb") as file:
            digest = hashlib.sha256(
                file.read(131072)
            ).digest()

        slot = (
            int.from_bytes(
                digest[:4],
                "big",
            ) % 100
        )

        if slot < CLEAN_SAMPLE_PERCENT:
            return {
                "scale",
                "speech",
                "match",
                "visual",
                "mirror",
            }

    return set()


# ==========================================
# MAIN
# ==========================================

def inspect(
    path,
    key,
    model,
    mode,
    thumbnail,
    reference="",
):
    started = time.perf_counter()

    path = Path(path)
    thumbnail = Path(thumbnail)

    duration, has_audio = _probe(path)
    fps = FPS.get(mode, 2)

    reference = _text(
        reference, 1300
    )

    with tempfile.TemporaryDirectory(
        prefix="qc39_"
    ) as temp:
        folder = Path(temp)

        frames = _extract(
            path,
            folder / "frames",
            fps,
        )

        sheets = _sheets(
            frames,
            fps,
            folder,
        )

        indices = sorted({
            round(
                (len(frames) - 1) * position
            )
            for position in (
                0.08,
                0.5,
                0.92,
            )
        })

        detail = [
            (
                i / fps,
                frames[i],
            )
            for i in indices
        ]

        thumbnail.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not thumbnail.is_file():
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

        wav = _audio(
            path,
            folder,
            has_audio,
        )

        prepared_at = time.perf_counter()

        # FIRST PASS

        first = _api(
            key,
            model,
            _parts(
                FIRST_PROMPT,
                reference,
                duration,
                sheets,
                detail,
                wav,
            ),
            3300,
        )

        first_at = time.perf_counter()

        initial_fail, initial_needs = _judge(
            first,
            duration,
            reference,
        )

        second = None
        focus = set()

        final_fail = initial_fail
        pending = list(initial_needs)

        # SECOND PASS ONLY WHEN NEEDED

        if not initial_fail:
            focus = _audit_focus(
                initial_needs,
                path,
            )

            if focus:
                image_focus = bool(
                    focus & {
                        "scale",
                        "match",
                        "visual",
                        "mirror",
                    }
                )

                audio_focus = (
                    "speech" in focus
                    or "match" in focus
                )

                if "mirror" in focus:
                    chosen_sheets = sheets

                else:
                    step = max(
                        1,
                        len(sheets) // 3,
                    )

                    chosen_sheets = (
                        sheets[::step][:3]
                    )

                if (
                    "scale" in focus
                    or "mirror" in focus
                ):
                    chosen_details = detail

                else:
                    chosen_details = detail[:1]

                instruction = (
                    AUDIT_PROMPT
                    + "\nตรวจเฉพาะด้าน: "
                    + ", ".join(sorted(focus))
                )

                second = _api(
                    key,
                    model,
                    _parts(
                        instruction,
                        reference,
                        duration,
                        (
                            chosen_sheets
                            if image_focus
                            else []
                        ),
                        (
                            chosen_details
                            if image_focus
                            else []
                        ),
                        (
                            wav
                            if audio_focus
                            else None
                        ),
                    ),
                    2900,
                )

                final_fail, pending = _judge(
                    second,
                    duration,
                    reference,
                    scope=focus,
                )

                unresolved = set(pending)

                for name in initial_needs:
                    if name not in focus:
                        unresolved.add(name)
                        continue

                    second_status = _status(
                        second.get(name, {})
                    )

                    if second_status == "REVIEW":
                        unresolved.add(name)

                pending = sorted(unresolved)

        if final_fail:
            decision = "FAIL"

        elif pending:
            decision = "REVIEW"

        else:
            decision = "PASS"

        reasons = [
            issue[2]
            for issue in final_fail
        ]

        if not reasons and pending:
            reasons = [
                "รอตรวจด้วยคน: "
                + ", ".join(pending)
                + " ยังไม่มีหลักฐานให้อนุมัติ"
                "หรือปฏิเสธแน่นอน"
            ]

        if not reasons:
            reasons = [
                _text(
                    first.get("summary_th"),
                    700,
                )
                or "ไม่พบข้อผิดพลาดสำคัญตามภาพและเสียง"
            ]

        issues = [
            {
                "category": category,
                "time_sec": round(sec, 2),
                "detail_th": why,
                "severity": "critical",
            }
            for category, sec, why
            in final_fail
        ]

        if pending and not final_fail:
            issues.append({
                "category": "human_review",
                "time_sec": 0,
                "detail_th": reasons[0],
                "severity": "warning",
            })

        source = (
            second
            if second and final_fail
            else first
        )

        ended = time.perf_counter()

        return {
            "decision": decision,

            "reason_th": (
                "; ".join(reasons)[:850]
            ),

            "video_type": (
                _text(
                    source.get("video_type"),
                    30,
                ).upper()
                or "PRODUCT"
            ),

            "product_name_th": _text(
                source.get("visible_product_th"),
                180,
            ),

            "product_type_th": _text(
                source.get("product_type_th"),
                180,
            ),

            "spoken_product_th": _text(
                source.get("spoken_product_th"),
                180,
            ),

            "speech_summary_th": _text(
                first.get("speech_summary_th"),
                700,
            ),

            "scale_review": source["scale"],
            "audio_review": source["speech"],
            "match_review": source["match"],
            "visual_review": source["visual"],
            "mirror_review": source["mirror"],

            "first_scale_review": first["scale"],
            "first_audio_review": first["speech"],

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

            "mirror_audit": (
                second["mirror"]
                if second else None
            ),

            "issues": issues,

            "audit_done": (
                second is not None
            ),

            "audit_trigger": (
                ", ".join(sorted(focus))
                if focus else "none"
            ),

            "review_required": (
                decision == "REVIEW"
            ),

            "product_reference_used": bool(
                reference
            ),

            "duration": round(
                duration, 2
            ),

            "elapsed": round(
                ended - started,
                2,
            ),

            "timings": {
                "prepare_seconds": round(
                    prepared_at - started,
                    2,
                ),

                "first_api_seconds": round(
                    first_at - prepared_at,
                    2,
                ),

                "audit_api_seconds": (
                    round(
                        ended - first_at,
                        2,
                    )
                    if second else 0
                ),
            },

            "engine": ENGINE_VERSION,
        }
