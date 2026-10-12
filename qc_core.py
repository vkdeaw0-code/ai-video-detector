
"""V3.9.1 MP4-only video QC: evidence-first, fast single pass, selective audit."""
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

ENGINE_VERSION = "3.9.1-mp4-only-20261012"

MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
)

DEFAULT_MODEL = MODELS[0]
ALLOWED_MODELS = set(MODELS)

FPS = {
    "Fast": 1,
    "Balanced": 2,
    "Deep": 3,
}

FRAME_EDGE = 640
MAX_SECONDS = 35


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


def available_models(api_key: str) -> list[str]:
    """ตรวจสอบโมเดลที่ API Key สามารถมองเห็นผ่าน Google models.list"""
    if not api_key:
        return []

    names = set()
    token = None

    try:
        for _ in range(4):
            params = {"pageSize": 100}

            if token:
                params["pageToken"] = token

            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key},
                params=params,
                timeout=(8, 12),
            )

            if response.status_code >= 400:
                raise AnalysisError(
                    f"ตรวจสอบโมเดล HTTP {response.status_code}: "
                    f"{response.text[:180]}"
                )

            result = response.json()

            for item in result.get("models", []):
                methods = item.get(
                    "supportedGenerationMethods", []
                )

                name = str(
                    item.get("name", "")
                ).removeprefix("models/")

                if (
                    "generateContent" in methods
                    and name in MODELS
                ):
                    names.add(name)

            token = result.get("nextPageToken")

            if not token:
                break

    except (
        requests.RequestException,
        ValueError,
    ) as exc:
        raise AnalysisError(
            f"ตรวจสอบโมเดลกับ Google ไม่ได้: {exc}"
        ) from exc

    return [
        m for m in MODELS
        if m in names
    ]


def _text(value, n=300):
    return str(value or "").strip()[:n]


def _float(value, default=0.0):
    try:
        n = float(value)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


def _truth(value):
    return (
        value is True
        or (
            isinstance(value, str)
            and value.lower() == "true"
        )
    )


def _status(section):
    s = _text(
        section.get("status")
        if isinstance(section, dict)
        else ""
    ).upper()

    return (
        s if s in {"PASS", "FAIL", "REVIEW"}
        else "REVIEW"
    )


def _proof(value):
    s = _text(value).lower()

    return (
        len(s) >= 10
        and not s.startswith((
            "ไม่มี",
            "ไม่พบ",
            "ไม่แน่ใจ",
            "ไม่ชัด",
            "ไม่สามารถ",
            "ปกติ",
            "unknown",
            "none",
            "not visible",
            "uncertain",
        ))
    )


def _run(args, timeout=85):
    try:
        proc = subprocess.run(
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
            f"เปิด/แปลงวิดีโอด้วย FFmpeg ไม่ได้: {exc}"
        ) from exc

    if proc.returncode:
        raise AnalysisError(
            "FFmpeg: "
            + (proc.stderr or "")[-450:]
        )

    return proc.stdout


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
        info = json.loads(raw)
        duration = float(
            info["format"]["duration"]
        )

        tracks = {
            x.get("codec_type")
            for x in info["streams"]
        }

    except (
        KeyError,
        ValueError,
        TypeError,
    ) as exc:
        raise AnalysisError(
            "อ่านความยาวหรือสตรีมวิดีโอไม่ได้"
        ) from exc

    if (
        not math.isfinite(duration)
        or not 0.5 <= duration <= MAX_SECONDS
    ):
        raise AnalysisError(
            f"รองรับคลิปยาว 0.5–{MAX_SECONDS} วินาที"
        )

    if "video" not in tracks:
        raise AnalysisError(
            "ไม่พบภาพวิดีโอ"
        )

    return duration, "audio" in tracks


def _extract(video, folder, fps):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    vf = (
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
        "-vf", vf,
        "-q:v", "5",
        str(folder / "frame_%03d.jpg"),
    ], 120)

    frames = sorted(
        folder.glob("frame_*.jpg")
    )

    if not frames:
        raise AnalysisError(
            "ดึงภาพจากคลิปไม่สำเร็จ"
        )

    return frames


def _sheets(frames, fps, folder):
    output = []

    for start in range(
        0, len(frames), 6
    ):
        canvas = Image.new(
            "RGB",
            (990, 560),
            "#171c27",
        )

        draw = ImageDraw.Draw(canvas)

        for i, src in enumerate(
            frames[start:start + 6]
        ):
            with Image.open(src) as image:
                thumb = ImageOps.contain(
                    image.convert("RGB"),
                    (320, 240),
                )

            x = (
                (i % 3) * 330
                + (330 - thumb.width) // 2
            )

            y = (
                (i // 3) * 280
                + 26
                + (240 - thumb.height) // 2
            )

            canvas.paste(thumb, (x, y))

            draw.text(
                (
                    (i % 3) * 330 + 7,
                    (i // 3) * 280 + 6,
                ),
                f"{(start + i) / fps:.1f}s",
                fill="white",
            )

        dest = folder / (
            f"sheet_{len(output):02d}.jpg"
        )

        canvas.save(
            dest,
            quality=76,
            optimize=True,
        )

        canvas.close()
        output.append(dest)

    return output


def _audio(video, folder, has_audio):
    if not has_audio:
        return None

    dest = folder / "voice.wav"

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
        str(dest),
    ], 70)

    with wave.open(
        str(dest), "rb"
    ) as audio:
        content = audio.readframes(
            audio.getnframes()
        )

    samples = array("h")
    samples.frombytes(content)

    if not samples:
        return None

    rms = math.sqrt(
        sum(v * v for v in samples)
        / len(samples)
    ) / 32768

    return dest if rms >= 0.0008 else None


def _inline(path, mime):
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(
                Path(path).read_bytes()
            ).decode("ascii"),
        }
    }


SCHEMA = r'''
{
 "product_type_th":"ประเภทสินค้า",
 "product_seen_th":"สินค้าที่เห็น",
 "product_spoken_th":"สินค้าที่พูดถึง",
 "speech_summary_th":"คำพูดโดยสรุป",
 "scale":{
   "status":"PASS",
   "confidence":0.0,
   "anchor_th":"มือ/คน/สิ่งแวดล้อมที่ใช้เทียบ",
   "mini_correct":false,
   "events":[
     {
       "time_sec":1.0,
       "seen_th":"สิ่งที่เห็น",
       "problem_th":"เหตุผลว่าผิดจริงอย่างไร"
     }
   ],
   "reason_th":""
 },
 "speech":{
   "status":"PASS",
   "confidence":0.0,
   "has_speech":true,
   "wrong_words":[
     {
       "time_sec":1.0,
       "heard_th":"คำที่ได้ยิน",
       "problem_th":"ผิดอย่างไร"
     }
   ],
   "gibberish":false,
   "gibberish_evidence_th":"",
   "reason_th":""
 },
 "match":{
   "status":"PASS",
   "confidence":0.0,
   "spoken_claim_th":"",
   "visual_fact_th":"",
   "reason_th":""
 },
 "visual":{
   "status":"PASS",
   "confidence":0.0,
   "events":[
     {
       "time_sec":1.0,
       "seen_th":"",
       "problem_th":""
     }
   ],
   "reason_th":""
 },
 "mirror":{
   "applicable":false,
   "status":"PASS",
   "confidence":0.0,
   "reflection_should_be_visible":false,
   "angle_uncertain":false,
   "events":[
     {
       "type":"missing_hand_reflection",
       "start_sec":1.0,
       "end_sec":3.2,
       "observed_th":"",
       "geometry_th":"",
       "sample_times_sec":[1.0,2.0,3.2]
     }
   ],
   "reason_th":""
 },
 "important_uncertainties":[],
 "summary_th":"ภาพรวมสั้นๆ"
}
'''


FIRST_PROMPT = r'''
คุณเป็นผู้ตรวจ QC โฆษณาสินค้าที่สร้างด้วย AI
วัตถุประสงค์คือกันคลิปผิดรุนแรง
แต่ให้สินค้าปกติผ่านได้
อย่าบังคับให้ FAIL ตามอัตราส่วนที่กำหนด
ห้ามเดาหลักฐาน

ให้สังเกตภาพทุกช่วงเวลาที่ให้และฟัง WAV จริง
ไม่จำเป็นต้องมี JSON

ก่อนตรวจ จำแนกชนิดสินค้าที่ขายจากวัตถุหลัก
และคำพูดอย่างละเอียด

ไม่จำกัดประเภท:
กีฬา เฟอร์นิเจอร์ เครื่องครัว กระจก
เสื้อผ้า รองเท้า เครื่องใช้ไฟฟ้า
ของเล่น เครื่องมือ ความงาม อาหาร
อุปกรณ์เด็ก สัตว์เลี้ยง ฯลฯ

เลือกหลักการใช้งานเฉพาะประเภทสินค้านั้น
อย่าตรวจสเกลของทุกสินค้าด้วยกฎเดียว

1 สเกลและการใช้งาน:
- เทียบกับนิ้ว มือ คนและวัตถุระนาบเดียวกัน
- สังเกตสัดส่วน
- ของจิ๋วหลอกเป็นสินค้าขนาดใช้งานจริง
- สินค้าขยาย/หดผิดธรรมชาติ
- ใช้งานไม่ได้ตามที่พูด
- พิจารณาการพับ/หดเสา มุมกล้อง
- สินค้า MINI ที่เป็นของจริง
- อย่าเดาขนาดเป็น cm จากภาพ
- ถ้าเสียงบอกขนาดให้ตรวจความสมเหตุสมผล
- ตัวเลขที่ตรวจไม่ได้ต้อง REVIEW

2 เสียง:
- ฟังภาษาไทยและคำพูดทุกช่วง
- ผิดเสียงจริง >=2 ตำแหน่งคำให้ FAIL
  เมื่อระบุคำและเวลาได้
- พูดมั่วต่อเนื่องจนฟังไม่ได้ FAIL
- คำผิดเล็กน้อย 1 คำ สำเนียง
  หรือชื่อแบรนด์ ไม่ FAIL
- ห้ามแต่งคำผิด
- แยกคำผิดจริงจากสำเนียง

3 ความสัมพันธ์เสียงกับภาพ:
- พูดขายสินค้าอื่น
- บอกสี จำนวน การใช้ ลักษณะ
  ซึ่งขัดกับสิ่งที่เห็นอย่างชัด ให้ FAIL
- หากยืนยันด้วยภาพไม่ได้ให้ REVIEW
- ไม่ถือว่าไม่มี JSON เป็นข้อผิดพลาด

4 ภาพ/มือ/สินค้าหลัก:
- นิ้วผิดรุนแรง
- มือทะลุสินค้า
- มือหายขณะจับ
- สินค้าเปลี่ยนชนิด
- สินค้าละลาย
- บิดจนผิดหน้าที่
- ใช้งานผิดฟิสิกส์รุนแรง
สิ่งเหล่านี้ให้ FAIL เมื่อเห็นหลักฐานชัด

AI artifact เล็กน้อยที่ไม่กระทบสินค้าให้ผ่าน

5 กระจกเงาที่เป็นสินค้าหลัก:
- ดูว่ามือหรือวัตถุควรสะท้อน
  ตามมุมเรขาคณิตหรือไม่
- ถ้าหายเกิน 2 วินาที
  ต้องมีตัวอย่างภาพหลายช่วงเวลาที่พิสูจน์จริง
- ไม่ใช่เดาจากเฟรมเดียว
- ภาพสะท้อนบิดผิดธรรมชาติรุนแรงก็ผิด
- ถ้ามุมกล้องไม่แน่ใจ REVIEW
- สินค้าแวววาวไม่ใช่กระจกเงาโดยอัตโนมัติ

ตัดสินแต่ละด้าน status=PASS/FAIL/REVIEW
confidence=ความมั่นใจของโมเดล
ไม่ใช่หลักฐานด้วยตัวเอง

ระบุ time_sec และข้อสังเกตจริง
หาก FAIL หรือ REVIEW

ไม่มีเหตุการณ์ให้ events=[]
ไม่มีคำผิดให้ wrong_words=[]

อย่าใส่ค่าตัวอย่างใน JSON จริง
ต้องตอบ JSON object เท่านั้น ตาม schema:
''' + SCHEMA


SECOND_PROMPT = r'''
คุณเป็นผู้ตรวจคุณภาพวิดีโออิสระรอบยืนยัน
ตรวจเพียงหัวข้อที่ระบุ

ห้ามอาศัยผลรอบแรก
ไม่สร้างคำผิดหรือหลักฐานที่มองไม่เห็นเอง

การผิดสเกลต้องมีมือ/คน/สภาพแวดล้อม
และข้อขัดแย้งจริง

คำพูดต้องเทียบสินค้าที่เห็นและการใช้จริง
เสียงผิด >=2 คำต้องได้ยินและมีเวลา

สินค้ากระจกต้องพิจารณามุมสะท้อน
และหลายเฟรมก่อนสรุปว่ามือหาย >2s

หากไม่ยืนยันได้ให้ REVIEW
คลิปที่ไม่มีข้อผิดพลาดสำคัญให้ PASS

ตอบ JSON ตาม schema เดิมทุกหัวข้อ
หัวข้อที่ไม่ได้ตรวจให้ PASS/[]:
''' + SCHEMA


def _parts(
    prompt,
    duration,
    sheets,
    details,
    wav,
):
    parts = [{
        "text": (
            f"{prompt}\n"
            f"ความยาววิดีโอ {duration:.2f} วินาที"
        )
    }]

    for path in sheets:
        parts.extend([
            {
                "text": (
                    "ภาพตามลำดับเวลา "
                    "มี timestamp บนแต่ละเฟรม"
                )
            },
            _inline(path, "image/jpeg"),
        ])

    for sec, path in details:
        parts.extend([
            {
                "text": f"ภาพขยาย {sec:.1f}s"
            },
            _inline(path, "image/jpeg"),
        ])

    if wav is not None:
        parts.extend([
            {
                "text": "เสียงจริงทั้งคลิป ฟังทุกคำ"
            },
            _inline(wav, "audio/wav"),
        ])

    else:
        parts.append({
            "text": (
                "คลิปไม่มีเสียงที่ตรวจได้ "
                "ห้ามแต่งคำพูดหรือบังคับให้ผิดเสียง"
            )
        })

    return parts


def _api(
    api_key,
    model,
    parts,
    max_tokens=3000,
):
    model = (
        model if model in MODELS
        else DEFAULT_MODEL
    )

    if not api_key:
        raise AnalysisError(
            "ไม่พบ GEMINI_API_KEY"
        )

    models_to_try = [model]

    for selected in models_to_try:
        url = (
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/{selected}:generateContent"
        )

        request = {
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
                        "x-goog-api-key": api_key,
                    },
                    json=request,
                    timeout=(12, 105),
                )

            except requests.RequestException as exc:
                if attempt == 0:
                    time.sleep(1)
                    continue

                raise AnalysisError(
                    f"เชื่อมต่อ Gemini ไม่ได้: {exc}"
                ) from exc

            if response.status_code == 429:
                raise QuotaError(
                    "Gemini API ถึงโควตา 429; "
                    "รอแล้วจึงกดตรวจใหม่"
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
                    message = response.json().get(
                        "error", {}
                    ).get(
                        "message",
                        response.text[:220],
                    )

                except ValueError:
                    message = response.text[:220]

                # Model 404:
                # Try another model visible to this key.
                if (
                    response.status_code == 404
                    and selected == model
                ):
                    try:
                        for replacement in available_models(api_key):
                            if replacement not in models_to_try:
                                models_to_try.append(
                                    replacement
                                )

                    except AnalysisError:
                        pass

                    if len(models_to_try) > 1:
                        break

                raise AnalysisError(
                    f"Gemini HTTP {response.status_code} "
                    f"({selected}): {message[:320]}"
                )

            try:
                candidate = response.json()[
                    "candidates"
                ][0]

                raw = "".join(
                    p.get("text", "")
                    for p in candidate[
                        "content"
                    ]["parts"]
                ).strip()

                raw = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    raw,
                )

                parsed = json.loads(raw)

                required = (
                    "scale",
                    "speech",
                    "match",
                    "visual",
                    "mirror",
                )

                if (
                    not isinstance(parsed, dict)
                    or not all(
                        isinstance(parsed.get(k), dict)
                        for k in required
                    )
                ):
                    raise ValueError(
                        "คำตอบขาดหัวข้อ QC บางส่วน"
                    )

                return parsed

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                raise AnalysisError(
                    f"Gemini ส่ง JSON ไม่สมบูรณ์: {exc}"
                ) from exc

    raise AnalysisError(
        "ไม่สามารถตรวจผ่าน Gemini ได้"
    )


def _events(section, duration):
    raw = section.get("events", [])

    if not isinstance(raw, list):
        return []

    valid = []

    for e in raw[:12]:
        if not isinstance(e, dict):
            continue

        t = _float(
            e.get("time_sec"), -1
        )

        seen = _text(
            e.get("seen_th"), 220
        )

        wrong = _text(
            e.get("problem_th"), 250
        )

        if (
            0 <= t <= duration
            and _proof(seen)
            and _proof(wrong)
        ):
            valid.append((
                t, seen, wrong
            ))

    return valid


def _words(section, duration):
    raw = section.get(
        "wrong_words", []
    )

    if not isinstance(raw, list):
        return []

    found = []

    for item in raw[:15]:
        if not isinstance(item, dict):
            continue

        t = _float(
            item.get("time_sec"), -1
        )

        heard = _text(
            item.get("heard_th"), 70
        )

        why = _text(
            item.get("problem_th"), 190
        )

        if (
            0 <= t <= duration
            and heard
            and _proof(why)
            and all(
                abs(t - x[0]) > 0.18
                for x in found
            )
        ):
            found.append((
                t, heard, why
            ))

    return found


def _mirror_events(section, duration):
    raw = section.get(
        "events", []
    )

    if not isinstance(raw, list):
        return []

    found = []

    kinds = {
        "missing_hand_reflection",
        "severe_warp",
        "impossible_reflection",
        "unrelated_motion",
    }

    for e in raw[:10]:
        if (
            not isinstance(e, dict)
            or e.get("type") not in kinds
        ):
            continue

        start = _float(
            e.get("start_sec"), -1
        )

        end = _float(
            e.get("end_sec"), -1
        )

        seen = _text(
            e.get("observed_th"), 260
        )

        geo = _text(
            e.get("geometry_th"), 260
        )

        stamps = e.get(
            "sample_times_sec", []
        )

        if not isinstance(stamps, list):
            stamps = []

        valid_stamps = sorted({
            round(_float(t, -1), 1)
            for t in stamps
            if (
                start - 0.5
                <= _float(t, -1)
                <= end + 0.5
            )
        })

        if (
            0 <= start < end <= duration
            and _proof(seen)
            and _proof(geo)
        ):
            found.append({
                "kind": e["type"],
                "time": start,
                "duration": end - start,
                "proof": seen + "; " + geo,
                "multiple": (
                    len(valid_stamps) >= 3
                    and valid_stamps[-1] - valid_stamps[0] >= 1.8
                ),
            })

    return found


def _is_mirror(data):
    label = _text(
        data.get("product_type_th"),
        120,
    )

    return (
        bool(re.search(
            r"กระจกเงา|กระจกส่อง|กระจกแต่งหน้า|mirror",
            label,
            re.I,
        ))
        or _truth(
            data["mirror"].get("applicable")
        )
    )


def _judge(
    data,
    duration,
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

    fail = []
    review = []

    sc, sp, ma, vi, mi = (
        data[k]
        for k in (
            "scale",
            "speech",
            "match",
            "visual",
            "mirror",
        )
    )

    # SCALE
    if "scale" in scope:
        events = _events(
            sc, duration
        )

        anchor = _proof(
            sc.get("anchor_th")
        )

        if (
            _status(sc) == "FAIL"
            and _float(
                sc.get("confidence")
            ) >= 0.77
            and events
            and (
                anchor
                or len(events) >= 2
            )
            and not _truth(
                sc.get("mini_correct")
            )
        ):
            t, seen, reason = events[0]

            fail.append((
                "scale",
                t,
                (
                    f"สเกล/การใช้งานผิด: "
                    f"{seen}; {reason}"
                ),
            ))

        elif (
            _status(sc) != "PASS"
            or (
                events
                and not _truth(
                    sc.get("mini_correct")
                )
            )
        ):
            review.append("scale")

    # SPEECH
    if "speech" in scope:
        words = _words(
            sp, duration
        )

        if (
            _status(sp) == "FAIL"
            and _float(
                sp.get("confidence")
            ) >= 0.75
            and len(words) >= 2
        ):
            fail.append((
                "speech",
                words[0][0],
                (
                    "เสียงออกผิด >=2 ตำแหน่ง: "
                    + "; ".join(
                        f"{t:.1f}s {w} ({why})"
                        for t, w, why
                        in words[:3]
                    )
                ),
            ))

        elif (
            _status(sp) == "FAIL"
            and _float(
                sp.get("confidence")
            ) >= 0.80
            and _truth(
                sp.get("gibberish")
            )
            and _proof(
                sp.get("gibberish_evidence_th")
            )
        ):
            fail.append((
                "speech",
                0,
                (
                    "เสียงมั่วต่อเนื่อง: "
                    + _text(
                        sp.get(
                            "gibberish_evidence_th"
                        )
                    )
                ),
            ))

        elif (
            _status(sp) != "PASS"
            or len(words) >= 2
        ):
            review.append("speech")

    # SPEECH / IMAGE MATCH
    if "match" in scope:
        if (
            _status(ma) == "FAIL"
            and _float(
                ma.get("confidence")
            ) >= 0.77
            and _proof(
                ma.get("spoken_claim_th")
            )
            and _proof(
                ma.get("visual_fact_th")
            )
        ):
            fail.append((
                "match",
                0,
                (
                    "คำพูดกับภาพไม่ตรง: "
                    + _text(
                        ma.get("reason_th")
                        or ma.get("spoken_claim_th")
                    )
                ),
            ))

        elif _status(ma) != "PASS":
            review.append("match")

    # VISUAL
    if "visual" in scope:
        events = _events(
            vi, duration
        )

        if (
            _status(vi) == "FAIL"
            and _float(
                vi.get("confidence")
            ) >= 0.78
            and events
        ):
            t, seen, reason = events[0]

            fail.append((
                "visual",
                t,
                (
                    "สินค้า/มือผิดธรรมชาติรุนแรง: "
                    f"{seen}; {reason}"
                ),
            ))

        elif (
            _status(vi) != "PASS"
            or events
        ):
            review.append("visual")

    # MIRROR
    if (
        "mirror" in scope
        and _is_mirror(data)
    ):
        events = _mirror_events(
            mi, duration
        )

        confirmed = [
            e for e in events
            if (
                e["multiple"]
                and not _truth(
                    mi.get("angle_uncertain")
                )
                and (
                    e["kind"] != "missing_hand_reflection"
                    or (
                        e["duration"] > 2
                        and _truth(
                            mi.get(
                                "reflection_should_be_visible"
                            )
                        )
                    )
                )
            )
        ]

        if (
            _status(mi) == "FAIL"
            and _float(
                mi.get("confidence")
            ) >= 0.82
            and confirmed
        ):
            e = confirmed[0]

            fail.append((
                "mirror",
                e["time"],
                (
                    "กระจก/เงาสะท้อนผิด: "
                    + e["proof"]
                ),
            ))

        elif (
            _status(mi) != "PASS"
            or events
        ):
            review.append("mirror")

    # IMPORTANT UNCERTAINTY
    uncertainties = data.get(
        "important_uncertainties", []
    )

    if (
        isinstance(uncertainties, list)
        and any(
            _proof(s)
            for s in uncertainties[:5]
        )
    ):
        review.append("uncertainty")

    return (
        fail,
        list(dict.fromkeys(review)),
    )


def inspect(
    path,
    key,
    model,
    mode,
    thumbnail,
    reference="",
):
    """
    Compatible with the existing queue signature.
    JSON reference is intentionally ignored.
    """
    started = time.perf_counter()

    video = Path(path)
    thumb = Path(thumbnail)

    duration, has_audio = _probe(video)
    fps = FPS.get(mode, 2)

    with tempfile.TemporaryDirectory(
        prefix="qc391_"
    ) as td:
        folder = Path(td)

        frames = _extract(
            video,
            folder / "frames",
            fps,
        )

        sheets = _sheets(
            frames, fps, folder
        )

        indices = sorted({
            round(
                (len(frames) - 1) * i
            )
            for i in (
                0.1, 0.5, 0.9
            )
        })

        details = [
            (i / fps, frames[i])
            for i in indices
        ]

        thumb.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not thumb.exists():
            with Image.open(
                frames[0]
            ) as raw:
                ImageOps.contain(
                    raw.convert("RGB"),
                    (320, 320),
                ).save(
                    thumb,
                    "JPEG",
                    quality=73,
                )

        wav = _audio(
            video,
            folder,
            has_audio,
        )

        prepared = time.perf_counter()

        # FIRST PASS
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
            3100,
        )

        first_end = time.perf_counter()

        fails, risks = _judge(
            first, duration
        )

        second = None
        focus = set()

        final_fails = fails[:]
        pending = risks[:]

        # SECOND PASS ONLY FOR RISKS
        if not fails and risks:
            focus = {
                r for r in risks
                if r in {
                    "scale",
                    "speech",
                    "match",
                    "visual",
                    "mirror",
                }
            }

            if (
                "uncertainty" in risks
                and not focus
            ):
                focus = {
                    "scale", "match"
                }

            image_needed = bool(
                focus & {
                    "scale",
                    "match",
                    "visual",
                    "mirror",
                }
            )

            audio_needed = bool(
                focus & {
                    "speech", "match"
                }
            )

            if "mirror" in focus:
                chosen = sheets
            else:
                step = max(
                    1,
                    len(sheets) // 3,
                )

                chosen = sheets[::step][:3]

            second = _api(
                key,
                model,
                _parts(
                    (
                        SECOND_PROMPT
                        + "\nจุดที่ต้องตรวจ: "
                        + ", ".join(sorted(focus))
                    ),
                    duration,
                    (
                        chosen
                        if image_needed
                        else []
                    ),
                    (
                        details
                        if "mirror" in focus
                        else (
                            details[:2]
                            if image_needed
                            else []
                        )
                    ),
                    (
                        wav
                        if audio_needed
                        else None
                    ),
                ),
                2600,
            )

            final_fails, newer_risks = _judge(
                second,
                duration,
                focus,
            )

            pending = list(newer_risks)

            # Do not silently erase
            # a well-evidenced concern.
            for area in focus:
                block = first.get(
                    area, {}
                )

                if (
                    _status(block) == "REVIEW"
                    and _status(
                        second.get(area, {})
                    ) == "PASS"
                ):
                    if (
                        area in ("scale", "visual")
                        and _events(block, duration)
                    ):
                        pending.append(area)

                    if (
                        area == "mirror"
                        and _mirror_events(
                            block, duration
                        )
                    ):
                        pending.append(area)

            pending = list(
                dict.fromkeys(pending)
            )

        if final_fails:
            decision = "FAIL"
        elif pending:
            decision = "REVIEW"
        else:
            decision = "PASS"

        if final_fails:
            reason = "; ".join(
                x[2]
                for x in final_fails[:3]
            )

        elif pending:
            reason = (
                "รอตรวจด้วยคน: "
                "ยังไม่ยืนยันประเด็น "
                + ", ".join(pending)
            )

        else:
            reason = (
                _text(
                    first.get("summary_th"),
                    700,
                )
                or (
                    "ไม่พบข้อผิดพลาดสำคัญ"
                    "จากภาพและเสียง"
                )
            )

        source = (
            second
            if final_fails and second
            else first
        )

        ended = time.perf_counter()

        issues = [
            {
                "category": cat,
                "time_sec": round(t, 2),
                "detail_th": why,
                "severity": "critical",
            }
            for cat, t, why
            in final_fails
        ]

        if decision == "REVIEW":
            issues.append({
                "category": "human_review",
                "time_sec": 0,
                "detail_th": reason,
                "severity": "warning",
            })

        return {
            "decision": decision,
            "reason_th": reason[:850],

            "product_type_th": _text(
                source.get("product_type_th"),
                160,
            ),

            "product_name_th": _text(
                source.get("product_seen_th"),
                160,
            ),

            "spoken_product_th": _text(
                source.get("product_spoken_th"),
                160,
            ),

            "speech_summary_th": _text(
                first.get("speech_summary_th"),
                650,
            ),

            "video_type": "PRODUCT",
            "scale_review": source["scale"],
            "audio_review": source["speech"],
            "match_review": source["match"],
            "visual_review": source["visual"],
            "mirror_review": source["mirror"],

            "scale_audit": (
                second.get("scale")
                if second else None
            ),

            "speech_audit": (
                second.get("speech")
                if second else None
            ),

            "mirror_audit": (
                second.get("mirror")
                if second else None
            ),

            "issues": issues,

            "review_required": (
                decision == "REVIEW"
            ),

            "audit_done": (
                second is not None
            ),

            "audit_trigger": (
                ",".join(sorted(focus))
                or "none"
            ),

            "product_reference_used": False,
            "duration": round(duration, 2),
            "elapsed": round(
                ended - started, 2
            ),

            "engine": ENGINE_VERSION,

            "timings": {
                "prepare_seconds": round(
                    prepared - started, 2
                ),

                "first_api_seconds": round(
                    first_end - prepared, 2
                ),

                "audit_api_seconds": (
                    round(
                        ended - first_end, 2
                    )
                    if second
                    else 0.0
                ),
            },
        }
