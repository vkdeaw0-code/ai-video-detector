
"""V3.9.3 Evidence-First / MP4 Only.

Drop-in qc_core.py for the V3.9.1 MP4-only app and qc_store.py.
Keep: inspect(path, key, model, mode, thumbnail, reference="").
Decisions: PASS, FAIL, REVIEW. One Gemini call for clean clips;
a focused second call only for evidence gaps / suspected problems / safety samples.

This is not a calibrated forensic measurement or a guaranteed defect detector.
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
from PIL import Image, ImageChops, ImageDraw, ImageOps

ENGINE_VERSION = "3.9.3-evidence-first-mp4-20261012-r1"
MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
)
DEFAULT_MODEL = MODELS[0]
ALLOWED_MODELS = set(MODELS)
FPS = {"Fast": 2, "Balanced": 4, "Deep": 5}
FRAME_EDGE = 768
MAX_SECONDS = 35
AUDIT_FPS = 5
AUDIT_WINDOW = 2.8

# Normal clips use only one Gemini call. A small reproducible sample of
# clean passes receives a second, independent challenge. Raise these values
# only after measuring actual accuracy and speed on your labeled clips.
CLEAN_AUDIT_PERCENT = 8
INTERACTION_AUDIT_PERCENT = 18


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


def available_models(api_key: str) -> list[str]:
    """List configured models reported by this API key (not a quota test)."""
    if not api_key:
        return []
    found, token = set(), None
    for _ in range(5):
        params = {"pageSize": 100}
        if token:
            params["pageToken"] = token
        try:
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key},
                params=params,
                timeout=(8, 15),
            )
        except requests.RequestException as exc:
            raise AnalysisError(f"ตรวจชื่อโมเดลไม่สำเร็จ: {exc}") from exc
        if response.status_code != 200:
            raise AnalysisError(
                f"ตรวจรายชื่อโมเดล HTTP {response.status_code}: "
                + response.text[:180]
            )
        try:
            obj = response.json()
        except ValueError as exc:
            raise AnalysisError("Google ส่งรายชื่อโมเดลไม่ถูกต้อง") from exc
        for item in obj.get("models", []):
            name = str(item.get("name", "")).removeprefix("models/")
            if name in MODELS and "generateContent" in item.get(
                "supportedGenerationMethods", []
            ):
                found.add(name)
        token = obj.get("nextPageToken")
        if not token:
            break
    return [model for model in MODELS if model in found]


def _str(value, limit=400):
    return str(value or "").strip()[:limit]


def _float(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _yes(value):
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _state(section):
    if not isinstance(section, dict):
        return "REVIEW"
    value = _str(section.get("status", section.get("result", "")), 15).upper()
    return value if value in {"PASS", "FAIL", "REVIEW"} else "REVIEW"


def _evidence(value, minimum=10):
    s = _str(value).lower()
    if len(s) < minimum:
        return False
    if s.startswith((
        "ไม่มี", "ไม่พบ", "ไม่ชัด", "ไม่แน่ใจ", "ยืนยันไม่ได้",
        "n/a", "none", "unknown", "not visible", "no evidence",
        "ทุกอย่างปกติ", "ตรวจแล้วปกติ", "ปกติทั้งหมด",
    )):
        return False
    return True


def _run(command, timeout=100):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            check=False, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AnalysisError(f"FFmpeg/FFprobe เปิดไฟล์ไม่ได้: {exc}") from exc
    if result.returncode:
        raise AnalysisError("FFmpeg: " + (result.stderr or "")[-450:])
    return result.stdout


def _probe(video):
    raw = _run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type", "-of", "json", str(video),
    ], 25)
    try:
        obj = json.loads(raw)
        duration = float(obj["format"]["duration"])
        streams = {x.get("codec_type") for x in obj["streams"]}
    except (ValueError, KeyError, TypeError) as exc:
        raise AnalysisError("ไม่สามารถอ่านข้อมูลวิดีโอ") from exc
    if not math.isfinite(duration) or not 0.5 <= duration <= MAX_SECONDS:
        raise AnalysisError(f"รองรับคลิป 0.5–{MAX_SECONDS} วินาที")
    if "video" not in streams:
        raise AnalysisError("ไฟล์ไม่มีภาพวิดีโอ")
    return duration, "audio" in streams


def _frames(video, folder, fps, start=0.0, length=None):
    folder.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "2"]
    if start > 0:
        cmd.extend(["-ss", f"{start:.3f}"])
    cmd.extend(["-i", str(video)])
    if length is not None:
        cmd.extend(["-t", f"{length:.3f}"])
    cmd.extend([
        "-vf", f"fps={fps},scale={FRAME_EDGE}:{FRAME_EDGE}:"
               "force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-q:v", "6", str(folder / "frame_%03d.jpg"),
    ])
    _run(cmd, 125)
    files = sorted(folder.glob("frame_*.jpg"))
    if not files:
        raise AnalysisError("ดึงภาพวิดีโอไม่สำเร็จ")
    return files


def _sheets(frames, fps, folder, start=0.0, prefix="sheet"):
    result = []
    for offset in range(0, len(frames), 6):
        sheet = Image.new("RGB", (990, 550), (23, 28, 36))
        pen = ImageDraw.Draw(sheet)
        for j, path in enumerate(frames[offset:offset + 6]):
            with Image.open(path) as img:
                thumb = ImageOps.contain(img.convert("RGB"), (320, 229))
            x = j % 3 * 330 + (330 - thumb.width) // 2
            y = j // 3 * 275 + 25 + (229 - thumb.height) // 2
            sheet.paste(thumb, (x, y))
            pen.text((j % 3 * 330 + 8, j // 3 * 275 + 6),
                     f"{start + (offset + j) / fps:.2f}s", fill="white")
        dest = folder / f"{prefix}_{len(result):02d}.jpg"
        sheet.save(dest, "JPEG", quality=72, optimize=True)
        sheet.close()
        result.append(dest)
    return result


def _audio(video, folder, has_audio):
    if not has_audio:
        return None
    dest = folder / "audio.wav"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(dest),
    ], 70)
    with wave.open(str(dest), "rb") as sound:
        content = sound.readframes(sound.getnframes())
    values = array("h")
    values.frombytes(content)
    if not values:
        return None
    stride = max(1, len(values) // 8000)
    sample = values[::stride]
    rms = math.sqrt(sum(v * v for v in sample) / len(sample)) / 32768
    return dest if rms >= 0.0008 else None


def _motion_values(frames):
    """Lightweight motion cues; not proof that an artifact exists."""
    scores = [0.0]
    previous = None
    for path in frames:
        with Image.open(path) as src:
            im = ImageOps.fit(src.convert("L"), (64, 64))
        if previous is not None:
            histogram = ImageChops.difference(previous, im).histogram()
            score = sum(i * n for i, n in enumerate(histogram)) / (4096 * 255)
            scores.append(score)
        previous = im
    # Scores are between 0 and 1; remove the unused first duplicate.
    return scores[:len(frames)]


def _details(frames, fps, motion, max_count=5):
    """Diverse detail frames: opening, middle, end and significant motion."""
    length = len(frames)
    selected = {min(length - 1, round((length - 1) * p)) for p in (0.14, 0.51, 0.86)}
    ranks = sorted(range(1, length), key=lambda i: motion[i], reverse=True)
    for idx in ranks:
        if len(selected) >= max_count:
            break
        if all(abs(idx - s) >= max(2, round(fps * 0.7)) for s in selected):
            selected.add(idx)
    return [(i / fps, frames[i]) for i in sorted(selected)]


def _inline(path, mime):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": mime, "data": encoded}}


SCHEMA = r'''
{
 "product_type_th":"ประเภทสินค้าให้เฉพาะเจาะจง",
 "visible_product_th":"สินค้าที่เห็น",
 "spoken_product_th":"สินค้าในคำพูด หรือ ไม่มีคำพูด",
 "intended_use_th":"การใช้งานที่เห็นหรือได้ยินจริง",
 "selling_context_th":"กำลังสาธิต/ขายอะไรและมีข้ออ้างอะไร",
 "scale":{"status":"PASS","checked":true,"confidence":0.0,
  "anchor_th":"จุดเทียบขนาด มือ คน โต๊ะ หรือบอกเหตุผลเมื่อไม่มี",
  "normal_reason_th":"เหตุผลเฉพาะของสเกลที่สมจริง",
  "large_use_claim":false,"tabletop_miniature":false,"mini_is_correct":false,
  "events":[{"time_sec":1.0,"seen_th":"","problem_th":""}]},
 "speech":{"status":"PASS","checked":true,"confidence":0.0,
  "has_speech":true,"heard_excerpt_th":"คำที่ฟังได้จริงประมาณหนึ่งวลี",
  "wrong_words":[{"time_sec":1.0,"heard_th":"","problem_th":""}],
  "gibberish":false,"gibberish_evidence_th":""},
 "match":{"status":"PASS","checked":true,"confidence":0.0,
  "spoken_claim_th":"คำอ้างเกี่ยวกับสินค้าในเสียง",
  "visual_fact_th":"สิ่งที่เห็นในภาพ ณ ช่วงเดียวกัน",
  "comparison_th":"ความสอดคล้องหรือความขัดแย้งของคำพูดกับภาพ"},
 "people":{"status":"PASS","checked":true,"confidence":0.0,
  "visible":false,"interaction_th":"สิ่งที่คนหรือมือทำต่อสินค้า",
  "events":[{"time_sec":1.0,"seen_th":"","problem_th":""}]},
 "visual":{"status":"PASS","checked":true,"confidence":0.0,
  "continuity_th":"สิ่งที่เปลี่ยนหรือคงรูปของสินค้าระหว่างคลิป",
  "events":[{"time_sec":1.0,"seen_th":"","problem_th":""}]},
 "mirror":{"status":"PASS","checked":true,"applicable":false,
  "confidence":0.0,"reflection_expected":false,"angle_uncertain":false,
  "reason_th":"อธิบายเรขาคณิตการสะท้อนเมื่อเป็นกระจก",
  "events":[{"type":"missing_hand_reflection","start_sec":1.0,"end_sec":3.1,
             "observed_th":"","geometry_th":"","sample_times_sec":[1.0,2.0,3.1]}]},
 "risk_flags":[],
 "risk_moments_sec":[],
 "summary_th":"สรุปสั้นตามสิ่งที่เห็นและฟัง"
}
'''


FIRST_PROMPT = r'''
คุณคือผู้ตรวจโฆษณาสินค้า AI แบบจับข้อผิดพลาดจริง (MP4 ล้วน ไม่มี JSON)
คำสั่งสำคัญ: อย่าตอบ PASS รวม ๆ เพียงเพราะภาพสวยหรือเสียงพูดลื่นไหล
ตรวจทุกด้านตามตัวสินค้าและประเมินว่ากำลังโฆษณาสิ่งใดจริง
ห้ามตั้งเป้าสัดส่วน FAIL ห้ามเดาขนาด เซนติเมตร รุ่น คุณสมบัติที่ไม่ปรากฏ

1) PRODUCT CONTEXT: บอกสินค้าที่เห็น สินค้าที่พูด หน้าที่ที่สาธิต
   จำแนกสินค้าได้ทุกหมวด เช่น กีฬา เฟอร์นิเจอร์ กระจก ของใช้ครัว
   ถังขยะ ชั้นวาง รถเข็น อุปกรณ์เด็ก อิเล็กทรอนิกส์ เสื้อผ้า
   ใช้หลักการตรวจตามประเภท อย่าสมมติว่าของที่เล็กคือของปลอมเสมอ

2) SCALE: หาจุดเทียบในระนาบเดียวกัน (นิ้ว มือ คน ชั้น โต๊ะ พื้น)
   ตรวจสัดส่วนและการใช้งานตามภาพ/คำพูด ตรวจของจิ๋วที่ถูกใช้โฆษณา
   แทนของขนาดใช้งานจริง แต่ถ้าเป็นสินค้ามินิที่ขายตรงตามขนาดจริง
   หรืออุปกรณ์ปรับระดับ/พับเก็บตามปกติ อย่าตัด FAIL
   ถ้าผู้พูดอ้างว่าใช้จริง/ตั้งพื้น แต่ภาพเป็นรุ่นเล็กตั้งโต๊ะ
   ให้ระบุ large_use_claim=true และ tabletop_miniature=true
   พร้อมเหตุการณ์และจุดขัดแย้ง ห้ามเดาเรื่องความสูงตัวเลข

3) SPEECH: ฟังเสียงจริงทุกช่วง ขอวลีที่ได้ยินจริงอย่างน้อยหนึ่งวลี
   เสียงไทยผิดชัดเจนตั้งแต่สองตำแหน่งคำขึ้นไปให้รายงานคำพร้อมเวลา
   เสียงมั่วต่อเนื่องจนเข้าใจไม่ได้เป็นปัญหาสำคัญ
   หนึ่งคำเล็กน้อย สำเนียง ชื่อแบรนด์ ไม่ใช่เหตุให้ FAIL
   อย่าสร้างคำพูดหรือคำที่ไม่เคยได้ยิน

4) MATCH: เอาวลีจากเสียงเทียบภาพในเวลาเดียวกัน
   ชนิดสินค้า สี จำนวน รุ่น/คุณสมบัติที่พิสูจน์ได้ วิธีการใช้
   ถ้าพูดสินค้าคนละชนิดกับภาพ ให้รายงานชัด
   ถ้าพูดข้อมูลที่ภาพตรวจไม่ได้ อย่าเดาว่าผิด

5) PEOPLE: ดูรูปมือ/นิ้ว การสัมผัสสินค้า การทะลุ การหาย การเพิ่มนิ้ว
   ต้องมีเหตุการณ์จริงตามเวลา จับผิด AI artifacts ที่รุนแรง
   มือที่ออกจากขอบภาพตามมุมกล้องไม่ใช่มือหายผิดธรรมชาติ

6) VISUAL: สินค้ายืดหดผิดฟิสิกส์ ล้อ/ขา/ฐานขาด สินค้าเปลี่ยนชนิด
   จำนวนสินค้าเปลี่ยนเอง การวางของผิดฟิสิกส์รุนแรง
   การมีกระปุกเอียงหรือชั้นบาง ๆ เพียงอย่างเดียวไม่พอ FAIL
   ต้องมีข้อผิดพลาดชัดเจนของสินค้าและการใช้งาน

7) MIRROR: เฉพาะสินค้ากระจกเงาที่สะท้อนภาพ
   หากมือควรอยู่ในภาพสะท้อนแต่หายเกิน 2 วินาที ต้องระบุ
   หลายตำแหน่งเวลาที่มองเห็นเหตุการณ์จริงและอธิบายมุมสะท้อน
   ตรวจเงาบิดผิดรูป/เคลื่อนไม่ตรงกับของจริง
   มุมกล้องหรือสิ่งบังที่ไม่แน่ชัด = REVIEW ไม่เดาว่าผิด

ทุกส่วนต้องมี checked=true หากตรวจแล้ว; หากไม่เกี่ยวข้องให้ PASS
และอธิบายตามสมควร ถ้ามีเสียงพูดต้องกรอก heard_excerpt_th
PASS ต้องมีเหตุผลสังเกตได้จริง ไม่ใช้คำว่า "ทุกอย่างปกติ" เฉย ๆ
บอกเหตุการณ์ที่ผิดจริงเป็น time_sec / seen_th / problem_th
ถ้าไม่มีเหตุการณ์ให้ events=[]; ถ้าไม่มีคำผิดให้ wrong_words=[]
หากหลักฐานไม่ชัดแต่มีข้อสงสัยสำคัญ ให้ REVIEW ไม่ใช่ FAIL
ตอบเป็น JSON object เท่านั้นตามตัวอย่างโครงสร้าง (อย่าคัดลอกตัวอย่างเป็นเหตุการณ์):
''' + SCHEMA


SECOND_PROMPT = r'''
คุณคือผู้ตรวจยืนยันอิสระด้านคุณภาพวิดีโอสินค้า
ไม่ทราบคำตัดสินรอบแรก ตรวจจากภาพ/เสียงใหม่เท่านั้น
ตรวจเฉพาะหัวข้อที่ระบุ ให้เหตุผลว่าทำไมถูกหรือผิด
ห้ามสมมติขนาดสินค้าหรือคำที่ไม่ได้ยิน
ถ้าเป็นภาพเพียงบางช่วง อย่าสรุปว่ามีข้อผิดพลาดตลอดคลิป
ตรวจสเกลด้วยมือ คน การใช้งานและความต่างระหว่างของจิ๋วกับของจริง
ตรวจคน/มือ: ถ้ามือบิด/หาย/ทะลุ ให้ระบุจุดในเวลา
ตรวจเสียง: รายงานสองคำผิดจริงคนละเวลา และคำพูดเทียบภาพ
ตรวจเงากระจก: เหตุการณ์หลายเฟรม มุมสะท้อน และเวลาเกิน 2 วินาที
มีข้อขัดแย้งยืนยันได้ = FAIL; สงสัยแต่ยังพิสูจน์ไม่ได้ = REVIEW
กรณีไม่พบข้อผิดพลาด = PASS พร้อมข้อมูลที่สังเกตได้
ด้านอื่นที่ไม่ได้รับหลักฐานให้ checked=false และ events=[]
ตอบ JSON object ตามโครงสร้างนี้:
''' + SCHEMA


def _parts(prompt, duration, sheets, details, audio):
    content = [{"text": f"{prompt}\nวิดีโอความยาว {duration:.2f} วินาที"}]
    for file in sheets:
        content.extend([{"text": "ภาพตามเวลา พร้อมป้ายเวลาในแต่ละช่อง"},
                        _inline(file, "image/jpeg")])
    for sec, file in details:
        content.extend([{"text": f"ภาพเดี่ยวรายละเอียดสูงที่ {sec:.2f}s"},
                        _inline(file, "image/jpeg")])
    if audio is not None:
        content.extend([{"text": "เสียงจริงตลอดคลิป ให้ฟังภาษาไทยด้วย"},
                        _inline(audio, "audio/wav")])
    else:
        content.append({"text": "ไม่มีเสียงที่ได้ยินเพื่อตรวจคำพูด ห้ามแต่งคำพูด"})
    return content


def _api(api_key, model, parts, max_tokens):
    if not api_key:
        raise AnalysisError("ไม่พบ GEMINI_API_KEY")
    choice = model if model in MODELS else DEFAULT_MODEL
    alternatives = [choice]
    index = 0
    while index < len(alternatives):
        selected = alternatives[index]
        index += 1
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               + selected + ":generateContent")
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "maxOutputTokens": max_tokens,
            },
        }
        for retry in range(2):
            try:
                response = requests.post(
                    url, headers={"x-goog-api-key": api_key}, json=payload,
                    timeout=(12, 105),
                )
            except requests.RequestException as exc:
                if retry == 0:
                    time.sleep(1)
                    continue
                raise AnalysisError(f"ติดต่อ Gemini ไม่สำเร็จ: {exc}") from exc
            if response.status_code == 429:
                raise QuotaError("Gemini API ถึงโควตา (429) กรุณาพักแล้วลองใหม่")
            if response.status_code in (500, 502, 503, 504) and retry == 0:
                time.sleep(1.0)
                continue
            if response.status_code >= 400:
                try:
                    msg = response.json().get("error", {}).get("message", "")
                except ValueError:
                    msg = response.text[:300]
                if response.status_code == 404:
                    # If a formerly selected model disappears, try a model
                    # *actually listed* for this key. Never silently choose
                    # an invented model name.
                    try:
                        for name in available_models(api_key):
                            if name not in alternatives:
                                alternatives.append(name)
                    except AnalysisError:
                        pass
                    if index < len(alternatives):
                        break
                raise AnalysisError(f"Gemini HTTP {response.status_code} "
                                    f"({selected}): {_str(msg, 330)}")
            try:
                pieces = response.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in pieces).strip()
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
                obj = json.loads(text)
                sections = ("scale", "speech", "match", "people", "visual", "mirror")
                if not isinstance(obj, dict) or any(
                    not isinstance(obj.get(s), dict) for s in sections
                ):
                    raise ValueError("ผล QC ไม่ครบ 6 หัวข้อ")
                return obj
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                raise AnalysisError(f"Gemini ตอบ JSON ไม่สมบูรณ์: {exc}") from exc
    raise AnalysisError("ไม่สามารถใช้โมเดลที่คีย์เข้าถึงได้")


def _events(block, duration):
    records = block.get("events", [])
    if not isinstance(records, list):
        return []
    out = []
    for e in records[:14]:
        if not isinstance(e, dict):
            continue
        sec = _float(e.get("time_sec"), -1)
        seen, reason = _str(e.get("seen_th"), 240), _str(e.get("problem_th"), 330)
        if 0 <= sec <= duration and _evidence(seen) and _evidence(reason):
            out.append((sec, seen, reason))
    return out


def _words(block, duration):
    raw = block.get("wrong_words", [])
    if not isinstance(raw, list):
        return []
    out = []
    for e in raw[:16]:
        if not isinstance(e, dict):
            continue
        sec = _float(e.get("time_sec"), -1)
        heard, problem = _str(e.get("heard_th"), 80), _str(e.get("problem_th"), 190)
        if (0 <= sec <= duration and heard and _evidence(problem)
                and all(abs(sec - other[0]) >= 0.20 for other in out)):
            out.append((sec, heard, problem))
    return out


def _mirror_events(block, duration):
    events = block.get("events", [])
    if not isinstance(events, list):
        return []
    permitted = {"missing_hand_reflection", "severe_warp", "impossible_reflection", "unrelated_motion"}
    result = []
    for event in events[:12]:
        if not isinstance(event, dict) or event.get("type") not in permitted:
            continue
        a, b = _float(event.get("start_sec"), -1), _float(event.get("end_sec"), -1)
        if not 0 <= a < b <= duration:
            continue
        seen = _str(event.get("observed_th"), 250)
        geometry = _str(event.get("geometry_th"), 250)
        if not (_evidence(seen) and _evidence(geometry)):
            continue
        times = event.get("sample_times_sec", [])
        if not isinstance(times, list):
            times = []
        pts = sorted({round(_float(t, -1), 2) for t in times
                      if a - 0.5 <= _float(t, -1) <= b + 0.5})
        multiple = len(pts) >= 3 and (pts[-1] - pts[0]) >= 1.8
        result.append((a, b, event["type"], seen, geometry, multiple))
    return result


def _mirror_applies(obj):
    label = _str(obj.get("product_type_th"), 180)
    return (
        _yes(obj["mirror"].get("applicable"))
        or bool(re.search(r"กระจกเงา|กระจกส่อง|กระจกแต่งหน้า|\bmirror\b", label, re.I))
    )


def _missing_evidence(obj, audible, scope):
    """Evidence coverage is not physical validation; it catches lazy PASS."""
    missing = set()
    product = _str(obj.get("visible_product_th"))
    if not _evidence(product, 4) or not _evidence(obj.get("product_type_th"), 4):
        missing.add("scale")
        missing.add("visual")
    for category in scope:
        item = obj[category]
        if category in {"speech", "match"} and not audible:
            continue
        if category == "mirror" and not _mirror_applies(obj):
            continue
        if not _yes(item.get("checked")):
            missing.add(category)
    sc = obj["scale"]
    if "scale" in scope and _state(sc) == "PASS":
        if _yes(sc.get("large_use_claim")) and _yes(sc.get("tabletop_miniature")) and not _yes(sc.get("mini_is_correct")):
            missing.add("scale")
        if not (_evidence(sc.get("anchor_th")) or _evidence(sc.get("normal_reason_th"))):
            missing.add("scale")
    sp = obj["speech"]
    if (audible and "speech" in scope and _yes(sp.get("has_speech"))
            and _state(sp) == "PASS" and not _evidence(sp.get("heard_excerpt_th"), 5)):
        missing.add("speech")
    match = obj["match"]
    if (audible and "match" in scope and _yes(sp.get("has_speech"))
            and _state(match) == "PASS" and not _evidence(match.get("comparison_th"))):
        missing.add("match")
    pe = obj["people"]
    if ("people" in scope and _yes(pe.get("visible")) and _state(pe) == "PASS"
            and not _evidence(pe.get("interaction_th"))):
        missing.add("people")
    vis = obj["visual"]
    if ("visual" in scope and _state(vis) == "PASS"
            and not _evidence(vis.get("continuity_th"))):
        missing.add("visual")
    if ("mirror" in scope and _mirror_applies(obj) and _state(obj["mirror"]) == "PASS"
            and not _evidence(obj["mirror"].get("reason_th"))):
        missing.add("mirror")
    return missing & set(scope)


def _judge(obj, duration, audible, scope=None):
    scope = set(scope if scope is not None else
                ("scale", "speech", "match", "people", "visual", "mirror"))
    fails, concerns = [], set()
    coverage = _missing_evidence(obj, audible, scope)
    for area in scope:
        b = obj[area]
        state, conf = _state(b), _float(b.get("confidence"))
        if area in ("scale", "people", "visual"):
            events = _events(b, duration)
            anchor = _evidence(b.get("anchor_th")) if area == "scale" else True
            confirmed = bool(events) and conf >= (0.70 if area == "scale" else 0.76)
            if area == "scale":
                confirmed = confirmed and not _yes(b.get("mini_is_correct")) and (anchor or len(events) >= 2)
            if state == "FAIL" and confirmed:
                t, seen, problem = events[0]
                fails.append((area, t, f"{area}: {seen} — {problem}"))
            elif state != "PASS" or events:
                concerns.add(area)
            if (area == "scale" and _yes(b.get("large_use_claim"))
                    and _yes(b.get("tabletop_miniature"))
                    and not _yes(b.get("mini_is_correct"))):
                # A contradictory presentation cannot be approved only
                # by an AI-generated 'PASS' label.
                concerns.add("scale")
        elif area == "speech" and audible:
            words = _words(b, duration)
            if state == "FAIL" and conf >= 0.71 and len(words) >= 2:
                msg = "; ".join(f"{t:.1f}s '{w}' ({why})" for t, w, why in words[:3])
                fails.append(("speech", words[0][0], "เสียงพูดเพี้ยน >=2 ตำแหน่ง: " + msg))
            elif (state == "FAIL" and conf >= 0.80
                  and _yes(b.get("gibberish")) and _evidence(b.get("gibberish_evidence_th"))):
                fails.append(("speech", 0.0, "เสียงมั่วต่อเนื่อง: " + _str(b.get("gibberish_evidence_th"))))
            elif state != "PASS" or len(words) >= 2:
                concerns.add("speech")
        elif area == "match" and audible:
            if (state == "FAIL" and conf >= 0.72
                    and _evidence(b.get("spoken_claim_th"))
                    and _evidence(b.get("visual_fact_th"))
                    and _evidence(b.get("comparison_th"))):
                fails.append(("match", 0.0, "คำพูดไม่ตรงภาพ: " + _str(b.get("comparison_th"))))
            elif state != "PASS":
                concerns.add("match")
        elif area == "mirror" and _mirror_applies(obj):
            events = _mirror_events(b, duration)
            valid = []
            for t1, t2, kind, seen, geo, samples_ok in events:
                if not samples_ok or _yes(b.get("angle_uncertain")):
                    continue
                if kind == "missing_hand_reflection" and (
                    t2 - t1 <= 2.0 or not _yes(b.get("reflection_expected"))
                ):
                    continue
                valid.append((t1, kind, seen, geo))
            if state == "FAIL" and conf >= 0.80 and valid:
                t, kind, seen, geo = valid[0]
                fails.append(("mirror", t, f"เงาสะท้อนผิด ({kind}): {seen}; {geo}"))
            elif state != "PASS" or events:
                concerns.add("mirror")
    for flag in (obj.get("risk_flags") if isinstance(obj.get("risk_flags"), list) else [])[:8]:
        s = _str(flag).lower()
        if not _evidence(s):
            continue
        if any(term in s for term in ("สเกล", "ขนาด", "เล็ก", "scale", "mini")):
            concerns.add("scale")
        elif any(term in s for term in ("เสียง", "พูด", "speech", "voice", "audio")):
            if audible:
                concerns.add("speech")
        elif any(term in s for term in ("กระจก", "เงา", "mirror", "reflect")):
            concerns.add("mirror")
        elif any(term in s for term in ("นิ้ว", "มือ", "hand", "finger")):
            concerns.add("people")
        elif any(term in s for term in ("ไม่ตรง", "คำพูด", "mismatch")):
            if audible:
                concerns.add("match")
        else:
            concerns.add("visual")
    return fails, (concerns | coverage) & scope, coverage


def _audit_sample(video, first):
    """Small deterministic challenge of clean passes; not a target FAIL quota."""
    with open(video, "rb") as f:
        seed = int.from_bytes(hashlib.sha256(f.read(65536)).digest()[:4], "big") % 100
    pe = first["people"]
    risky_interaction = _yes(pe.get("visible")) or _mirror_applies(first)
    return seed < (INTERACTION_AUDIT_PERCENT if risky_interaction else CLEAN_AUDIT_PERCENT)


def _focus_time(first, focus, duration):
    times = []
    for area in focus:
        block = first[area]
        if area in ("scale", "people", "visual"):
            times.extend(t for t, *_ in _events(block, duration))
        elif area == "speech":
            times.extend(t for t, *_ in _words(block, duration))
        elif area == "mirror":
            times.extend((a + b) / 2 for a, b, *_ in _mirror_events(block, duration))
    hints = first.get("risk_moments_sec", [])
    if isinstance(hints, list):
        times.extend(_float(t, -1) for t in hints[:5] if 0 <= _float(t, -1) <= duration)
    return min(times) if times else duration * 0.5


def _audit_media(video, folder, duration, first, focus, wav, sheets, details):
    images = bool(focus & {"scale", "match", "people", "visual", "mirror"})
    sound = bool(focus & {"speech", "match"})
    window_info = None
    if not images:
        return [], [], wav if sound else None, window_info
    if focus & {"scale", "people", "visual", "mirror"}:
        t = _focus_time(first, focus, duration)
        start = max(0.0, min(duration - min(duration, AUDIT_WINDOW), t - AUDIT_WINDOW / 2))
        length = min(AUDIT_WINDOW, duration - start)
        local_frames = _frames(video, folder / "audit_frames", AUDIT_FPS,
                               start=start, length=length)
        local_sheets = _sheets(local_frames, AUDIT_FPS, folder,
                               start=start, prefix="audit_sheet")
        picks = sorted({0, len(local_frames) // 2, len(local_frames) - 1})
        local_details = [(start + i / AUDIT_FPS, local_frames[i]) for i in picks]
        window_info = [round(start, 2), round(start + length, 2)]
        return local_sheets[:3], local_details[:3], wav if sound else None, window_info
    # Mismatch requires broader video context.
    step = max(1, len(sheets) // 3)
    return sheets[::step][:3], details[:2], wav if sound else None, window_info


def inspect(path, key, model, mode, thumbnail, reference=""):
    """V3.9.1 app / queue compatible; reference ignored (MP4 only)."""
    start_all = time.perf_counter()
    video, thumb = Path(path), Path(thumbnail)
    duration, has_audio = _probe(video)
    fps = FPS.get(mode, 4)
    with tempfile.TemporaryDirectory(prefix="qc393_") as temp:
        folder = Path(temp)
        frames = _frames(video, folder / "frames", fps)
        sheets = _sheets(frames, fps, folder)
        motion = _motion_values(frames)
        details = _details(frames, fps, motion, max_count=5)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        if not thumb.is_file():
            with Image.open(frames[0]) as image:
                ImageOps.contain(image.convert("RGB"), (320, 320)).save(
                    thumb, "JPEG", quality=72)
        wav = _audio(video, folder, has_audio)
        prep_end = time.perf_counter()

        first = _api(key, model, _parts(FIRST_PROMPT, duration, sheets, details, wav), 3300)
        first_end = time.perf_counter()
        fails, issues, coverage = _judge(first, duration, wav is not None)
        second, focus, window = None, set(), None
        final_fails = list(fails)
        pending = set(issues)
        if not final_fails:
            focus = set(issues)
            if not focus and _audit_sample(video, first):
                focus = {"scale", "people", "visual"}
                if wav is not None and _yes(first["speech"].get("has_speech")):
                    focus.update({"speech", "match"})
                if _mirror_applies(first):
                    focus.add("mirror")
            if focus:
                sheets2, details2, wav2, window = _audit_media(
                    video, folder, duration, first, focus, wav, sheets, details)
                note = "\nตรวจเฉพาะหัวข้อ: " + ", ".join(sorted(focus))
                if window:
                    note += (f"\nภาพสำหรับตรวจยืนยันเป็นเพียงช่วง "
                             f"{window[0]}–{window[1]} วินาที ไม่ใช่ทั้งคลิป")
                second = _api(key, model, _parts(SECOND_PROMPT + note, duration,
                                                  sheets2, details2, wav2), 2700)
                final_fails, pending, _ = _judge(second, duration, wav is not None, focus)
                # A strong, timestamped original concern is not erased by
                # a generic PASS when the second view contains only a crop.
                for area in issues & focus:
                    first_events = (_mirror_events(first[area], duration)
                                    if area == "mirror" else
                                    _words(first[area], duration)
                                    if area == "speech" else
                                    _events(first[area], duration)
                                    if area in {"scale", "people", "visual"} else [])
                    if first_events and not any(x[0] == area for x in final_fails):
                        pending.add(area)
                # A model may claim first-pass PASS but second-pass REVIEW.
                # That clip is held for review instead of going into PASS ZIP.
        decision = "FAIL" if final_fails else "REVIEW" if pending else "PASS"
        if final_fails:
            reason = "; ".join(x[2] for x in final_fails[:3])
        elif pending:
            reason = "รอตรวจด้วยคน: พบประเด็นหรือหลักฐานยังไม่ครบด้าน " + ", ".join(sorted(pending))
        else:
            reason = _str(first.get("summary_th"), 800) or "ไม่พบข้อผิดพลาดสำคัญจากภาพและเสียง"
        source = second if second is not None and final_fails else first
        finished = time.perf_counter()
        return {
            "decision": decision,
            "reason_th": reason[:850],
            "product_type_th": _str(source.get("product_type_th"), 170),
            "product_name_th": _str(source.get("visible_product_th"), 170),
            "spoken_product_th": _str(source.get("spoken_product_th"), 170),
            "speech_summary_th": _str(first["speech"].get("heard_excerpt_th"), 550),
            "video_type": "PRODUCT",
            "scale_review": source["scale"],
            "audio_review": source["speech"],
            "match_review": source["match"],
            "people_review": source["people"],
            "visual_review": source["visual"],
            "mirror_review": source["mirror"],
            "first_scale_review": first["scale"],
            "first_audio_review": first["speech"],
            "scale_audit": second.get("scale") if second else None,
            "speech_audit": second.get("speech") if second else None,
            "match_audit": second.get("match") if second else None,
            "people_audit": second.get("people") if second else None,
            "visual_audit": second.get("visual") if second else None,
            "mirror_audit": second.get("mirror") if second else None,
            "issues": ([{"category": a, "time_sec": round(t, 2),
                         "detail_th": why, "severity": "critical"}
                        for a, t, why in final_fails]
                       + ([{"category": "human_review", "time_sec": 0,
                            "detail_th": reason, "severity": "warning"}]
                          if pending and not final_fails else [])),
            "evidence_coverage_missing": sorted(coverage),
            "review_required": decision == "REVIEW",
            "audit_done": second is not None,
            "audit_trigger": ", ".join(sorted(focus)) if focus else "none",
            "audit_dense_window_sec": window,
            "product_reference_used": False,
            "duration": round(duration, 2),
            "elapsed": round(finished - start_all, 2),
            "engine": ENGINE_VERSION,
            "timings": {
                "prepare_seconds": round(prep_end - start_all, 2),
                "first_api_seconds": round(first_end - prep_end, 2),
                "audit_api_seconds": round(finished - first_end, 2) if second else 0.0,
            },
        }
