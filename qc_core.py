
"""V3.3 Quality Inspector: scale, Thai speech, and independent PASS audit.

Drop-in replacement for V3 qc_core.py used by app.py / qc_store.py.
Requires ffmpeg, ffprobe, requests, Pillow. No API key embedded.
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
from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat


# เปลี่ยน ENGINE_VERSION เพื่อไม่ให้ qc_store ดึง Cache เวอร์ชันก่อน
ENGINE_VERSION = "v3-3-integrated-scale-speech-audit-20261011"
FPS = {"Fast": 1, "Balanced": 2, "Deep": 3}
ALLOWED_MODELS = {
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
}

# True: ตรวจเสียง + สเกลซ้ำ ทุกคลิปที่รอบแรกได้ PASS
# ค่า True เน้นคุณภาพ (อาจเรียก Gemini 2 ครั้งต่อคลิป)
RECHECK_EVERY_PASS = True
FRAME_SIZE = 768
MAX_FRAMES = 75


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


FIRST_PROMPT = """
คุณเป็นผู้ตรวจคุณภาพวิดีโอโฆษณาที่สร้างด้วย AI สำหรับการขายสินค้า
ฟังเสียงจริงและตรวจสินค้าจากทุกภาพที่ได้รับ ไม่เดาเสียงจากภาพ
เน้นจับข้อผิดพลาดรุนแรง แต่ยอมให้ข้อบกพร่องเล็กน้อยผ่าน

ภาพ:
- จำแนก HUMAN/HANDS/PRODUCT/MIXED แต่ตรวจขนาดสินค้าทุกประเภท แม้มีคนขาย
- เทียบสินค้าและมือ/นิ้วที่แตะสินค้า, วัตถุข้างเคียงที่ระยะใกล้กัน, ความสามารถใช้งานจริง
- ระวังวัตถุขนาดจิ๋วปลอมเป็นของขนาดปกติ, ใหญ่เกินจริง, หรือเปลี่ยนขนาดระหว่างคลิป
- อย่าให้ PASS เพียงเพราะภาพสวย สินค้าสะอาด หรือฉากดูสมจริง
- ถ้าขนาดที่แสดงทำให้ใช้ตามคำโฆษณาไม่ได้อย่างชัดเจน ให้ FAIL
- แต่ถ้าเป็นสินค้ามินิ/ของเล่นจริงที่นำเสนอถูกประเภท ให้ PASS ได้
- มุมกล้อง/ซูม/การพับกางตามกลไกปกติ ไม่ใช่หลักฐานผิดสเกล
- ขนาดเล็กเพียงอย่างเดียวไม่ใช่หลักฐานว่าโฆษณาหลอก
- ตรวจความต่อเนื่องสินค้า รูปทรง มือ การสัมผัสกับสินค้า

เสียงไทย:
- ต้องฟังเสียงจริง ห้ามสร้างคำพูดที่ไม่ได้ยิน
- คำเพี้ยนไม่สำคัญ 1 คำ หรือสำเนียง AI เล็กน้อย ให้ผ่านได้
- ถ้าได้ยินคำที่ออกมาเป็นเสียงมั่ว/เพี้ยนจริง >=2 คำ (แม้ไม่ติดกัน) ให้ FAIL
- ถ้ามีเสียงมั่วหลายคำติดกันจนแยกไม่ได้ ให้ FAIL พร้อมเวลาช่วงนั้น
- หากชื่อสินค้า/ขนาด/จำนวน/ความจุสำคัญฟังไม่รู้เรื่อง หรือขัดกับข้อมูลที่ยืนยันได้ ให้ FAIL
- อย่านับการพูดเร็ว สำเนียงท้องถิ่น ชื่อแบรนด์ และศัพท์เฉพาะเป็นคำผิด
- อย่าถือว่าเสียงผิดจากกราฟความดังหรือเสียงสังเคราะห์เพียงอย่างเดียว
- หากไม่มีเสียงพูด ไม่จำเป็นต้อง FAIL หากไม่ได้กำหนดว่าต้องมีบทพูด

ความสัมพันธ์:
- ตรวจว่าสิ่งที่พูดคือสินค้าที่ถือหรือโชว์จริงหรือไม่
- ต้องไม่เดาสเปกนิ้ว/เซนติเมตรหรือจำนวนจากมุมกล้องที่พิสูจน์ไม่ได้
- ข้อสงสัยยังยืนยันไม่ได้ให้ระบุ suspicious/uncertain ไม่แต่งหลักฐาน

JSON object เท่านั้น ไม่ใช้ markdown โดยกรอกทุก key ที่ระบุ:
{
 "decision":"PASS",
 "video_type":"HANDS",
 "product_name_th":"ชื่อสินค้าหลัก",
 "reason_th":"เหตุผลสั้นชัดเจน",
 "speech_summary_th":"คำพูดที่เข้าใจได้จริง ไม่แต่งส่วนฟังไม่รู้เรื่อง",
 "issues":[{"time_sec":2.0,"category":"product_scale","severity":"minor","detail_th":"รายละเอียด"}],
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
   "reason_th":""
 },
 "audio_review":{
   "status":"clear",
   "speech_present":true,
   "confidence":0.0,
   "broken_word_count":0,
   "broken_words_evidence_th":"",
   "severe_unintelligible":false,
   "key_information_unintelligible":false,
   "meaning_conflict":false,
   "severe_audio_artifact":false,
   "time_sec":0.0,
   "evidence_th":""
 }
}
size_grade: normal, intentional_mini, suspicious, wrong_scale, unknown
status: clear, minor, critical, no_speech, uncertain
confidence คือความมั่นใจว่ามีข้อผิดพลาดร้ายแรง (ไม่ใช่ความมั่นใจว่าเสียงดี)
"""


# ผู้ตรวจยืนยันจะไม่เห็นผลรอบแรก เพื่อไม่ให้เชื่อตาม PASS เดิม
AUDIT_PROMPT = """
คุณเป็นผู้ตรวจคุณภาพอิสระจากภาพหลายช่วงและเสียงจริงของคลิปโฆษณา AI
ห้ามสมมติว่าคลิปนี้ผ่านการตรวจใด ๆ มาก่อน
หน้าที่คือหาปัญหาที่ผู้ตรวจทั่วไปอาจพลาด โดยไม่กล่าวหาสินค้ามินิจริง ๆ

งาน A: ตรวจสเกลเชิงหลักฐาน
1) ความสัมพันธ์ขนาดกับมือ/นิ้วที่สัมผัสสินค้า ไม่ใช้มือที่อยู่ไกลกล้องมาเทียบ
2) ขนาดกับวัตถุในระนาบใกล้เคียงกันและสภาพแวดล้อม
3) หน้าที่ที่โฆษณาว่าใช้งานได้จริง และคำพูดที่ได้ยิน
4) สินค้าหด/ขยาย/เปลี่ยนโครงสร้างข้ามเวลาอย่างเป็นไปไม่ได้หรือไม่
ถ้ารูปทรงถูกต้องแต่ขนาดผิดรุนแรงจนใช้งานตามโฆษณาไม่ได้ ให้ scale.decision=FAIL
ต้องยกหลักฐานอิสระจากอย่างน้อย 2 ด้าน และ 1 ด้านต้องเกี่ยวกับการใช้งาน
ห้ามตัด FAIL เพียงเพราะของเล็กหรือถือได้ด้วยมือเดียว
สินค้าที่ขายเป็น mini/toy/ตั้งโต๊ะจริง ใช้งานตามวัตถุประสงค์ mini ได้ ให้ PASS ด้านสเกล
อย่าอนุมานว่าขนาดจิ๋วถูกต้องเพียงเพราะเห็นของจิ๋ว ต้องหาหลักฐานการนำเสนอจริง
มุมกล้อง การซูม และชั้นพับตามกลไก ไม่ใช่การเปลี่ยนขนาดผิดธรรมชาติ
หากหลักฐานไม่พอ ให้ UNCERTAIN แทนการเดา

งาน B: ตรวจเสียงพูดภาษาไทยจริงแบบอิสระ
- ฟังตั้งแต่ต้นจนจบ หาคำเพี้ยนเป็นเสียงมั่ว/เสียงผิดที่เข้าใจไม่ได้
- 0-1 คำผิดเล็กน้อย (ไม่เปลี่ยนสาระ) ให้ PASS ด้านเสียง
- ตั้งแต่ 2 คำผิดจริงขึ้นไป (อาจอยู่คนละช่วง) ให้ FAIL ด้านเสียง
- คำมั่วต่อเนื่องจนแบ่งไม่ได้ >=2 คำ ให้ continuous_gibberish=true
- หากข้อมูลชื่อสินค้า ขนาด จำนวน ฯลฯ พูดผิดจากหลักฐานชัด ให้ FAIL
- อย่านับสำเนียง แบรนด์ คำทับศัพท์ พูดเร็ว เสียง TTS เพียงอย่างเดียว
- ห้ามประดิษฐ์คำเพี้ยนเพื่อให้ครบ 2 คำ ต้องแสดงเวลาของทุกคำที่ตรวจพบจริง
- ถ้าไม่พบคำผิด ให้ events=[] ไม่ต้องกรอกตัวอย่างคำผิดใน schema
- ถ้าไม่มีเสียงพูด ให้ speech_present=false, decision=PASS, events=[]
- รูปภาพช่วยบอกชื่อสินค้า แต่ใช้แทนการฟังเสียงไม่ได้

ตอบ JSON object ตาม schema และทุก key:
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
     {"time_sec":2.2,"heard_fragment_th":"เสียงคำที่ได้ยินจริง","why_problem_th":"ทำไมฟังไม่เป็นคำหรือเพี้ยน","confidence":0.0}
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
decision สำหรับแต่ละด้าน: PASS, FAIL, UNCERTAIN
confidence คือความมั่นใจว่ามีข้อผิดพลาดรุนแรงจริง (0 เมื่อปกติ)
ห้ามสร้างหลักฐานให้ครบเกณฑ์ ห้ามรายงานคำพูดที่ฟังไม่ได้ว่าเป็นคำจริง
"""


def _run(args, timeout=120):
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AnalysisError(f"FFmpeg/FFprobe ทำงานไม่ได้: {exc}") from exc
    if p.returncode:
        raise AnalysisError(f"วิดีโอ/เสียงอ่านไม่ได้: {(p.stderr or '')[-400:]}")
    return p.stdout


def probe(path):
    raw = _run(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type", "-of", "json", str(path)])
    try:
        data = json.loads(raw)
        duration = float(data.get("format", {}).get("duration", 0))
        streams = data.get("streams", [])
    except (TypeError, ValueError, KeyError) as exc:
        raise AnalysisError("ไม่สามารถอ่านข้อมูลวิดีโอ") from exc
    if not math.isfinite(duration) or not 0.5 <= duration <= 25:
        raise AnalysisError(f"รองรับวิดีโอ 0.5–25 วินาที (ไฟล์นี้ {duration:.1f}s)")
    if not any(s.get("codec_type") == "video" for s in streams):
        raise AnalysisError("ไม่พบภาพในไฟล์วิดีโอ")
    return {"duration": duration,
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def _frames(path, folder, fps):
    folder.mkdir(parents=True, exist_ok=True)
    filt = (f"fps={fps},scale={FRAME_SIZE}:{FRAME_SIZE}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2")
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
          "-threads", "2", "-i", str(path), "-vf", filt,
          "-q:v", "5", str(folder / "frame_%03d.jpg")], 140)
    paths = sorted(folder.glob("frame_*.jpg"))[:MAX_FRAMES]
    if not paths:
        raise AnalysisError("ดึงภาพจากวิดีโอไม่สำเร็จ")
    return paths


def _sheets(frames, fps, folder):
    outputs = []
    for offset in range(0, len(frames), 12):
        canvas = Image.new("RGB", (4 * 260, 3 * 242), (24, 28, 39))
        draw = ImageDraw.Draw(canvas)
        for n, path in enumerate(frames[offset:offset + 12]):
            with Image.open(path) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (250, 208))
            x = (n % 4) * 260 + (260 - thumb.width) // 2
            y = (n // 4) * 242 + 22 + (208 - thumb.height) // 2
            canvas.paste(thumb, (x, y))
            draw.text(((n % 4) * 260 + 8, (n // 4) * 242 + 4),
                      f"{(offset + n) / fps:.1f}s", fill="white")
        dest = folder / f"sheet_{len(outputs):02d}.jpg"
        canvas.save(dest, "JPEG", quality=80, optimize=True)
        outputs.append(dest)
    return outputs


def _motion_samples(path, frames, fps, duration, folder):
    changes = []
    old = None
    for i, file in enumerate(frames):
        with Image.open(file) as im:
            cur = ImageOps.grayscale(im).resize((48, 48))
        if old is not None:
            score = ImageStat.Stat(ImageChops.difference(cur, old)).mean[0]
            changes.append((score, i / fps))
        old = cur
    changes.sort(reverse=True)
    picks = []
    for score, sec in changes:
        if score < 15 or sec > duration - .1 or sec < .15:
            continue
        if all(abs(sec - p) > .8 for p in picks):
            picks.append(sec)
        if len(picks) == 2:
            break
    out = []
    for i, sec in enumerate(sorted(picks)):
        dest = folder / f"motion_{i}.jpg"
        try:
            _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                  "-y", "-ss", f"{max(0, sec-.14):.2f}", "-i", str(path),
                  "-frames:v", "1", "-vf",
                  f"scale={FRAME_SIZE}:{FRAME_SIZE}:"
                  "force_original_aspect_ratio=decrease:force_divisible_by=2",
                  str(dest)], 30)
            if dest.is_file():
                out.append((sec, dest))
        except AnalysisError:
            pass  # ไม่ให้ภาพเสริมที่ดึงไม่ได้ทำคลิปทั้งงานล้ม
    return out


def _audio(path, folder, has_audio):
    if not has_audio:
        return None, {"has_audio": False}
    dest = folder / "audio.wav"
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
          "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
          "-c:a", "pcm_s16le", str(dest)], 75)
    with wave.open(str(dest), "rb") as w:
        samples = array("h")
        samples.frombytes(w.readframes(w.getnframes()))
    if not samples:
        return None, {"has_audio": True, "silent": True}
    rms = math.sqrt(sum(v * v for v in samples) / len(samples)) / 32768.0
    clipping = sum(abs(v) > 32300 for v in samples) / len(samples)
    stats = {"has_audio": True, "rms": round(rms, 4),
             "clipped_ratio": round(clipping, 4)}
    if rms < 0.001:
        return None, {**stats, "silent": True}
    return dest, stats


def _part(path, mime):
    return {"inline_data": {
        "mime_type": mime,
        "data": base64.b64encode(Path(path).read_bytes()).decode("ascii")
    }}


def _json_api(key, model, parts, max_tokens):
    if not key:
        raise AnalysisError("ไม่พบ GEMINI_API_KEY")
    if model not in ALLOWED_MODELS:
        raise AnalysisError(f"โมเดล {model} ไม่อยู่ในรายการที่อนุญาต")
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.1,
                                 "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json"}}
    for attempt in range(3):
        try:
            r = requests.post(url, headers={"x-goog-api-key": key,
                                            "Content-Type": "application/json"},
                              json=body, timeout=(15, 110))
        except requests.RequestException as exc:
            if attempt == 2:
                raise AnalysisError(f"เชื่อมต่อ Gemini ไม่สำเร็จ: {exc}") from exc
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429:
            raise QuotaError("Gemini ถึงโควตา API กรุณารอแล้วเริ่มคิวต่อ")
        if r.status_code in (500, 502, 503, 504) and attempt < 2:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", {}).get("message", "")
            except ValueError:
                msg = r.text[:300]
            raise AnalysisError(f"Gemini HTTP {r.status_code}: {str(msg)[:350]}")
        try:
            obj = r.json()
            chunks = obj["candidates"][0]["content"]["parts"]
            txt = "".join(str(x.get("text", "")) for x in chunks).strip()
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
            data = json.loads(txt)
            if not isinstance(data, dict):
                raise ValueError("ผลไม่ใช่ JSON object")
            return data
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AnalysisError(f"ผล JSON จาก Gemini ไม่สมบูรณ์: {exc}") from exc
    raise AnalysisError("Gemini ไม่ตอบกลับ")


def _number(v, default=0.0):
    try:
        value = float(v)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def _bool(v):
    return v is True or isinstance(v, str) and v.lower() == "true"


def _text(v, max_len=400):
    return str(v or "").strip()[:max_len]


def _evidence(v):
    s = _text(v)
    if len(s) < 14 or s.startswith(("ไม่พบ", "ไม่มี", "ปกติ", "ไม่ชัด", "ไม่แน่ใจ", "ยืนยันไม่ได้")):
        return ""
    return s


def _issue(result, category, reason, sec=0):
    issues = result.get("issues")
    if not isinstance(issues, list):
        issues = []
    issues = [x for x in issues if isinstance(x, dict)][:22]
    issues.append({"time_sec": round(max(0.0, _number(sec)), 2),
                   "category": category, "severity": "critical",
                   "detail_th": _text(reason, 570)})
    result["issues"] = issues


def _fail(result, category, reason, sec=0):
    previous = _text(result.get("reason_th"), 300)
    result["decision"] = "FAIL"
    result["reason_th"] = (_text(reason, 520) +
                           (" | " + previous if previous and previous != reason else ""))[:850]
    _issue(result, category, reason, sec)


def _first_pass(key, model, sheets, details, wav, duration, stats, reference):
    instruction = FIRST_PROMPT + f"\nคลิปยาว {duration:.2f} วินาที"
    if reference:
        instruction += "\nข้อมูลสินค้าที่ผู้ขายให้มา (อ้างอิง): " + reference[:900]
    instruction += "\nสถิติเสียง (ไม่ใช่ผลถอดคำพูด): " + json.dumps(stats)
    parts = [{"text": instruction}]
    for p in sheets:
        parts.extend([{"text": "ภาพรวมหลายช่วงเวลา"}, _part(p, "image/jpeg")])
    for sec, p in details:
        parts.extend([{"text": f"ภาพเดี่ยวที่เวลาประมาณ {sec:.2f} วินาที"},
                      _part(p, "image/jpeg")])
    if wav is not None:
        parts.extend([{"text": "เสียงจริงทั้งคลิป ตรวจคำพูดภาษาไทย"},
                      _part(wav, "audio/wav")])
    data = _json_api(key, model, parts, 3800)
    decision = _text(data.get("decision")).upper()
    kind = _text(data.get("video_type")).upper()
    if decision not in ("PASS", "FAIL") or kind not in ("HUMAN", "HANDS", "PRODUCT", "MIXED"):
        raise AnalysisError("Gemini ส่งคำตัดสินหรือประเภทคลิปไม่ถูกต้อง")
    scale = data.get("scale_review")
    audio = data.get("audio_review")
    if not isinstance(scale, dict) or (wav is not None and not isinstance(audio, dict)):
        raise AnalysisError("Gemini ส่งข้อมูลตรวจสเกลหรือเสียงไม่ครบ")
    if not isinstance(audio, dict):
        audio = {"status": "no_speech", "speech_present": False}
    return {
        "decision": decision, "video_type": kind,
        "product_name_th": _text(data.get("product_name_th"), 180),
        "reason_th": _text(data.get("reason_th"), 850),
        "speech_summary_th": _text(data.get("speech_summary_th"), 800),
        "issues": [i for i in data.get("issues", []) if isinstance(i, dict)][:22]
            if isinstance(data.get("issues"), list) else [],
        "scale_review": scale, "audio_review": audio,
        "audio_stats": stats, "scale_override": False,
        "audio_override": False, "scale_audit_override": False,
        "speech_audit_override": False,
    }


def _apply_first_safety(result):
    """If Gemini marked PASS despite its own severe structured evidence."""
    scale = result.get("scale_review", {})
    if (_text(scale.get("size_grade")).lower() == "wrong_scale"
            and _number(scale.get("confidence")) >= .82
            and not _bool(scale.get("miniature_disclosed"))
            and _evidence(scale.get("function_evidence_th") or scale.get("usage_evidence_th"))
            and (_evidence(scale.get("hand_evidence_th"))
                 or _evidence(scale.get("environment_evidence_th")))):
        was_pass = result["decision"] == "PASS"
        _fail(result, "product_scale", "รอบแรกพบสินค้าผิดสเกลชัดเจน: " +
              _text(scale.get("reason_th") or scale.get("function_evidence_th")))
        result["scale_override"] = was_pass
    audio = result.get("audio_review", {})
    if (_text(audio.get("status")).lower() == "critical"
            and _number(audio.get("confidence")) >= .80
            and _evidence(audio.get("evidence_th"))
            and (_bool(audio.get("severe_unintelligible"))
                 or _bool(audio.get("key_information_unintelligible"))
                 or _bool(audio.get("meaning_conflict"))
                 or _number(audio.get("broken_word_count")) >= 2)):
        was_pass = result["decision"] == "PASS"
        _fail(result, "audio", "รอบแรกพบเสียงพูดเสียรุนแรง: " +
              _text(audio.get("evidence_th")), audio.get("time_sec"))
        result["audio_override"] = was_pass
    return result


def _independent_audit(key, model, sheets, details, wav, duration, reference):
    prompt = AUDIT_PROMPT + f"\nคลิปยาว {duration:.2f} วินาที"
    if reference:
        prompt += "\nข้อมูลสินค้าที่ผู้ขายให้มา (อ้างอิง): " + reference[:900]
    parts = [{"text": prompt}]
    # ชุดภาพหลายเวลา เพื่อไม่หลงเชื่อเพียงเฟรมที่สเกลดูปกติ
    for p in sheets[:5]:
        parts.extend([{"text": "ภาพตามลำดับเวลาจากคลิป"}, _part(p, "image/jpeg")])
    for sec, p in details[:8]:
        parts.extend([{"text": f"ดูขนาดสินค้าและสิ่งที่มือจับ ณ {sec:.2f}s"},
                      _part(p, "image/jpeg")])
    if wav is not None:
        parts.extend([{"text": "เสียงพูดจริงทั้งคลิป: ต้องฟังเพื่อหาคำเพี้ยน"},
                      _part(wav, "audio/wav")])
    data = _json_api(key, model, parts, 3100)
    scale, speech = data.get("scale"), data.get("speech")
    if not isinstance(scale, dict) or not isinstance(speech, dict):
        raise AnalysisError("ผลตรวจยืนยันไม่มี scale/speech")
    if _text(scale.get("decision")).upper() not in ("PASS", "FAIL", "UNCERTAIN"):
        raise AnalysisError("ผลตรวจสเกลยืนยันไม่ถูกต้อง")
    if _text(speech.get("decision")).upper() not in ("PASS", "FAIL", "UNCERTAIN"):
        raise AnalysisError("ผลตรวจเสียงยืนยันไม่ถูกต้อง")
    return scale, speech


def _apply_audit(result, scale, speech, duration):
    result["scale_audit"] = scale
    result["speech_audit"] = speech
    result["audit_done"] = True
    reasons = []

    # สเกลผิดต้องขัดกับการใช้งานจริงและมีหลักฐานภาพประกอบ
    confidence = _number(scale.get("confidence"))
    hand = _evidence(scale.get("hand_evidence_th"))
    env = _evidence(scale.get("environment_evidence_th"))
    function = _evidence(scale.get("function_evidence_th"))
    confirmed_mini = (_bool(scale.get("intentional_mini"))
                      and bool(_evidence(scale.get("mini_evidence_th"))))
    confirmed_scale = (
        _text(scale.get("decision")).upper() == "FAIL"
        and confidence >= .77 and bool(function) and bool(hand or env)
        and not confirmed_mini
    )
    a = _number(scale.get("time_a"), -1)
    b = _number(scale.get("time_b"), -1)
    morph = (
        _text(scale.get("decision")).upper() == "FAIL"
        and _bool(scale.get("impossible_size_change"))
        and confidence >= .86
        and 0 <= a < b <= duration and b - a >= .3
        and bool(_evidence(scale.get("temporal_evidence_th")))
    )
    if confirmed_scale or morph:
        detail = (_text(scale.get("reason_th")) or
                  "; ".join(t for t in (hand, env, function) if t))
        reasons.append("สเกลสินค้าผิดรุนแรงจากการตรวจยืนยัน: " + detail)
        result["scale_audit_override"] = result["decision"] == "PASS"
        _fail(result, "product_scale", reasons[-1], a if morph else 0)

    # เสียงผิด >=2 คำ ต้องมีหลักฐานแยกคำ หรือช่วงเสียงมั่วต่อเนื่อง
    score = _number(speech.get("confidence"))
    valid_events = []
    stamps = []
    events = speech.get("events", [])
    for event in (events[:20] if isinstance(events, list) else []):
        if not isinstance(event, dict):
            continue
        sec = _number(event.get("time_sec"), -1)
        fragment = _text(event.get("heard_fragment_th"), 100)
        why = _evidence(event.get("why_problem_th"))
        ec = _number(event.get("confidence"))
        if (0 <= sec <= duration and fragment and why and ec >= .75
                and all(abs(sec - other) >= .10 for other in stamps)):
            valid_events.append((sec, fragment, why))
            stamps.append(sec)

    start = _number(speech.get("continuous_start_sec"), -1)
    end = _number(speech.get("continuous_end_sec"), -1)
    continuous = (
        _bool(speech.get("continuous_gibberish")) and score >= .82
        and _number(speech.get("continuous_broken_word_count")) >= 2
        and 0 <= start < end <= duration and end - start >= .35
        and bool(_evidence(speech.get("continuous_evidence_th")))
    )
    broken_words = len(valid_events) >= 2 and score >= .77
    spoken = _evidence(speech.get("spoken_claim_th"))
    visual = _evidence(speech.get("visual_or_reference_evidence_th"))
    mismatch = (_bool(speech.get("critical_claim_verified"))
                and score >= .87 and bool(spoken and visual))
    severe_audio = (
        _text(speech.get("decision")).upper() == "FAIL"
        and _bool(speech.get("speech_present"))
        and (continuous or broken_words or mismatch)
    )
    if severe_audio:
        if broken_words:
            why = "; ".join(f"{sec:.1f}s {frag}: {explanation}"
                            for sec, frag, explanation in valid_events[:3])
            at = valid_events[0][0]
            cause = "เสียงพูดเพี้ยนอย่างน้อย 2 คำ: " + why
        elif continuous:
            at = start
            cause = "เสียงพูดมั่วหลายคำต่อเนื่อง: " + _text(speech.get("continuous_evidence_th"))
        else:
            at = 0.0
            cause = "คำพูดขัดกับสินค้าที่ตรวจยืนยันได้: " + spoken + " / " + visual
        result["speech_audit_override"] = result["decision"] == "PASS"
        _fail(result, "audio", "ตรวจเสียงยืนยันพบปัญหารุนแรง: " + cause, at)
        audio = dict(result.get("audio_review") or {})
        audio.update({"status": "critical", "speech_present": True,
                      "confidence": score, "time_sec": at,
                      "evidence_th": cause[:500],
                      "severe_unintelligible": bool(continuous or broken_words),
                      "meaning_conflict": bool(mismatch)})
        result["audio_review"] = audio
    if _text(speech.get("speech_summary_th")):
        result["speech_summary_th"] = _text(speech.get("speech_summary_th"), 800)
    if result["decision"] == "PASS" and (
            _text(scale.get("decision")).upper() == "UNCERTAIN"
            or _text(speech.get("decision")).upper() == "UNCERTAIN"):
        result["audit_warning_th"] = "มีข้อสงสัยแต่หลักฐานไม่พอบังคับ FAIL; โปรดตรวจด้วยคนถ้าสำคัญ"
        result["reason_th"] = (_text(result.get("reason_th"), 620)
                               + " | ⚠️ สเกลหรือเสียงยังมีข้อสงสัย ควรตรวจด้วยคน")[:850]
    return result


def inspect(path, key, model, mode, thumbnail, reference=""):
    """Main entry point: same 5 positional args as V3 qc_store.py."""
    start = time.perf_counter()
    path = Path(path)
    thumbnail = Path(thumbnail)
    metadata = probe(path)
    fps = FPS.get(mode, 2)
    with tempfile.TemporaryDirectory(prefix="qc_v33_") as tmp:
        folder = Path(tmp)
        frames = _frames(path, folder / "frames", fps)
        thumbnail.parent.mkdir(parents=True, exist_ok=True)
        if not thumbnail.is_file():
            with Image.open(frames[0]) as im:
                small = ImageOps.contain(im.convert("RGB"), (300, 300))
                small.save(thumbnail, "JPEG", quality=75)
        sheets = _sheets(frames, fps, folder)
        selection = sorted(set(round((len(frames) - 1) * pos)
                               for pos in (.06, .23, .40, .57, .74, .93)))
        details = [(i / fps, frames[i]) for i in selection]
        if mode != "Fast":
            details += _motion_samples(path, frames, fps,
                                       metadata["duration"], folder)
        wav, audio_stats = _audio(path, folder, metadata["has_audio"])
        result = _first_pass(key, model, sheets, details, wav,
                             metadata["duration"], audio_stats,
                             str(reference or ""))
        result = _apply_first_safety(result)
        result["audit_done"] = False
        if RECHECK_EVERY_PASS and result["decision"] == "PASS":
            scale, speech = _independent_audit(
                key, model, sheets, details, wav,
                metadata["duration"], str(reference or ""))
            result = _apply_audit(result, scale, speech,
                                  metadata["duration"])
    result["duration"] = round(metadata["duration"], 2)
    result["elapsed"] = round(time.perf_counter() - start, 2)
    return result
