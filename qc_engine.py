
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

SUPPORTED = {
    ".mp4", ".mov", ".m4v", ".webm", ".avi"
}

MAX_TOTAL_BYTES = 1_000_000_000


def command(args, timeout=35):
    try:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=True
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"โปรแกรมอ่านไฟล์ทำงานเกิน {timeout} วินาที"
        ) from exc

    except subprocess.CalledProcessError as exc:
        detail = (
            exc.stderr or ""
        )[-350:].replace("\n", " ")

        raise RuntimeError(
            f"อ่าน/แปลงไฟล์ไม่สำเร็จ: {detail}"
        ) from exc


def probe_video(path):

    raw = command(
        [
            "ffprobe",
            "-v", "error",
            "-show_streams",
            "-show_format",
            "-of", "json",
            str(path)
        ],
        timeout=20
    ).stdout

    data = json.loads(raw)
    streams = data.get("streams", [])

    videos = [
        s for s in streams
        if s.get("codec_type") == "video"
    ]

    if not videos:
        raise ValueError(
            "ไฟล์นี้ไม่มีวิดีโอที่อ่านได้"
        )

    video = videos[0]

    duration = float(
        video.get("duration")
        or data.get(
            "format", {}
        ).get("duration")
        or 0
    )

    if not 0 < duration <= 600:
        raise ValueError(
            "ระยะเวลาวิดีโอไม่ถูกต้อง "
            "หรือยาวเกิน 10 นาที"
        )

    return {
        "duration": round(duration, 2),
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "has_audio": any(
            s.get("codec_type") == "audio"
            for s in streams
        ),
        "codec": video.get(
            "codec_name", ""
        )
    }


def sample_frames(
    path,
    duration,
    samples_per_second=2,
    max_frames=40
):
    count = min(
        max_frames,
        max(
            2,
            int(duration * samples_per_second)
        )
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(path),
        "-vf",
        f"fps={samples_per_second},scale=384:-2",
        "-frames:v", str(count),
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "5",
        "pipe:1"
    ]

    try:
        run = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "อ่านภาพใช้เวลานานผิดปกติ"
        ) from exc

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "ffmpeg อ่านภาพไม่สำเร็จ: "
            + exc.stderr[-200:].decode(
                "utf-8",
                errors="replace"
            )
        ) from exc

    raw = run.stdout
    frames = []
    offset = 0

    while len(frames) < count:

        begin = raw.find(
            b"\xff\xd8",
            offset
        )

        if begin == -1:
            break

        end = raw.find(
            b"\xff\xd9",
            begin + 2
        )

        if end == -1:
            break

        jpeg = np.frombuffer(
            raw[begin:end + 2],
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            jpeg,
            cv2.IMREAD_COLOR
        )

        if frame is not None:
            frames.append(
                (
                    round(
                        len(frames)
                        / samples_per_second,
                        2
                    ),
                    frame
                )
            )

        offset = end + 2

    if len(frames) < 2:
        raise ValueError(
            "อ่านเฟรมวิดีโอได้น้อยเกินไป"
        )

    return frames


def visual_scan(frames):

    observations = []

    previous = None
    freeze_start = None
    black_run = 0

    for sec, bgr in frames:

        gray = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY
        )

        thumb = cv2.resize(
            gray,
            (96, 96),
            interpolation=cv2.INTER_AREA
        )

        black_frac = float(
            np.mean(thumb < 8)
        )

        if black_frac > 0.98:
            black_run += 1
        else:
            black_run = 0

        if black_run >= 3:
            observations.append(
                "ช่วงภาพเกือบดำต่อเนื่อง"
                f"บริเวณ {sec:.1f}s"
            )
            black_run = 0

        if previous is not None:

            diff = float(
                np.mean(
                    cv2.absdiff(
                        thumb,
                        previous
                    )
                )
            )

            if diff < 0.4:

                if freeze_start is None:
                    freeze_start = sec

                elif sec - freeze_start >= 2.0:
                    observations.append(
                        "ภาพแทบไม่เปลี่ยนช่วง "
                        f"{freeze_start:.1f}-"
                        f"{sec:.1f}s "
                        "(อาจเป็นฉากนิ่งปกติ)"
                    )

                    freeze_start = sec + 999

            else:
                freeze_start = None

        previous = thumb

    return list(
        dict.fromkeys(observations)
    )[:6]


def wav_audio(
    video_path,
    wav_path
):
    command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(wav_path)
        ],
        timeout=35
    )


@lru_cache(maxsize=1)
def whisper_model(size):

    from faster_whisper import WhisperModel

    return WhisperModel(
        size,
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1
    )


def transcribe_thai(
    wav_path,
    model_size="tiny"
):
    model = whisper_model(
        model_size
    )

    segments, info = model.transcribe(
        str(wav_path),
        language="th",
        beam_size=1,
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=False
    )

    items = [
        {
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip()
        }
        for s in segments
        if s.text.strip()
    ]

    return (
        " ".join(
            i["text"]
            for i in items
        ),
        items
    )


def normalize(text):
    return re.sub(
        r'[\s\-_,.!?()\[\]"\']+',
        "",
        str(text or "").casefold()
    )


def match_required(
    transcript,
    required
):
    spoken = normalize(
        transcript
    )

    return [
        term
        for term in required
        if (
            term
            and normalize(term) not in spoken
        )
    ]


def read_json(text):

    text = str(
        text or ""
    ).strip()

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.I
    )

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        raise ValueError(
            "AI ไม่คืน JSON ที่อ่านได้"
        )

    return json.loads(
        text[start:end + 1]
    )


def gemini_inspect(
    video_path,
    frames,
    wav_path,
    meta,
    product,
    brand,
    required,
    api_key,
    model_name
):

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options={
            "timeout": 65000
        }
    )

    prompt = f"""
คุณคือผู้ตรวจ QC คลิป AI รีวิวสินค้า 10 วินาที
ตอบ JSON ภาษาไทยเท่านั้น

งาน:
ฟังเสียงพูดจริงทุกคำ
ตรวจชื่อแบรนด์ รุ่น ปริมาณ คุณสมบัติ
ตรงกับภาพและข้อมูลอ้างอิงหรือไม่

ตรวจภาพตามเวลา:
มือ นิ้ว ใบหน้า วัตถุทะลุ หาย ปรากฏเพิ่ม
รูปร่างเปลี่ยน และการเคลื่อนไหวผิดธรรมชาติ

แยกความเพี้ยนเล็กน้อยที่ยังเข้าใจได้ (PASS)
กับข้อผิดพลาดชัดเจน (FAIL)
และเรื่องยืนยันไม่ได้ (REVIEW)

ห้ามเดาแบรนด์จากภาพที่อ่านไม่ออก
ห้ามสร้างคำพูดเอง
หากได้ยินไม่ชัดให้ REVIEW

ถ้าจะ FAIL ต้องมีหลักฐานเจาะจง
ที่สังเกตได้ พร้อมวินาทีที่ใกล้เคียง

สินค้าอ้างอิง: {product or 'ไม่ได้ระบุ'}
ยี่ห้อที่ถูกต้อง: {brand or 'ไม่ได้ระบุ'}
คำสำคัญที่คาดว่าจะพูด:
{', '.join(required) or 'ไม่มี'}

ตอบ JSON object พร้อม keys:
verdict (PASS/FAIL/REVIEW)
transcript (คำพูดที่ได้ยินหรือว่างถ้าไม่แน่ใจ)
visual_findings (array ของ string)
audio_findings (array ของ string)
issues (array ของ object:
  category เป็น visual/audio/product,
  timestamp เป็นตัวเลขวินาที,
  detail เป็น string,
  severity เป็น minor/major/uncertain)
summary (string)
confidence (เลข 0-1)

ต้องดูและฟังสื่อจริง
ไม่ใช่ประเมินจากชื่อไฟล์
หรือคำแนะนำนี้เพียงอย่างเดียว
"""

    parts = [prompt]

    if (
        video_path.suffix.lower() == ".mp4"
        and video_path.stat().st_size <= 18_000_000
    ):

        parts.append(
            types.Part.from_bytes(
                data=video_path.read_bytes(),
                mime_type="video/mp4"
            )
        )

        mode = "video+audio"

    else:

        step = max(
            1,
            len(frames) // 12
        )

        for sec, bgr in frames[::step][:12]:

            ok, jpeg = cv2.imencode(
                ".jpg",
                bgr,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    72
                ]
            )

            if ok:
                parts.append(
                    f"ภาพที่เวลา {sec} วินาที"
                )

                parts.append(
                    types.Part.from_bytes(
                        data=jpeg.tobytes(),
                        mime_type="image/jpeg"
                    )
                )

        if (
            wav_path
            and wav_path.exists()
            and wav_path.stat().st_size < 5_000_000
        ):
            parts.append(
                types.Part.from_bytes(
                    data=wav_path.read_bytes(),
                    mime_type="audio/wav"
                )
            )

        mode = "sampled-frames+audio"

    response = client.models.generate_content(
        model=model_name,
        contents=parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            max_output_tokens=1800
        )
    )

    payload = read_json(
        response.text
    )

    if payload.get("verdict") not in (
        "PASS",
        "FAIL",
        "REVIEW"
    ):
        raise ValueError(
            "AI ส่งสถานะผิดรูปแบบ"
        )

    return payload, mode


def inspect_video(
    path,
    filename,
    settings,
    reference=None
):

    started = time.monotonic()

    reference = reference or {}

    product = str(
        reference.get("product", "") or ""
    )

    brand = str(
        reference.get("brand", "") or ""
    )

    required = [
        t.strip()
        for t in str(
            reference.get(
                "required_terms", ""
            ) or ""
        ).split(";")
        if t.strip()
    ]

    meta = probe_video(path)

    frames = sample_frames(
        path,
        meta["duration"],
        settings.get("fps", 2)
    )

    observations = visual_scan(
        frames
    )

    transcript = ""
    segments = []
    audio_findings = []
    ai_mode = "off"
    ai_payload = None

    with tempfile.TemporaryDirectory(
        prefix="qc_audio_"
    ) as tmp:

        wav_path = (
            Path(tmp) / "audio.wav"
        )

        audio_path = None

        if meta["has_audio"]:

            try:
                wav_audio(
                    path,
                    wav_path
                )

                audio_path = wav_path

            except Exception as exc:
                audio_findings.append(
                    "ดึงเสียงไม่สำเร็จ: "
                    f"{exc}"
                )

        else:
            audio_findings.append(
                "ไม่พบแทร็กเสียง "
                "(ตรวจว่าคลิปตั้งใจไม่มีเสียงหรือไม่)"
            )

        if (
            settings.get("asr")
            and audio_path is not None
        ):

            try:
                transcript, segments = transcribe_thai(
                    audio_path,
                    settings.get(
                        "whisper", "tiny"
                    )
                )

            except Exception as exc:
                audio_findings.append(
                    "ถอดเสียงในเครื่องไม่สำเร็จ: "
                    f"{exc}"
                )

        if settings.get("api_key"):

            try:
                ai_payload, ai_mode = gemini_inspect(
                    path,
                    frames,
                    audio_path,
                    meta,
                    product,
                    brand,
                    required,
                    settings["api_key"],
                    settings["gemini_model"]
                )

            except Exception as exc:
                audio_findings.append(
                    "AI ตรวจเชิงลึกไม่สำเร็จ: "
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:160]}"
                )

    issues = []

    if ai_payload:

        verdict = ai_payload["verdict"]

        transcript = str(
            ai_payload.get("transcript")
            or transcript
        )

        issues = (
            ai_payload.get("issues")
            if isinstance(
                ai_payload.get("issues"),
                list
            )
            else []
        )

        audio_findings += [
            str(x)
            for x in ai_payload.get(
                "audio_findings", []
            )
            if isinstance(x, str)
        ]

        observations += [
            str(x)
            for x in ai_payload.get(
                "visual_findings", []
            )
            if isinstance(x, str)
        ]

        reason = str(
            ai_payload.get("summary")
            or "ตรวจด้วย AI แล้ว"
        )

        if verdict == "FAIL":

            has_evidence = any(
                isinstance(x, dict)
                and x.get("detail")
                and x.get("timestamp") is not None
                for x in issues
            )

            if not has_evidence:
                verdict = "REVIEW"
                reason = (
                    "AI ระบุว่าไม่ผ่าน "
                    "แต่ยังไม่มีหลักฐานระบุช่วงเวลาชัดเจน"
                )

        if (
            verdict == "PASS"
            and any(
                "ไม่สำเร็จ" in x
                for x in audio_findings
            )
        ):
            verdict = "REVIEW"
            reason = (
                "มีขั้นตอนตรวจเสียงไม่สำเร็จ "
                "จึงยังไม่อนุมัติอัตโนมัติ"
            )

    else:

        verdict = "REVIEW"

        reason = (
            "โหมดฟรีตรวจสัญญาณได้ "
            "แต่ยังยืนยันความสมจริงและ"
            "ความหมายเสียงระดับคนไม่ได้"
        )

        if not settings.get("asr"):
            audio_findings.append(
                "ยังไม่ได้ถอดคำพูด: "
                "ต้องใช้โหมด AI "
                "หรือเปิดการถอดเสียงในเครื่อง"
            )

    missing = (
        match_required(
            transcript,
            required
        )
        if transcript
        else []
    )

    if missing:

        audio_findings.append(
            "คำสำคัญที่ยังไม่พบในผลถอดเสียง "
            "(ต้องฟังยืนยัน): "
            + ", ".join(missing)
        )

        if verdict == "PASS":
            verdict = "REVIEW"

            reason = (
                "คำสำคัญบางคำไม่ปรากฏ"
                "ในผลถอดเสียง ต้องตรวจด้วยคน"
            )

    return {
        "filename": filename,
        "status": verdict,
        "reason": reason,
        "duration_s": meta["duration"],
        "process_s": round(
            time.monotonic() - started,
            2
        ),
        "resolution": (
            f"{meta['width']}x{meta['height']}"
        ),
        "transcript": transcript,
        "segments": segments,
        "visual_findings": list(
            dict.fromkeys(observations)
        ),
        "audio_findings": list(
            dict.fromkeys(audio_findings)
        ),
        "issues": issues[:15],
        "mode": ai_mode,
        "product": product,
        "brand": brand
    }


def save_upload(
    uploaded,
    destination,
    max_bytes
):

    total = 0
    uploaded.seek(0)

    with open(
        destination,
        "wb"
    ) as out:

        while True:

            block = uploaded.read(
                1024 * 1024
            )

            if not block:
                break

            total += len(block)

            if total > max_bytes:
                raise ValueError(
                    "ไฟล์ใหญ่เกินขนาดที่กำหนด"
                )

            out.write(block)

    return total
