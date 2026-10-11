
"""Video quality analysis. FFmpeg/Pillow + Gemini REST (no paid SDK)."""
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

ENGINE_VERSION = "2026-10-11-v1"
FPS = {"Fast": 1, "Balanced": 2, "Deep": 3}
ALLOWED_MODELS = {
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
}

PROMPT = """คุณเป็นผู้ตรวจคุณภาพวิดีโอโฆษณาสินค้าที่สร้างด้วย AI
วัตถุประสงค์: คัดคลิปที่ใช้โฆษณาได้จริง ไม่ใช่จับผิดว่าเป็น AI หรือไม่
คลิปมีความยาวประมาณ 10 วินาที ภาพที่ส่งให้เป็น contact sheet เรียงตามเวลา (แต่ละภาพมีป้ายเวลา) และอาจมีเสียงต้นฉบับ

จงจำแนกประเภท:
คนขายสินค้า (HUMAN)
มือพรีเซนสินค้า (HANDS)
สินค้าอย่างเดียว (PRODUCT)
หรือสลับประเภท (MIXED)

ตรวจอย่างมีเหตุผลตามสิ่งที่มองเห็นหรือได้ยินจริง:
ใบหน้า รูปร่าง มือ/นิ้ว การจับสินค้า ขนาดและรูปทรงสินค้า
สี โลโก้และฉลากเมื่ออ่านได้ การเปลี่ยนรูปร่างระหว่างช่วง
การเคลื่อนไหว ฉาก เสียงภาษาไทย เสียงแตก/สะดุด
และความสัมพันธ์ของคำพูดกับสินค้าหากมีเสียง

อย่าคิดว่าการเปลี่ยนมุมกล้องหรือซูมเป็นขนาดสินค้าเพี้ยน
อย่าตัดตกเพราะเป็น AI หรือเพราะข้อผิดพลาดเล็กมากจนไม่รบกวนการดู

ตัดสิน FAIL ก็ต่อเมื่อมีข้อบกพร่องชัดเจนระดับส่งผลต่อคุณภาพโฆษณา
เช่น มือผิดรูปชัด สินค้าเปลี่ยนรูปร่างผิดปกติ
เสียงขาด/ผิดเพี้ยนจนรบกวน การเคลื่อนไหวพัง

หากคลิปโดยรวมดูดี ให้ PASS
ถ้าไม่มีคน ห้ามอ้างว่าปากไม่ตรงเสียง
ถ้าไม่มีเสียง ห้ามบอกว่าเสียงผิด
ถ้าข้อมูลไม่พอ อย่าแต่งเหตุผล

การวิเคราะห์ temporal จาก contact sheet เป็นการประมาณ
ไม่สามารถยืนยันเหตุการณ์ระหว่างเฟรมที่ไม่ได้เห็น

คืน JSON เท่านั้น โดยใช้ฟิลด์ต่อไปนี้:
{
 "decision": "PASS หรือ FAIL",
 "video_type": "HUMAN|HANDS|PRODUCT|MIXED",
 "reason_th": "เหตุผลสรุปสั้นภาษาไทย",
 "issues": [
   {
    "time_sec": 0.0,
    "category": "person|hand|product|motion|audio|other",
    "severity": "minor|moderate|critical",
    "detail_th": "คำอธิบาย"
   }
 ],
 "speech_summary_th": "สรุปคำพูดภาษาไทยถ้ามี ไม่ต้องเดาหากไม่มี"
}

โปรดให้ time_sec อยู่ในช่วง 0 ถึงระยะเวลาคลิป
และหลีกเลี่ยงการระบุวินาทีแม่นเกินกว่าที่ภาพตัวอย่างรองรับ
"""


class AnalysisError(RuntimeError):
    pass


class QuotaError(AnalysisError):
    pass


def command(args: list[str], timeout: int = 90) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AnalysisError(
            f"FFmpeg/FFprobe ไม่พร้อมหรือหมดเวลา: {exc}"
        ) from exc

    if p.returncode != 0:
        raise AnalysisError(
            (p.stderr or "FFmpeg ผิดพลาด")[-450:]
        )

    return p


def probe(path: Path) -> dict:
    data = json.loads(
        command([
            "ffprobe", "-v", "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of", "json", str(path)
        ]).stdout
    )

    duration = float(
        data.get("format", {}).get("duration", 0) or 0
    )

    streams = data.get("streams", [])

    if not any(
        s.get("codec_type") == "video" for s in streams
    ):
        raise AnalysisError(
            "ไม่มีแทร็กวิดีโอที่อ่านได้"
        )

    if not 0.5 <= duration <= 25:
        raise AnalysisError(
            f"รองรับคลิปประมาณ 10 วินาที "
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

    result = sorted(
        folder.glob("f_*.jpg")
    )

    if not result:
        raise AnalysisError(
            "ไม่สามารถอ่านเฟรมวิดีโอได้"
        )

    return result[:80]


def motion_peaks(
    frames: list[Path],
    fps: int,
    duration: float,
) -> list[float]:

    # Rough changes, not proof of defects.
    changes = []
    previous = None

    for index, path in enumerate(frames):
        with Image.open(path) as im:
            gray = ImageOps.grayscale(im).resize(
                (48, 48)
            )

        if previous is not None:
            score = ImageStat.Stat(
                ImageChops.difference(
                    gray, previous
                )
            ).mean[0]

            if score > 48:
                changes.append((
                    score,
                    min(
                        duration - 0.1,
                        index / fps
                    )
                ))

        previous = gray

    changes.sort(reverse=True)

    out = []

    for _, t in changes:
        if t > 0.2 and all(
            abs(t - v) > 0.7 for v in out
        ):
            out.append(t)

        if len(out) == 2:
            break

    return sorted(out)


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

        return out.exists()

    except AnalysisError:
        return False


def contact_sheets(
    frames: list[Path],
    fps: int,
    folder: Path,
) -> list[Path]:

    result = []

    for start in range(
        0, len(frames), 10
    ):
        sheet = Image.new(
            "RGB",
            (5 * 244, 2 * 280),
            (22, 27, 38),
        )

        draw = ImageDraw.Draw(sheet)

        for i, frame_path in enumerate(
            frames[start:start + 10]
        ):
            index = start + i

            with Image.open(frame_path) as im:
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

        outfile = (
            folder
            / f"sheet_{len(result):02d}.jpg"
        )

        sheet.save(
            outfile,
            "JPEG",
            quality=78,
            optimize=True,
        )

        result.append(outfile)

    return result


def extract_audio(
    path: Path,
    folder: Path,
    has_audio: bool,
) -> tuple[Path | None, dict]:

    if not has_audio:
        return None, {
            "has_audio": False
        }

    wav_path = folder / "audio.wav"

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
        str(wav_path),
    ], timeout=60)

    with wave.open(
        str(wav_path), "rb"
    ) as wav:

        if wav.getsampwidth() != 2:
            raise AnalysisError(
                "รูปแบบเสียงที่แปลงมาไม่ถูกต้อง"
            )

        values = array("h")

        values.frombytes(
            wav.readframes(
                wav.getnframes()
            )
        )

    if not values:
        return None, {
            "has_audio": True,
            "silent": True,
        }

    rms = math.sqrt(
        sum(v * v for v in values)
        / len(values)
    ) / 32768.0

    clipped = sum(
        abs(v) >= 32300
        for v in values
    ) / len(values)

    return wav_path, {
        "has_audio": True,
        "rms": round(rms, 4),
        "clipped_ratio": round(
            clipped, 4
        ),
        "possibly_silent": rms < 0.002,
    }


def inline_part(
    path: Path,
    mime: str,
) -> dict:

    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(
                path.read_bytes()
            ).decode("ascii"),
        }
    }


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
            "โมเดลนี้ไม่ได้อยู่ในรายการที่แอปอนุญาต"
        )

    parts = [{
        "text": (
            PROMPT
            + f"\nระยะเวลาคลิป: "
            f"{duration:.2f} วินาที; "
            f"สถิติเสียง: "
            f"{json.dumps(audio_stats, ensure_ascii=False)}"
        )
    }]

    for sheet in sheets:
        parts.append({
            "text": "ภาพตามลำดับเวลา (timeline):"
        })

        parts.append(
            inline_part(
                sheet,
                "image/jpeg",
            )
        )

    for sec, image_file in details:
        parts.append({
            "text": (
                f"ภาพขยายช่วงประมาณ "
                f"{sec:.2f} วินาที:"
            )
        })

        parts.append(
            inline_part(
                image_file,
                "image/jpeg",
            )
        )

    if wav is not None:
        parts.append({
            "text": (
                "เสียงจากคลิปเดียวกัน "
                "ใช้ตรวจคำพูดภาษาไทยและคุณภาพเสียง:"
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
            "maxOutputTokens": 1600,
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
                "Gemini API ถึงโควตาหรือจำกัดความถี่แล้ว "
                "กรุณารอและกดเริ่มต่อ"
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
                message = (
                    response.json()
                    .get("error", {})
                    .get(
                        "message",
                        response.text[:300],
                    )
                )
            except ValueError:
                message = response.text[:300]

            raise AnalysisError(
                f"Gemini HTTP "
                f"{response.status_code}: "
                f"{message[:350]}"
            )

        try:
            obj = response.json()

            text = "".join(
                p.get("text", "")
                for p in obj["candidates"][0]
                ["content"]["parts"]
            )

            text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text.strip(),
            )

            report = json.loads(text)

            decision = str(
                report["decision"]
            ).upper()

            kind = str(
                report["video_type"]
            ).upper()

            if (
                decision not in ("PASS", "FAIL")
                or kind not in (
                    "HUMAN",
                    "HANDS",
                    "PRODUCT",
                    "MIXED",
                )
            ):
                raise ValueError(
                    "สถานะหรือประเภทวิดีโอไม่ถูกต้อง"
                )

            if not isinstance(
                report.get("issues", []),
                list,
            ):
                raise ValueError(
                    "issues ต้องเป็น list"
                )

            return {
                "decision": decision,
                "video_type": kind,
                "reason_th": (
                    str(
                        report.get(
                            "reason_th", ""
                        )
                    )[:800]
                    or "ไม่มีคำอธิบายจาก AI"
                ),
                "issues": [
                    x for x in report.get(
                        "issues", []
                    )
                    if isinstance(x, dict)
                ][:25],
                "speech_summary_th": str(
                    report.get(
                        "speech_summary_th", ""
                    )
                )[:800],
                "audio_stats": audio_stats,
            }

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:

            raise AnalysisError(
                "Gemini ส่งผลลัพธ์ JSON "
                f"ไม่สมบูรณ์: {exc}"
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
        prefix="qc_frames_"
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

        extras = []

        if mode != "Fast":
            for n, t in enumerate(
                motion_peaks(
                    frames,
                    fps,
                    meta["duration"],
                )
            ):
                out = (
                    folder
                    / f"detail_{n}.jpg"
                )

                if extra_frame(
                    path,
                    max(
                        0.0,
                        t - 0.18,
                    ),
                    out,
                ):
                    extras.append(
                        (t, out)
                    )

        wav, audio_stats = extract_audio(
            path,
            folder,
            meta["has_audio"],
        )

        result = gemini_analyze(
            key,
            model,
            sheets,
            extras,
            wav,
            meta["duration"],
            audio_stats,
        )

    result["duration"] = round(
        meta["duration"], 2
    )

    result["elapsed"] = round(
        time.perf_counter() - start,
        2,
    )

    return result
