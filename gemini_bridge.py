
"""Connect the existing video checker to Google Gemini."""
from __future__ import annotations

import base64
from types import SimpleNamespace

from google import genai
from google.genai import types


class GeminiAdapter:
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        if not api_key or not api_key.strip():
            raise ValueError("ยังไม่ได้ตั้งค่า GEMINI_API_KEY")

        self.model = model
        self.client = genai.Client(api_key=api_key.strip())

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._chat)
        )
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._transcribe)
        )

    @staticmethod
    def _parts(messages: list[dict]) -> list:
        parts = []

        for msg in messages:
            content = msg.get("content", "")

            if isinstance(content, str):
                if content.strip():
                    prefix = (
                        "คำสั่งระบบ: "
                        if msg.get("role") == "system" else ""
                    )
                    parts.append(
                        types.Part.from_text(text=prefix + content)
                    )
                continue

            for item in content:
                kind = item.get("type")

                if kind == "text":
                    parts.append(
                        types.Part.from_text(
                            text=item.get("text", "")
                        )
                    )

                elif kind == "image_url":
                    url = item.get("image_url", {}).get("url", "")

                    if not url.startswith("data:image/") or ";base64," not in url:
                        raise ValueError("รองรับภาพ base64 เท่านั้น")

                    header, payload = url.split(",", 1)
                    mime_type = header.split(";", 1)[0][5:]

                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(
                                payload, validate=True
                            ),
                            mime_type=mime_type
                        )
                    )

                elif kind == "input_audio":
                    block = item.get("input_audio", {})

                    if block.get("format") != "wav":
                        raise ValueError("รองรับเสียง WAV เท่านั้น")

                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(
                                block["data"], validate=True
                            ),
                            mime_type="audio/wav"
                        )
                    )

                else:
                    raise ValueError(
                        f"ข้อมูลที่ส่งให้ AI ไม่รองรับ: {kind}"
                    )

        if not parts:
            raise ValueError("ไม่มีข้อมูลสำหรับวิเคราะห์")

        return parts

    def _chat(
        self,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
        **kwargs
    ):
        parts = self._parts(messages)

        config = {
            "response_mime_type": "application/json",
            "thinking_config": types.ThinkingConfig(
                thinking_level="low"
            ),
            "temperature": 0,
        }

        schema = (
            (response_format or {})
            .get("json_schema", {})
            .get("schema")
        )

        if schema:
            config["response_json_schema"] = schema

        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(**config)
        )

        raw = (getattr(response, "text", None) or "").strip()

        if not raw:
            raise ValueError(
                "Gemini ไม่ส่งผลกลับมา กรุณาตรวจสอบ API และโควตา"
            )

        candidates = getattr(response, "candidates", None) or []
        finish = (
            getattr(candidates[0], "finish_reason", None)
            if candidates else None
        )

        name = (
            getattr(finish, "name", str(finish))
            if finish else "STOP"
        )

        status = (
            "stop"
            if str(name).upper().endswith("STOP")
            else "length"
        )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=raw,
                        refusal=None
                    ),
                    finish_reason=status
                )
            ]
        )

    def _transcribe(self, model: str, file, **kwargs):
        wav = file.read()

        if not wav:
            raise ValueError("ไฟล์เสียงว่าง")

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_text(
                    text=(
                        "ถอดคำพูดในเสียงนี้เป็นข้อความภาษาไทย"
                        "ตามที่ได้ยิน คงชื่อสินค้าและชื่อแบรนด์ "
                        "ไม่แสดงความคิดเห็น ไม่เติมคำพูด "
                        "หากไม่มีคำพูดให้ตอบว่า ไม่มีคำพูด"
                    )
                ),
                types.Part.from_bytes(
                    data=wav,
                    mime_type="audio/wav"
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            )
        )

        raw = (getattr(response, "text", None) or "").strip()

        if not raw:
            raise ValueError("Gemini ถอดเสียงไม่ได้")

        return SimpleNamespace(text=raw)

