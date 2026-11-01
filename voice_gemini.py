"""Gemini-only speech helpers for Streamlit Voice Agent UI."""

from __future__ import annotations

import io
import os
import wave

import numpy as np
from google import genai
from google.genai import types

STT_MODEL = os.getenv("VOICE_STT_MODEL", "gemini-3.5-flash")
TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "Kore")
TTS_SAMPLE_RATE = 24_000


def _client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def pcm_to_wav_bytes(pcm: np.ndarray | bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    if isinstance(pcm, bytes):
        pcm_arr = np.frombuffer(pcm, dtype=np.int16)
    else:
        pcm_arr = np.asarray(pcm, dtype=np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_arr.tobytes())
    return buffer.getvalue()


def transcribe_audio_bytes(
    api_key: str,
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
) -> str:
    """Transcribe browser/mic audio with Gemini."""
    client = _client(api_key)
    # Streamlit audio_input often returns audio/wav; fall back safely.
    mime = (mime_type or "audio/wav").split(";")[0].strip() or "audio/wav"
    response = client.models.generate_content(
        model=STT_MODEL,
        contents=[
            types.Content(
                parts=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                    types.Part.from_text(
                        text=(
                            "Transcribe the spoken words exactly. "
                            "Return only the transcript text with no commentary."
                        )
                    ),
                ]
            )
        ],
    )
    return (response.text or "").strip()


def synthesize_speech_wav(api_key: str, text: str, voice: str | None = None) -> bytes:
    """Generate WAV audio from text using Gemini TTS."""
    spoken = " ".join((text or "").split())
    if len(spoken) > 1200:
        spoken = spoken[:1200].rsplit(" ", 1)[0] + "."

    client = _client(api_key)
    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=spoken,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice or TTS_VOICE
                    )
                )
            ),
        ),
    )
    pcm = response.candidates[0].content.parts[0].inline_data.data
    return pcm_to_wav_bytes(pcm, sample_rate=TTS_SAMPLE_RATE)
