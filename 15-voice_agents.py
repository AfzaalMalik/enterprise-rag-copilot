"""
Gemini-only Voice Q&A over the same Hybrid RAG index used by Streamlit.

No OpenAI key required.

Workflow:
1. Upload + index a PDF in Streamlit (rag.py) — saves FAISS to data/faiss_index
2. Ensure GEMINI_API_KEY is in .env
3. Run: python 15-voice_agents.py
"""

from __future__ import annotations

import io
import os
import wave

import numpy as np
import sounddevice as sd
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types

load_dotenv(find_dotenv(), override=True)

from rag_core import (
    answer_hybrid,
    load_vectorstore,
    resolve_gemini_api_key,
)

SAMPLE_RATE = 24_000
GEMINI_API_KEY = resolve_gemini_api_key()
VECTORSTORE, INDEX_META = load_vectorstore(GEMINI_API_KEY) if GEMINI_API_KEY else (None, {})
ENABLE_WEB = os.getenv("VOICE_ENABLE_WEB", "true").lower() in {"1", "true", "yes"}
RAG_MODEL = os.getenv("VOICE_RAG_MODEL", "gemini-3.5-flash")
STT_MODEL = os.getenv("VOICE_STT_MODEL", "gemini-3.5-flash")
TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "Kore")


def record_from_mic() -> np.ndarray:
    print("Recording... press Enter to stop.")
    chunks: list[np.ndarray] = []

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input()

    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks, axis=0).flatten()


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buffer.getvalue()


def transcribe_audio(client: genai.Client, pcm: np.ndarray) -> str:
    wav_bytes = pcm_to_wav_bytes(pcm)
    response = client.models.generate_content(
        model=STT_MODEL,
        contents=[
            types.Content(
                parts=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
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


def synthesize_speech(client: genai.Client, text: str) -> np.ndarray:
    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=TTS_VOICE
                    )
                )
            ),
        ),
    )
    part = response.candidates[0].content.parts[0]
    audio_bytes = part.inline_data.data
    return np.frombuffer(audio_bytes, dtype=np.int16)


def play_pcm(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    sd.play(pcm, samplerate=sample_rate)
    sd.wait()


def main() -> None:
    if not GEMINI_API_KEY:
        raise SystemExit(
            "Missing GEMINI_API_KEY / GOOGLE_API_KEY. Add it to your .env file."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    if VECTORSTORE is None:
        print("No saved FAISS index found.")
        print("Open Streamlit, upload a PDF, wait for indexing, then re-run this script.")
        print("Continuing in general-knowledge + web mode...\n")
    else:
        print(f"Loaded document index: {INDEX_META.get('name', 'unknown')}")
        print(f"Chunks: {INDEX_META.get('chunks', '?')} | Web search: {ENABLE_WEB}\n")

    input("Press Enter to start recording your question...")
    pcm = record_from_mic()
    if pcm.size == 0:
        raise SystemExit("No audio captured. Check your microphone and try again.")

    print("Transcribing with Gemini...")
    question = transcribe_audio(client, pcm)
    if not question:
        raise SystemExit("Could not transcribe audio. Please try speaking again.")

    print(f"You said: {question}")
    print("Thinking with Hybrid RAG (Gemini)...")
    result = answer_hybrid(
        question,
        api_key=GEMINI_API_KEY,
        model_name=RAG_MODEL,
        vectorstore=VECTORSTORE,
        enable_web_search=ENABLE_WEB,
        top_k=4,
        voice_mode=True,
    )
    answer = result["answer"].strip()
    print(f"\nAnswer:\n{answer}\n")

    print("Speaking with Gemini TTS...")
    speech = synthesize_speech(client, answer)
    play_pcm(speech)
    print("Done.")


if __name__ == "__main__":
    main()
