"""Shared Gemini Live + Hybrid RAG helpers for CLI and WebSocket bridge."""

from __future__ import annotations

import os
from typing import Any

from google.genai import types

from rag_core import answer_hybrid, load_vectorstore, resolve_gemini_api_key

LIVE_MODEL = os.getenv("VOICE_LIVE_MODEL", "gemini-3.8-live")
LIVE_VOICE = os.getenv("VOICE_LIVE_VOICE", "Kore")
RAG_MODEL = os.getenv("VOICE_RAG_MODEL", "gemini-3.5-flash")
ENABLE_WEB = os.getenv("VOICE_ENABLE_WEB", "true").lower() in {"1", "true", "yes"}

SYSTEM_INSTRUCTION = (
    "You are a friendly Hybrid RAG voice copilot for an enterprise document system.\n"
    "Speak naturally in short conversational turns (2-5 sentences).\n"
    "When the user asks about documents, resumes, skills, companies, careers, or technical topics, "
    "ALWAYS call query_knowledge_base first, then answer from that grounded result.\n"
    "Do not invent document facts. If the tool says something is not documented, say so clearly.\n"
    "No markdown, bullets, or emoji — answers are spoken aloud."
)


def get_api_key() -> str:
    return resolve_gemini_api_key()


def load_rag_state(api_key: str | None = None) -> tuple[Any, dict]:
    key = api_key or get_api_key()
    if not key:
        return None, {}
    return load_vectorstore(key)


def run_knowledge_query(question: str, api_key: str, vectorstore: Any) -> str:
    result = answer_hybrid(
        question,
        api_key=api_key,
        model_name=RAG_MODEL,
        vectorstore=vectorstore,
        enable_web_search=ENABLE_WEB,
        top_k=4,
        voice_mode=True,
    )
    return result["answer"]


def build_live_config(*, doc_name: str | None = None) -> types.LiveConnectConfig:
    doc_line = (
        f"Active document index: {doc_name}."
        if doc_name
        else "No document is indexed yet; use web/general knowledge when needed."
    )
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="query_knowledge_base",
                description=(
                    "Search the uploaded PDF (FAISS) and optional live web results, "
                    "then return a grounded answer for the user's question."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "question": types.Schema(
                            type=types.Type.STRING,
                            description="The user's question to ground against documents/web.",
                        )
                    },
                    required=["question"],
                ),
            )
        ]
    )
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=f"{SYSTEM_INSTRUCTION}\n{doc_line}",
        tools=[tool],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=LIVE_VOICE)
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


def extract_tool_args(function_call: Any) -> dict[str, Any]:
    raw = getattr(function_call, "args", None) or {}
    if isinstance(raw, dict):
        return raw
    try:
        return dict(raw)
    except Exception:
        return {}
