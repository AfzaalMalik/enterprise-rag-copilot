"""
Gemini-only Realtime voice agent (CLI) with Hybrid RAG grounding.

Continuous conversation with barge-in support via Gemini Live API.
Prefer the Streamlit Live Talk UI / realtime_server.py for browser use.

Run:
  python 16-realtime_agents.py

Tips: use headphones to avoid echo / self-interruption.
"""

from __future__ import annotations

import asyncio
import queue
import sys

import numpy as np
import sounddevice as sd
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types

load_dotenv(find_dotenv(), override=True)

from gemini_live_rag import (
    LIVE_MODEL,
    build_live_config,
    extract_tool_args,
    get_api_key,
    load_rag_state,
    run_knowledge_query,
)

SEND_RATE = 16_000
RECV_RATE = 24_000


async def main() -> None:
    api_key = get_api_key()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY / GOOGLE_API_KEY in .env")

    vectorstore, meta = load_rag_state(api_key)
    if vectorstore is None:
        print("No FAISS index found. Upload a PDF in Streamlit first for grounded answers.")
        print("Continuing with general + web knowledge.\n")
    else:
        print(f"Loaded document index: {meta.get('name', 'unknown')}\n")

    client = genai.Client(api_key=api_key)
    config = build_live_config(doc_name=meta.get("name"))

    audio_out: queue.Queue[bytes] = queue.Queue()
    stop_event = asyncio.Event()

    def output_callback(outdata, frames, time, status):
        outdata.fill(0)
        filled = 0
        buf = getattr(output_callback, "_buf", np.zeros(0, dtype=np.int16))
        while filled < frames:
            if buf.size == 0:
                try:
                    chunk = audio_out.get_nowait()
                except queue.Empty:
                    break
                buf = np.frombuffer(chunk, dtype=np.int16)
            n = min(frames - filled, buf.size)
            outdata[filled : filled + n, 0] = buf[:n]
            buf = buf[n:]
            filled += n
        output_callback._buf = buf

    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
        print(f"Realtime agent listening ({LIVE_MODEL}). Speak now. Ctrl+C to quit.\n")

        loop = asyncio.get_running_loop()

        def mic_callback(indata, frames, time, status):
            pcm = indata.copy().flatten().astype(np.int16).tobytes()
            asyncio.run_coroutine_threadsafe(
                session.send_realtime_input(
                    audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={SEND_RATE}")
                ),
                loop,
            )

        async def receive_loop() -> None:
            while not stop_event.is_set():
                turn = session.receive()
                async for response in turn:
                    if response.tool_call and response.tool_call.function_calls:
                        function_responses = []
                        for fc in response.tool_call.function_calls:
                            args = extract_tool_args(fc)
                            question = str(args.get("question") or "").strip()
                            if fc.name == "query_knowledge_base" and question:
                                print(f"\n[RAG] {question}")
                                result = await asyncio.to_thread(
                                    run_knowledge_query, question, api_key, vectorstore
                                )
                            else:
                                result = f"Unknown tool: {fc.name}"
                            function_responses.append(
                                types.FunctionResponse(
                                    name=fc.name,
                                    id=fc.id,
                                    response={"result": result},
                                )
                            )
                        await session.send_tool_response(
                            function_responses=function_responses
                        )
                        continue

                    content = response.server_content
                    if not content:
                        continue

                    if content.interrupted:
                        while not audio_out.empty():
                            try:
                                audio_out.get_nowait()
                            except queue.Empty:
                                break
                        output_callback._buf = np.zeros(0, dtype=np.int16)
                        print("\n…interrupted")

                    if content.input_transcription and content.input_transcription.text:
                        print(f"You: {content.input_transcription.text}")

                    if content.output_transcription and content.output_transcription.text:
                        print(f"AI: {content.output_transcription.text}", end="", flush=True)

                    if content.model_turn:
                        for part in content.model_turn.parts or []:
                            if part.inline_data and part.inline_data.data:
                                audio_out.put(part.inline_data.data)

                    if content.turn_complete:
                        print()

        with sd.OutputStream(
            samplerate=RECV_RATE,
            channels=1,
            dtype=np.int16,
            callback=output_callback,
            blocksize=int(RECV_RATE * 0.04),
        ):
            with sd.InputStream(
                samplerate=SEND_RATE,
                channels=1,
                dtype="int16",
                callback=mic_callback,
                blocksize=int(SEND_RATE * 0.04),
            ):
                try:
                    await receive_loop()
                except KeyboardInterrupt:
                    stop_event.set()
                    print("\nStopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
