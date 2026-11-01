"""
Gemini Live WebSocket bridge for the Streamlit Live Talk UI.

Run:
  python realtime_server.py

Then open http://localhost:8765/ in the browser (or use the iframe in Streamlit).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
import uvicorn

load_dotenv(find_dotenv(), override=True)

from gemini_live_rag import (
    LIVE_MODEL,
    build_live_config,
    extract_tool_args,
    get_api_key,
    load_rag_state,
    run_knowledge_query,
)

HOST = os.getenv("LIVE_HOST", "127.0.0.1")
PORT = int(os.getenv("LIVE_PORT", "8765"))
HTML_PATH = Path(__file__).resolve().parent / "live_talk.html"

app = FastAPI(title="RAG Gemini Live Bridge")


@app.get("/")
async def index() -> HTMLResponse:
    html = HTML_PATH.read_text(encoding="utf-8")
    ws_url = f"ws://{HOST}:{PORT}/ws/live"
    html = (
        html.replace("__WS_URL__", ws_url)
        .replace("__AUTO_START__", "0")
    )
    return HTMLResponse(html)


@app.get("/health")
async def health() -> dict:
    api_key = get_api_key()
    _, meta = load_rag_state(api_key) if api_key else (None, {})
    return {
        "ok": True,
        "model": LIVE_MODEL,
        "has_api_key": bool(api_key),
        "document": meta.get("name"),
    }


async def _handle_tool_calls(session, tool_call, api_key: str, vectorstore) -> None:
    if not tool_call or not tool_call.function_calls:
        return
    responses = []
    for fc in tool_call.function_calls:
        args = extract_tool_args(fc)
        question = str(args.get("question") or "").strip()
        if fc.name == "query_knowledge_base" and question:
            try:
                result = await asyncio.to_thread(
                    run_knowledge_query, question, api_key, vectorstore
                )
            except Exception as exc:
                result = f"Knowledge lookup failed: {exc}"
        else:
            result = f"Unknown tool: {fc.name}"
        responses.append(
            types.FunctionResponse(
                name=fc.name,
                id=fc.id,
                response={"result": result},
            )
        )
    await session.send_tool_response(function_responses=responses)


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    api_key = get_api_key()
    if not api_key:
        await websocket.send_json(
            {"type": "error", "message": "Missing GEMINI_API_KEY / GOOGLE_API_KEY in .env"}
        )
        await websocket.close()
        return

    vectorstore, meta = load_rag_state(api_key)
    doc_name = meta.get("name")
    client = genai.Client(api_key=api_key)
    config = build_live_config(doc_name=doc_name)

    await websocket.send_json(
        {
            "type": "status",
            "message": (
                f"Connected. Document: {doc_name}" if doc_name else "Connected. No document indexed."
            ),
        }
    )

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            await websocket.send_json(
                {"type": "status", "message": f"Live session ready ({LIVE_MODEL}). Speak now."}
            )

            async def pump_browser_to_gemini() -> None:
                while True:
                    raw = await websocket.receive_text()
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "audio":
                        pcm = base64.b64decode(msg["data"])
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=pcm,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                    elif mtype == "text":
                        text = (msg.get("text") or "").strip()
                        if text:
                            await session.send_realtime_input(text=text)
                    elif mtype == "end":
                        break

            async def pump_gemini_to_browser() -> None:
                while True:
                    turn = session.receive()
                    async for response in turn:
                        if response.tool_call:
                            await _handle_tool_calls(
                                session, response.tool_call, api_key, vectorstore
                            )
                            continue

                        content = response.server_content
                        if not content:
                            continue

                        if content.interrupted:
                            await websocket.send_json({"type": "interrupted"})

                        if content.input_transcription and content.input_transcription.text:
                            await websocket.send_json(
                                {
                                    "type": "input_transcript",
                                    "text": content.input_transcription.text,
                                }
                            )

                        if content.output_transcription and content.output_transcription.text:
                            await websocket.send_json(
                                {
                                    "type": "output_transcript",
                                    "text": content.output_transcription.text,
                                }
                            )

                        if content.model_turn:
                            for part in content.model_turn.parts or []:
                                if part.inline_data and part.inline_data.data:
                                    await websocket.send_json(
                                        {
                                            "type": "audio",
                                            "data": base64.b64encode(
                                                part.inline_data.data
                                            ).decode("ascii"),
                                        }
                                    )

            browser_task = asyncio.create_task(pump_browser_to_gemini())
            gemini_task = asyncio.create_task(pump_gemini_to_browser())
            done, pending = await asyncio.wait(
                {browser_task, gemini_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                    raise exc
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    print(f"RAG Live Talk server → http://{HOST}:{PORT}/")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
