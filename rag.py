import hashlib
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import importlib
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_core

# Avoid stale .pyc / long-running Streamlit process missing new exports
importlib.reload(rag_core)
from rag_core import (  # noqa: E402
    build_system_prompt,
    clear_persisted_index,
    get_embeddings,
    load_vectorstore,
    retrieve_doc_context,
    retrieve_web_context,
    save_vectorstore,
)
from voice_gemini import synthesize_speech_wav, transcribe_audio_bytes

# Load environment variables
load_dotenv()

LIVE_HOST = os.getenv("LIVE_HOST", "127.0.0.1")
LIVE_PORT = int(os.getenv("LIVE_PORT", "8765"))
LIVE_URL = f"http://{LIVE_HOST}:{LIVE_PORT}/"
ROOT_DIR = Path(__file__).resolve().parent


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def ensure_live_server() -> tuple[bool, str]:
    """Start realtime_server.py if needed. Returns (ok, message)."""
    if _port_open(LIVE_HOST, LIVE_PORT):
        return True, "Live Talk server is already running."

    server_script = ROOT_DIR / "realtime_server.py"
    if not server_script.exists():
        return False, "realtime_server.py not found."

    try:
        log_path = ROOT_DIR / "data" / "live_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
        subprocess.Popen(
            [sys.executable, str(server_script)],
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except Exception as exc:
        return False, f"Could not start Live Talk server: {exc}"

    for _ in range(25):
        time.sleep(0.2)
        if _port_open(LIVE_HOST, LIVE_PORT):
            return True, "Live Talk server started."
    return False, "Server process launched but port did not open. Check data/live_server.log."

# Page Setup
st.set_page_config(
    page_title="Groundly",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SESSION STATE SETUP -----------------
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None

if "processed_voice_hash" not in st.session_state:
    st.session_state["processed_voice_hash"] = None

if "voice_mode" not in st.session_state:
    st.session_state["voice_mode"] = True

if "enable_pdf" not in st.session_state:
    st.session_state["enable_pdf"] = True
if "enable_web" not in st.session_state:
    st.session_state["enable_web"] = True
if "enable_gemini" not in st.session_state:
    st.session_state["enable_gemini"] = True

if "pending_from_voice" not in st.session_state:
    st.session_state["pending_from_voice"] = False

if "chats" not in st.session_state:
    initial_id = str(uuid.uuid4())[:8]
    st.session_state["chats"] = {
        initial_id: {
            "id": initial_id,
            "title": "Adaptive Chat",
            "messages": []
        }
    }
    st.session_state["current_chat_id"] = initial_id

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "current_file" not in st.session_state:
    st.session_state["current_file"] = None

if "doc_stats" not in st.session_state:
    st.session_state["doc_stats"] = {}

# Reload last indexed PDF from disk (shared with Voice Agent)
if st.session_state["vectorstore"] is None:
    _boot_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    )
    if _boot_key:
        _store, _meta = load_vectorstore(_boot_key)
        if _store is not None:
            st.session_state["vectorstore"] = _store
            st.session_state["current_file"] = _meta.get("name")
            st.session_state["doc_stats"] = _meta

# Ensure current_chat_id is valid
if st.session_state["current_chat_id"] not in st.session_state["chats"]:
    st.session_state["current_chat_id"] = list(st.session_state["chats"].keys())[0]

current_chat = st.session_state["chats"][st.session_state["current_chat_id"]]

# ----------------- UI (native Streamlit only) -----------------
_logo = ROOT_DIR / "assets" / "voice_logo.svg"
if _logo.exists():
    st.logo(str(_logo), size="large")

doc_ready = st.session_state["vectorstore"] is not None
doc_label = st.session_state.get("current_file") or "No document"
st.caption(f"Document: {doc_label}")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("Groundly")
    st.caption("Hybrid RAG · Voice")
    st.divider()

    st.subheader("Navigation")
    if st.button("New chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state["chats"][new_id] = {
            "id": new_id,
            "title": f"Chat {len(st.session_state['chats']) + 1}",
            "messages": []
        }
        st.session_state["current_chat_id"] = new_id
        st.rerun()

    st.subheader("Projects")
    chat_keys = list(st.session_state["chats"].keys())
    for cid in chat_keys:
        item = st.session_state["chats"][cid]
        active = (cid == st.session_state["current_chat_id"])
        c_btn, c_del = st.columns([5, 1])
        with c_btn:
            if st.button(
                item["title"],
                key=f"chat_{cid}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state["current_chat_id"] = cid
                st.rerun()
        with c_del:
            if len(st.session_state["chats"]) > 1:
                if st.button("✕", key=f"del_{cid}"):
                    del st.session_state["chats"][cid]
                    if st.session_state["current_chat_id"] == cid:
                        st.session_state["current_chat_id"] = list(st.session_state["chats"].keys())[0]
                    st.rerun()

    st.subheader("Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_in")
    if st.session_state["doc_stats"]:
        stats = st.session_state["doc_stats"]
        st.caption(f"{stats.get('name', '')}")
        st.caption(f"{stats.get('pages', 0)} pages · {stats.get('chunks', 0)} chunks")
        if st.button("Detach", use_container_width=True):
            st.session_state["vectorstore"] = None
            st.session_state["current_file"] = None
            st.session_state["doc_stats"] = {}
            clear_persisted_index()
            st.rerun()

    st.subheader("Settings")
    secret_key = ""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass
    default_key = secret_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    gemini_api_key = st.text_input("Gemini API key", type="password", value=default_key)
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key

    model_name = st.selectbox(
        "Model",
        ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
        index=0,
    )

    st.subheader("Answer sources")
    st.caption("Turn sources on/off. At least one must stay on.")
    st.session_state["enable_pdf"] = st.toggle(
        "PDF document",
        value=st.session_state["enable_pdf"],
        help="Retrieve answers from the uploaded PDF (FAISS).",
    )
    st.session_state["enable_gemini"] = st.toggle(
        "Gemini knowledge",
        value=st.session_state["enable_gemini"],
        help="Allow Gemini's own trained knowledge in the answer.",
    )
    st.session_state["enable_web"] = st.toggle(
        "Internet search",
        value=st.session_state["enable_web"],
        help="Live web search (DuckDuckGo) grounding.",
    )
    enable_pdf = st.session_state["enable_pdf"]
    enable_gemini = st.session_state["enable_gemini"]
    enable_web_search = st.session_state["enable_web"]
    if not (enable_pdf or enable_gemini or enable_web_search):
        st.warning("Enable at least one answer source.")

    st.session_state["voice_mode"] = st.toggle(
        "Voice replies",
        value=st.session_state["voice_mode"],
    )
    with st.expander("Advanced"):
        top_k = st.slider("Top chunks", 1, 8, 4)
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
        chunk_overlap = st.slider("Overlap", 0, 400, 150, step=50)

    export_content = (
        f"# Groundly Report\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Document:** {st.session_state.get('current_file', 'None')}\n"
        f"**Model:** {model_name}\n\n---\n\n"
    )
    for idx, m in enumerate(current_chat["messages"]):
        speaker = "USER" if m["role"] == "user" else "ASSISTANT"
        export_content += f"## {idx+1}. {speaker}\n\n{m['content']}\n\n---\n\n"

    st.download_button(
        label="Export chat",
        data=export_content,
        file_name=f"RAG_Report_{current_chat['title'].replace(' ', '_')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()
    live_up = _port_open(LIVE_HOST, LIVE_PORT)
    st.caption("Live Talk server: " + ("online" if live_up else "offline"))
    if st.button("Start Live server", use_container_width=True, key="start_live_server"):
        ok, msg = ensure_live_server()
        (st.success if ok else st.error)(msg)
        if ok:
            st.rerun()
    st.link_button("Open Live Talk", LIVE_URL, use_container_width=True, disabled=not live_up)

# ----------------- PDF VECTORSTORE INDEXING -----------------
if uploaded_file is not None:
    if st.session_state["current_file"] != uploaded_file.name or st.session_state["vectorstore"] is None:
        if not gemini_api_key:
            st.error("⚠️ Please enter a valid Gemini API Key in the sidebar.")
        else:
            with st.spinner("Indexing PDF: Extracting text, chunking & generating Gemini embeddings..."):
                tmp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_file_path = tmp.name

                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    # Drop blank / whitespace-only pages (common in scanned PDFs)
                    docs = [d for d in docs if (d.page_content or "").strip()]

                    if not docs:
                        st.error(
                            "No extractable text found in this PDF. "
                            "It may be scanned/image-only — use a text PDF or OCR first."
                        )
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        chunks = splitter.split_documents(docs)
                        chunks = [c for c in chunks if (c.page_content or "").strip()]

                        if not chunks:
                            st.error(
                                "PDF loaded but produced 0 text chunks. "
                                "Try a different file or lower chunk size in Advanced settings."
                            )
                        else:
                            embeddings = get_embeddings(gemini_api_key)
                            # Free-tier embed limit is low (~100 RPM). Retry on 429.
                            vectorstore = None
                            last_embed_err: Exception | None = None
                            for attempt in range(1, 4):
                                try:
                                    vectorstore = FAISS.from_documents(chunks, embeddings)
                                    break
                                except Exception as embed_err:
                                    last_embed_err = embed_err
                                    err_text = str(embed_err)
                                    if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
                                        wait_s = 35 * attempt
                                        st.warning(
                                            f"Gemini embedding quota hit (free tier). "
                                            f"Waiting {wait_s}s then retry {attempt}/3…"
                                        )
                                        time.sleep(wait_s)
                                        continue
                                    raise
                            if vectorstore is None:
                                raise last_embed_err or RuntimeError("Embedding failed.")

                            st.session_state["vectorstore"] = vectorstore
                            st.session_state["current_file"] = uploaded_file.name

                            total_chars = sum(len(d.page_content) for d in docs)
                            st.session_state["doc_stats"] = {
                                "name": uploaded_file.name,
                                "pages": len(docs),
                                "chunks": len(chunks),
                                "chars": total_chars
                            }
                            save_vectorstore(
                                vectorstore,
                                file_name=uploaded_file.name,
                                pages=len(docs),
                                chunks=len(chunks),
                                chars=total_chars,
                            )
                            st.toast(
                                f"✅ Indexed '{uploaded_file.name}' into FAISS ({len(chunks)} chunks)",
                                icon="🚀",
                            )
                            st.rerun()

                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        st.error(
                            "Gemini Free Tier embedding quota exceeded "
                            "(~100 embed requests/minute). Wait ~1 minute and upload again, "
                            "or use a smaller PDF / paid plan. "
                            "Docs: https://ai.google.dev/gemini-api/docs/rate-limits"
                        )
                    else:
                        st.error(f"Error processing document: {err}")
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

# ----------------- EMPTY STATE (hides after first prompt) -----------------
show_hero = len(current_chat["messages"]) == 0

if show_hero:
    st.title("Design smarter systems with adaptive intelligence")
    _net_html = (ROOT_DIR / "brain_network.html").read_text(encoding="utf-8")
    components.html(_net_html, height=340, scrolling=False)

# ----------------- DISPLAY CHAT HISTORY -----------------
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("audio_wav"):
            st.audio(msg["audio_wav"], format="audio/wav")

        if msg.get("metrics"):
            m = msg["metrics"]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Latency", f"{m.get('latency', 0):.2f}s")
            mc2.metric("Tokens", f"~{m.get('tokens', 0)}")
            mc3.metric("Model", m.get("model", "Gemini"))

        # Display Source References Expander
        sources = msg.get("sources", {})
        doc_srcs = sources.get("docs", [])
        web_srcs = sources.get("web", [])

        if doc_srcs or web_srcs:
            with st.expander("Sources", expanded=False):
                if doc_srcs:
                    st.markdown("**Document**")
                    for idx, d in enumerate(doc_srcs):
                        page_num = d.get("page", 0) + 1
                        st.caption(f"p.{page_num} — {d.get('content', '')[:220]}")

                if web_srcs:
                    st.markdown("**Web**")
                    for w in web_srcs:
                        st.markdown(f"- [{w.get('title', 'Source')}]({w.get('href', '#')})")

# ----------------- LIVE TALK DIALOG (compact embedded window) -----------------
@st.dialog("Live Talk")
def live_talk_dialog() -> None:
    if not _port_open(LIVE_HOST, LIVE_PORT):
        with st.spinner("Starting Live server…"):
            ok, msg = ensure_live_server()
        if not ok:
            st.error(msg)
            if st.button("Retry", use_container_width=True, key="dialog_retry_live"):
                st.rerun()
            return

    html = (ROOT_DIR / "live_talk.html").read_text(encoding="utf-8")
    ws_url = f"ws://{LIVE_HOST}:{LIVE_PORT}/ws/live"
    html = html.replace("__WS_URL__", ws_url).replace("__AUTO_START__", "1")
    components.html(html, height=300, scrolling=False)


def _open_live_talk() -> None:
    live_talk_dialog()


# Live Talk icon placement
# Main issue: Streamlit chat_input does not allow custom children. Mic + send sit
# ~8px apart inside the widget, so a guessed `position:fixed; right:…` never lands
# between them. Fix: open a CSS gap before the send button, then pin our real
# st.button to the midpoint of [mic → send] via getBoundingClientRect.
st.markdown(
    """
    <style>
      div[data-testid="stDialog"] > div:first-child {
        max-width: min(440px, 94vw) !important;
        width: min(440px, 94vw) !important;
      }
      /* Create a slot between mic and arrow for the Live icon */
      [data-testid="stChatInputSubmitButton"] {
        margin-left: 2.35rem !important;
      }
      div[data-testid="stElementContainer"]:has([class*="st-key-chat_live_icon"]),
      div[data-testid="element-container"]:has([class*="st-key-chat_live_icon"]) {
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        overflow: visible !important;
      }
      [class*="st-key-chat_live_icon"] {
        position: fixed !important;
        z-index: 1000002 !important;
        width: 2rem !important;
        height: 2rem !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: auto !important;
      }
      [class*="st-key-chat_live_icon"] button {
        width: 2rem !important;
        min-width: 2rem !important;
        height: 2rem !important;
        border-radius: 999px !important;
        padding: 0 !important;
        margin: 0 !important;
        font-size: 0.95rem !important;
        line-height: 1 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: inherit !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- INPUT PROCESSING & STREAMING -----------------
prompt_from_pill = st.session_state["pending_prompt"]

if st.button("🔴", key="chat_live_icon", help="Live Talk"):
    _open_live_talk()

chat_value = st.chat_input(
    "Ask anything you want...",
    accept_audio=True,
    audio_sample_rate=16000,
    key="rag_chat_input",
)

components.html(
    """
    <script>
    (function () {
      let doc;
      try {
        doc = window.parent.document;
      } catch (err) {
        document.body.innerText = "parent blocked";
        return;
      }

      function pinLiveBetweenMicAndSend() {
        const mic = doc.querySelector('[data-testid="stChatInputMicButton"]');
        const send = doc.querySelector('[data-testid="stChatInputSubmitButton"]');
        const liveWrap = doc.querySelector('.st-key-chat_live_icon')
          || doc.querySelector('[class*="st-key-chat_live_icon"]');
        if (!mic || !send || !liveWrap) return;

        const micR = mic.getBoundingClientRect();
        const sendR = send.getBoundingClientRect();
        if (micR.width < 1 || sendR.width < 1) return;

        const size = 32;
        const x = (micR.right + sendR.left) / 2 - size / 2;
        const y = (Math.min(micR.top, sendR.top) + Math.max(micR.bottom, sendR.bottom)) / 2 - size / 2;

        liveWrap.style.setProperty("position", "fixed", "important");
        liveWrap.style.setProperty("left", x + "px", "important");
        liveWrap.style.setProperty("top", y + "px", "important");
        liveWrap.style.setProperty("width", size + "px", "important");
        liveWrap.style.setProperty("height", size + "px", "important");
        liveWrap.style.setProperty("min-height", size + "px", "important");
        liveWrap.style.setProperty("z-index", "1000002", "important");
        liveWrap.style.setProperty("overflow", "visible", "important");
      }

      pinLiveBetweenMicAndSend();
      window.parent.addEventListener("resize", pinLiveBetweenMicAndSend);
      window.parent.addEventListener("scroll", pinLiveBetweenMicAndSend, true);
      setInterval(pinLiveBetweenMicAndSend, 250);
      new MutationObserver(pinLiveBetweenMicAndSend)
        .observe(doc.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    height=1,
    width=1,
)

voice_prompt = None
from_voice = bool(st.session_state.get("pending_from_voice"))
user_prompt = None

if chat_value is not None:
    # With accept_audio=True Streamlit returns ChatInputValue; plain str is still possible.
    audio_file = getattr(chat_value, "audio", None) if not isinstance(chat_value, str) else None
    typed_text = chat_value if isinstance(chat_value, str) else (chat_value.text or "").strip()

    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash != st.session_state["processed_voice_hash"]:
            if not gemini_api_key:
                st.warning("⚠️ Please provide a Gemini API Key in the sidebar.")
            else:
                with st.spinner("🎧 Transcribing with Gemini..."):
                    try:
                        mime = getattr(audio_file, "type", None) or "audio/wav"
                        voice_prompt = transcribe_audio_bytes(
                            gemini_api_key, audio_bytes, mime_type=mime
                        )
                        st.session_state["processed_voice_hash"] = audio_hash
                        from_voice = True
                        if not voice_prompt:
                            st.warning("Could not understand the audio. Please try again.")
                    except Exception as e:
                        st.error(f"Voice transcription failed: {e}")
    elif typed_text:
        user_prompt = typed_text

# Resolve prompt (voice takes priority over typed input on same run)
active_prompt = voice_prompt or prompt_from_pill or user_prompt

if active_prompt:
    # Clear pending prompt / voice flags
    st.session_state["pending_prompt"] = None
    if prompt_from_pill and st.session_state.get("pending_from_voice"):
        from_voice = True
    st.session_state["pending_from_voice"] = False

    if not gemini_api_key:
        st.warning("⚠️ Please provide a Gemini API Key in the sidebar.")
    elif not (enable_pdf or enable_gemini or enable_web_search):
        st.warning("Enable at least one answer source in the sidebar (PDF / Gemini / Internet).")
    else:
        # Auto-name chat on first prompt
        if current_chat["title"] in ["Document Intelligence", "New Chat", "Session 1", "Welcome Chat"] or current_chat["title"].startswith("Session "):
            clean_t = active_prompt.strip()
            current_chat["title"] = (clean_t[:22] + "...") if len(clean_t) > 22 else clean_t

        user_label = f"Voice · {active_prompt}" if from_voice else active_prompt

        # User Message
        current_chat["messages"].append({"role": "user", "content": user_label})
        with st.chat_message("user"):
            st.markdown(user_label)

        # Assistant Message
        with st.chat_message("assistant"):
            start_time = time.time()
            retrieved_doc_sources = []
            retrieved_web_sources = []

            # 1. Document Retrieval (optional)
            if enable_pdf:
                with st.spinner("Retrieving PDF chunks…"):
                    doc_context_text, retrieved_doc_sources = retrieve_doc_context(
                        st.session_state["vectorstore"],
                        active_prompt,
                        top_k=top_k,
                        enabled=True,
                    )
            else:
                doc_context_text, retrieved_doc_sources = retrieve_doc_context(
                    None, active_prompt, enabled=False
                )

            # 2. Web Retrieval (optional)
            if enable_web_search:
                with st.spinner("Searching the internet…"):
                    web_context_text, retrieved_web_sources = retrieve_web_context(
                        active_prompt,
                        enabled=True,
                    )
            else:
                web_context_text, retrieved_web_sources = retrieve_web_context(
                    active_prompt,
                    enabled=False,
                )

            # 3. System Prompt & Reasoning Chain (respects source toggles)
            prompt_template = ChatPromptTemplate.from_template(
                build_system_prompt(
                    enable_pdf=enable_pdf,
                    enable_web=enable_web_search,
                    enable_gemini=enable_gemini,
                    voice_mode=from_voice,
                )
            )

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,
                google_api_key=gemini_api_key
            )

            chain = prompt_template | llm | StrOutputParser()

            # 4. Real-Time Token Streaming
            try:
                def response_generator():
                    for chunk in chain.stream({
                        "doc_context": doc_context_text,
                        "web_context": web_context_text,
                        "question": active_prompt
                    }):
                        yield chunk

                response_text = st.write_stream(response_generator())
                latency = time.time() - start_time

                # Token estimation
                approx_tokens = (len(active_prompt) + len(response_text) + len(doc_context_text)) // 4

                audio_wav = None
                if from_voice and st.session_state.get("voice_mode", True):
                    with st.spinner("🔊 Generating Gemini voice reply..."):
                        try:
                            audio_wav = synthesize_speech_wav(gemini_api_key, response_text)
                            st.audio(audio_wav, format="audio/wav", autoplay=True)
                        except Exception as tts_err:
                            st.warning(f"Answer ready, but TTS failed: {tts_err}")

                src_bits = []
                if enable_pdf:
                    src_bits.append("PDF")
                if enable_gemini:
                    src_bits.append("Gemini")
                if enable_web_search:
                    src_bits.append("Web")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Latency", f"{latency:.2f}s")
                mc2.metric("Tokens", f"~{approx_tokens}")
                mc3.metric("Model", model_name)
                mc4.metric("Sources", "+".join(src_bits) or "None")

                # Source Citations
                if retrieved_doc_sources or retrieved_web_sources:
                    with st.expander("Sources", expanded=False):
                        if retrieved_doc_sources:
                            st.markdown("**Document**")
                            for idx, d in enumerate(retrieved_doc_sources):
                                page_num = d.get("page", 0) + 1
                                st.caption(f"p.{page_num} — {d.get('content', '')[:220]}")

                        if retrieved_web_sources:
                            st.markdown("**Web**")
                            for w in retrieved_web_sources:
                                st.markdown(f"- [{w.get('title', 'Source')}]({w.get('href', '#')})")

                # Append to current chat history
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": response_text,
                    "audio_wav": audio_wav,
                    "sources": {
                        "docs": retrieved_doc_sources,
                        "web": retrieved_web_sources
                    },
                    "metrics": {
                        "latency": latency,
                        "tokens": approx_tokens,
                        "model": model_name
                    }
                })

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
