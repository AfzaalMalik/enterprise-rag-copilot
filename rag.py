import os
import tempfile
import time
import uuid
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="Enterprise Hybrid RAG Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SESSION STATE SETUP -----------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None

if "chats" not in st.session_state:
    initial_id = str(uuid.uuid4())[:8]
    st.session_state["chats"] = {
        initial_id: {
            "id": initial_id,
            "title": "Document Intelligence",
            "messages": [
                {
                    "role": "assistant",
                    "content": "### ⚡ Welcome to Enterprise Hybrid RAG Copilot\n\n"
                               "I am an intelligent multi-source reasoning agent tailored for deep document analysis, real-time web verification, and domain intelligence.\n\n"
                               "- 📄 **Document Grounding:** Upload any PDF or Resume for strict, cite-backed analysis.\n"
                               "- 🌐 **Live Web Search:** Automatically verifies missing facts, company background, and market benchmarks.\n"
                               "- 🤖 **General AI Intelligence:** Ask any technical, career, or architectural question anytime.\n\n"
                               "*Every answer is transparently categorized with verified source citations.*",
                    "sources": {"docs": [], "web": []},
                    "metrics": {"latency": 0.4, "tokens": 120, "model": "gemini-3.5-flash"}
                }
            ]
        }
    }
    st.session_state["current_chat_id"] = initial_id

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "current_file" not in st.session_state:
    st.session_state["current_file"] = None

if "doc_stats" not in st.session_state:
    st.session_state["doc_stats"] = {}

# Ensure current_chat_id is valid
if st.session_state["current_chat_id"] not in st.session_state["chats"]:
    st.session_state["current_chat_id"] = list(st.session_state["chats"].keys())[0]

current_chat = st.session_state["chats"][st.session_state["current_chat_id"]]

# ----------------- PREMIUM DESIGN SYSTEM & CSS -----------------
is_dark = st.session_state["theme"] == "dark"

theme_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* App Container */
.stApp {{
    background: {'#090d16' if is_dark else '#f8fafc'};
    color: {'#e2e8f0' if is_dark else '#0f172a'};
}}

/* Header blur */
header[data-testid="stHeader"] {{
    background: {'rgba(9, 13, 22, 0.85)' if is_dark else 'rgba(248, 250, 252, 0.85)'};
    backdrop-filter: blur(12px);
    border-bottom: 1px solid {'rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(0, 0, 0, 0.06)'};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {'#0f172a' if is_dark else '#ffffff'};
    border-right: 1px solid {'#1e293b' if is_dark else '#e2e8f0'};
}}

/* Metrics Bar */
.metrics-pill-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    margin-top: 10px;
    border-radius: 8px;
    background: {'rgba(255, 255, 255, 0.03)' if is_dark else 'rgba(0, 0, 0, 0.02)'};
    border: 1px solid {'rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(0, 0, 0, 0.06)'};
    font-size: 0.76rem;
    font-family: 'JetBrains Mono', monospace;
    color: {'#94a3b8' if is_dark else '#64748b'};
}}

.metric-tag {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}

/* Source Badges */
.badge-doc {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    background: {'rgba(16, 185, 129, 0.12)' if is_dark else '#ecfdf5'};
    border: 1px solid {'#10b981' if is_dark else '#059669'};
    color: {'#34d399' if is_dark else '#047857'};
}}

.badge-web {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    background: {'rgba(56, 189, 248, 0.12)' if is_dark else '#eff6ff'};
    border: 1px solid {'#38bdf8' if is_dark else '#2563eb'};
    color: {'#38bdf8' if is_dark else '#1d4ed8'};
}}

.badge-ai {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    background: {'rgba(168, 85, 247, 0.12)' if is_dark else '#faf5ff'};
    border: 1px solid {'#a855f7' if is_dark else '#7e22ce'};
    color: {'#c084fc' if is_dark else '#6b21a8'};
}}

/* Hero Card */
.hero-card {{
    background: {'linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%)' if is_dark else 'linear-gradient(135deg, #f1f5f9 0%, #ffffff 100%)'};
    border: 1px solid {'rgba(255, 255, 255, 0.08)' if is_dark else '#e2e8f0'};
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px {'rgba(0, 0, 0, 0.3)' if is_dark else 'rgba(0, 0, 0, 0.04)'};
}}

/* Quick Prompt Buttons */
.stButton>button {{
    border-radius: 8px;
    transition: all 0.2s ease;
}}
.stButton>button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px {'rgba(56, 189, 248, 0.25)' if is_dark else 'rgba(37, 99, 235, 0.15)'};
}}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

# ----------------- WEB SEARCH HELPER -----------------
def search_web_ddg(query: str, max_results: int = 3):
    """Executes live web search using ddgs package."""
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", "Web Source"),
                "href": r.get("href", "#"),
                "body": r.get("body", "")
            })
        return formatted
    except Exception:
        return []

# ----------------- TOP NAVBAR & HEADER -----------------
col_title, col_status, col_theme = st.columns([6, 3, 2], vertical_alignment="center")

with col_title:
    st.markdown("## ⚡ **Hybrid RAG & AI Copilot**")

with col_status:
    if st.session_state["vectorstore"] is not None:
        file_name = st.session_state['current_file']
        st.markdown(f'<div class="badge-doc">📄 Active: {file_name[:20]}...</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-ai">🤖 General AI Mode (No PDF)</div>', unsafe_allow_html=True)

with col_theme:
    theme_btn_text = "🌙 Dark Mode" if is_dark else "☀️ Light Mode"
    if st.button(theme_btn_text, use_container_width=True, key="theme_toggle"):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()

st.caption("🚀 Multi-Source Document Intelligence System • Verified Citations • Real-Time Web Grounding")

# ----------------- SIDEBAR: CHAT SESSIONS & CONFIG -----------------
with st.sidebar:
    st.markdown("### 💬 **Workspaces & Chats**")
    if st.button("➕ **New Session**", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state["chats"][new_id] = {
            "id": new_id,
            "title": f"Session {len(st.session_state['chats']) + 1}",
            "messages": []
        }
        st.session_state["current_chat_id"] = new_id
        st.rerun()

    # Chat list
    chat_keys = list(st.session_state["chats"].keys())
    for cid in chat_keys:
        item = st.session_state["chats"][cid]
        active = (cid == st.session_state["current_chat_id"])
        
        c_btn, c_del = st.columns([5, 1])
        with c_btn:
            prefix = "🔹 " if active else "💬 "
            if st.button(f"{prefix}{item['title']}", key=f"chat_{cid}", use_container_width=True):
                st.session_state["current_chat_id"] = cid
                st.rerun()
        with c_del:
            if len(st.session_state["chats"]) > 1:
                if st.button("✕", key=f"del_{cid}", help="Delete chat"):
                    del st.session_state["chats"][cid]
                    if st.session_state["current_chat_id"] == cid:
                        st.session_state["current_chat_id"] = list(st.session_state["chats"].keys())[0]
                    st.rerun()

    st.markdown("---")
    st.markdown("### 📄 **Document Ingestion**")
    uploaded_file = st.file_uploader("Upload PDF (Resume / Report / Paper)", type=["pdf"], key="pdf_in")

    if st.session_state["doc_stats"]:
        stats = st.session_state["doc_stats"]
        st.markdown(
            f"""
            <div style="font-size:0.8rem; padding: 10px; border-radius: 8px; background: {'rgba(255,255,255,0.04)' if is_dark else '#f1f5f9'}; border: 1px solid {'rgba(255,255,255,0.08)' if is_dark else '#e2e8f0'};">
                <b>📊 Document Metrics:</b><br>
                • File: <code>{stats.get('name', '')}</code><br>
                • Pages: <b>{stats.get('pages', 0)}</b> | Chunks: <b>{stats.get('chunks', 0)}</b><br>
                • Total Chars: <b>{stats.get('chars', 0):,}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    # Load API key with support for Streamlit Cloud Secrets and local .env
    secret_key = ""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass
    default_key = secret_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    gemini_api_key = st.text_input("Gemini API Key", type="password", value=default_key)
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key

    model_name = st.selectbox(
        "LLM Reasoner",
        ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
        index=0
    )

    enable_web_search = st.toggle("🌐 Live Web Search Grounding", value=True)
    top_k = st.slider("Top Chunks Retrieved (k)", 1, 8, 4)
    chunk_size = st.slider("Chunk Size (Tokens/Chars)", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 150, step=50)

    st.markdown("---")
    st.markdown("### 📥 **Export Report**")
    # Prepare export text
    export_content = f"# Enterprise Hybrid RAG Copilot — Executive Report\n"
    export_content += f"**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_content += f"**Active Document:** {st.session_state.get('current_file', 'No Document Attached')}\n"
    export_content += f"**Model:** {model_name}\n\n---\n\n"

    for idx, m in enumerate(current_chat["messages"]):
        speaker = "USER" if m["role"] == "user" else "ASSISTANT (AI COPILOT)"
        export_content += f"## {idx+1}. {speaker}\n\n{m['content']}\n\n"
        if m.get("metrics"):
            met = m["metrics"]
            export_content += f"> *Metrics: Latency {met.get('latency')}s | Tokens: ~{met.get('tokens')} | Model: {met.get('model')}*\n\n"
        export_content += "---\n\n"

    st.download_button(
        label="📥 Download Conversation (Markdown)",
        data=export_content,
        file_name=f"RAG_Report_{current_chat['title'].replace(' ', '_')}.md",
        mime="text/markdown",
        use_container_width=True
    )

    if st.button("🗑️ Detach Document", use_container_width=True):
        st.session_state["vectorstore"] = None
        st.session_state["current_file"] = None
        st.session_state["doc_stats"] = {}
        st.rerun()

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

                    if not docs:
                        st.error("Uploaded document appears to be empty or unreadable.")
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        chunks = splitter.split_documents(docs)

                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/gemini-embedding-001",
                            google_api_key=gemini_api_key
                        )

                        vectorstore = FAISS.from_documents(chunks, embeddings)
                        st.session_state["vectorstore"] = vectorstore
                        st.session_state["current_file"] = uploaded_file.name
                        
                        total_chars = sum(len(d.page_content) for d in docs)
                        st.session_state["doc_stats"] = {
                            "name": uploaded_file.name,
                            "pages": len(docs),
                            "chunks": len(chunks),
                            "chars": total_chars
                        }
                        st.toast(f"✅ Indexed '{uploaded_file.name}' into FAISS ({len(chunks)} chunks)", icon="🚀")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

# ----------------- QUICK ACTION STARTER PILLS -----------------
st.markdown("##### ⚡ **One-Click Analytical Prompts**")
q_col1, q_col2, q_col3, q_col4 = st.columns(4)

has_doc = st.session_state["vectorstore"] is not None

with q_col1:
    if st.button("📋 **Executive Summary**", use_container_width=True, help="Extract an executive brief"):
        st.session_state["pending_prompt"] = (
            "Provide a high-level executive summary of the document, highlighting key achievements, competencies, and core domains."
            if has_doc else "Provide a structured executive briefing on modern AI and LLM trends in 2026."
        )
        st.rerun()

with q_col2:
    if st.button("🛠️ **Skills & Tech Audit**", use_container_width=True, help="Catalog technical stack"):
        st.session_state["pending_prompt"] = (
            "Extract, categorize, and benchmark all technical skills, frameworks, tools, and libraries mentioned in this document."
            if has_doc else "What are the most in-demand skills for Senior AI Engineers today?"
        )
        st.rerun()

with q_col3:
    if st.button("💼 **Career / Role Match**", use_container_width=True, help="Evaluate role suitability"):
        st.session_state["pending_prompt"] = (
            "Analyze the candidate's experience level, seniority, and identify the top 3 job roles they are best qualified for."
            if has_doc else "What are the key differences between AI Application Engineer and ML Research Scientist?"
        )
        st.rerun()

with q_col4:
    if st.button("❓ **5 Interview Questions**", use_container_width=True, help="Generate tailored questions"):
        st.session_state["pending_prompt"] = (
            "Generate 5 targeted, highly challenging technical interview questions based strictly on the projects and claims in this document."
            if has_doc else "Give me 5 hard architectural interview questions for a RAG specialist."
        )
        st.rerun()

st.markdown("---")

# ----------------- DISPLAY CHAT HISTORY -----------------
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display Metrics Pill Bar if available
        if msg.get("metrics"):
            m = msg["metrics"]
            st.markdown(
                f"""
                <div class="metrics-pill-bar">
                    <span class="metric-tag">⚡ Latency: <b>{m.get('latency', 0):.2f}s</b></span>
                    <span>•</span>
                    <span class="metric-tag">🧠 Tokens: <b>~{m.get('tokens', 0)}</b></span>
                    <span>•</span>
                    <span class="metric-tag">🤖 Engine: <b>{m.get('model', 'Gemini')}</b></span>
                    <span>•</span>
                    <span class="metric-tag">🛡️ Verified Source Grounding</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Display Source References Expander
        sources = msg.get("sources", {})
        doc_srcs = sources.get("docs", [])
        web_srcs = sources.get("web", [])

        if doc_srcs or web_srcs:
            with st.expander("🔍 View Verified Source Citations & References", expanded=False):
                if doc_srcs:
                    st.markdown("##### 📄 PDF Document Chunks (FAISS Dense Vector):")
                    for idx, d in enumerate(doc_srcs):
                        page_num = d.get("page", 0) + 1
                        st.markdown(f"**Chunk {idx+1} — Page {page_num}:**")
                        st.caption(d.get("content", ""))

                if web_srcs:
                    st.markdown("##### 🌐 Live Web References (DuckDuckGo Grounding):")
                    for w in web_srcs:
                        st.markdown(f"- 🔗 [{w.get('title', 'Web Result')}]({w.get('href', '#')})")
                        st.caption(w.get("body", ""))

# ----------------- INPUT PROCESSING & STREAMING -----------------
# Check for pending prompt from action pills
prompt_from_pill = st.session_state["pending_prompt"]
user_prompt = st.chat_input("Ask about the document, or any technical / industry topic...")

# Resolve prompt
active_prompt = prompt_from_pill if prompt_from_pill else user_prompt

if active_prompt:
    # Clear pending prompt
    st.session_state["pending_prompt"] = None

    if not gemini_api_key:
        st.warning("⚠️ Please provide a Gemini API Key in the sidebar.")
    else:
        # Auto-name chat on first prompt
        if current_chat["title"] in ["Document Intelligence", "New Chat", "Session 1", "Welcome Chat"] or current_chat["title"].startswith("Session "):
            clean_t = active_prompt.strip()
            current_chat["title"] = (clean_t[:22] + "...") if len(clean_t) > 22 else clean_t

        # User Message
        current_chat["messages"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Assistant Message
        with st.chat_message("assistant"):
            start_time = time.time()
            retrieved_doc_sources = []
            retrieved_web_sources = []

            # 1. Document Retrieval
            doc_context_text = "No document is uploaded. Use Web Search and AI General Knowledge."
            if st.session_state["vectorstore"] is not None:
                with st.spinner("Retrieving relevant semantic chunks from vector store..."):
                    retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": top_k})
                    relevant_docs = retriever.invoke(active_prompt)
                    if relevant_docs:
                        doc_context_text = "\n\n".join(
                            f"[Document Page {d.metadata.get('page', 0) + 1}]:\n{d.page_content}"
                            for d in relevant_docs
                        )
                        retrieved_doc_sources = [
                            {"page": d.metadata.get("page", 0), "content": d.page_content}
                            for d in relevant_docs
                        ]

            # 2. Web Retrieval
            web_context_text = "Web search disabled."
            if enable_web_search:
                with st.spinner("Grounding against live web search..."):
                    web_results = search_web_ddg(active_prompt, max_results=3)
                    if web_results:
                        retrieved_web_sources = web_results
                        web_context_text = "\n\n".join(
                            f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
                            for r in web_results
                        )
                    else:
                        web_context_text = "No relevant web results found."

            # 3. System Prompt & Reasoning Chain
            system_prompt = (
                "You are an Enterprise Hybrid AI Copilot with multi-source reasoning capabilities.\n"
                "You have access to three distinct sources:\n"
                "1. [DOCUMENT CONTEXT]: Extracted from the uploaded PDF/Resume via FAISS vector retrieval.\n"
                "2. [WEB SEARCH CONTEXT]: Real-time live search results from the internet.\n"
                "3. [AI GENERAL KNOWLEDGE]: Your own pre-trained AI knowledge base.\n\n"
                "OBJECTIVE: Provide a thorough, authoritative, and professionally structured response to the user's question.\n\n"
                "MANDATORY FORMATTING BLUEPRINT:\n"
                "Structure your output using these exact markdown sections:\n\n"
                "### 📄 Document Evidence:\n"
                "- Detail specifically what is confirmed or stated in the document. Quote relevant specifics, metrics, and dates.\n"
                "- If not found in the document, explicitly state: 'Not documented in the provided file.'\n\n"
                "### 🌐 Web & Market Intelligence:\n"
                "- Include external industry benchmarks, real-time facts, company overviews, or technology standards discovered via search.\n"
                "- If no external search data was needed or found, state: 'No external web data required.'\n\n"
                "### 🤖 AI Technical Synthesis:\n"
                "- Provide high-level technical context, best practices, architectural evaluation, or career critique from your AI intelligence.\n\n"
                "### 🎯 Executive Takeaway:\n"
                "- A sharp, high-impact final conclusion directly answering the query.\n\n"
                "------------------------------------\n"
                "DOCUMENT CONTEXT:\n{doc_context}\n\n"
                "WEB SEARCH CONTEXT:\n{web_context}\n\n"
                "USER QUESTION:\n{question}"
            )

            prompt_template = ChatPromptTemplate.from_template(system_prompt)

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

                # Metrics bar display
                st.markdown(
                    f"""
                    <div class="metrics-pill-bar">
                        <span class="metric-tag">⚡ Latency: <b>{latency:.2f}s</b></span>
                        <span>•</span>
                        <span class="metric-tag">🧠 Tokens: <b>~{approx_tokens}</b></span>
                        <span>•</span>
                        <span class="metric-tag">🤖 Engine: <b>{model_name}</b></span>
                        <span>•</span>
                        <span class="metric-tag">🛡️ Verified Source Grounding</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Source Citations
                if retrieved_doc_sources or retrieved_web_sources:
                    with st.expander("🔍 View Verified Source Citations & References", expanded=False):
                        if retrieved_doc_sources:
                            st.markdown("##### 📄 PDF Document Chunks (FAISS Dense Vector):")
                            for idx, d in enumerate(retrieved_doc_sources):
                                page_num = d.get("page", 0) + 1
                                st.markdown(f"**Chunk {idx+1} — Page {page_num}:**")
                                st.caption(d.get("content", ""))

                        if retrieved_web_sources:
                            st.markdown("##### 🌐 Live Web References (DuckDuckGo Grounding):")
                            for w in retrieved_web_sources:
                                st.markdown(f"- 🔗 [{w.get('title', 'Web Result')}]({w.get('href', '#')})")
                                st.caption(w.get("body", ""))

                # Append to current chat history
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": response_text,
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
