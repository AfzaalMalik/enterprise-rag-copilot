# Groundly

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Flash-orange.svg)](https://ai.google.dev/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-red.svg)](https://github.com/facebookresearch/faiss)

**Groundly** is a hybrid RAG app (PDF + optional web + Gemini knowledge) with chat voice replies. Toggle each source from the sidebar.

---

## 🌟 Key Features

- **📄 Document Grounding:** High-precision PDF document ingestion, recursive character chunking, and FAISS dense vector search.
- **🌐 Real-Time Web Fallback:** Automatically fetches live web search results via DuckDuckGo when queries cannot be answered by document context alone.
- **🛡️ Strict Attribution & Fact Isolation:** Isolates answers into distinct visual sections:
  - 📄 *Document Evidence (with Page Citations)*
  - 🌐 *Web & Market Intelligence*
  - 🤖 *AI Technical Synthesis*
  - 🎯 *Executive Takeaway*
- **⚡ Real-Time Word-by-Word Streaming:** Instant token response using LangChain LCEL streaming.
- **📊 Observability & Metrics:** Live telemetry showing latency (seconds), token estimation, active model, and source grounding.
- **💬 Multi-Session Chat Management:** ChatGPT-style workspace list to create, switch, and delete multiple conversations.
- **🎙 Chat voice + Live Talk (local):** Mic in chat input; Live Talk needs the local realtime server.
- **📥 1-Click Report Export:** Download formatted briefing reports in Markdown.

---

## 🏗️ Architecture

```
User Query
   │
   ├──► [1] FAISS Vector Store ──► Retrieved Document Chunks (Page-Cited)
   ├──► [2] Live Web Search   ──► Real-Time Internet Snippets & URLs
   │
   ▼
Multi-Source Grounding Prompt
   │
   ▼
Google Gemini 3.5 Flash LLM
   │
   ▼
Real-Time Streamed Output with Isolated Attribution Badges
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/enterprise-rag-copilot.git
cd enterprise-rag-copilot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 5. Launch the application
```bash
streamlit run rag.py
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to **GitHub** (do not commit `.env`).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect GitHub and select this repo.
4. Main file path: `rag.py`.
5. **Advanced settings → Secrets**:
   ```toml
   GEMINI_API_KEY = "your_actual_key_here"
   ```
6. Deploy.

**Note:** PDF RAG, chat, web toggle, and chat-mic voice work on Streamlit Cloud. **Live Talk** (WebSocket server on `localhost:8765`) is local-only unless you host that server separately.

---

## 📜 License
Distributed under the MIT License.
