# ⚡ Enterprise Hybrid RAG Copilot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash-orange.svg)](https://ai.google.dev/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-red.svg)](https://github.com/facebookresearch/faiss)

A production-grade, multi-source **Hybrid Retrieval-Augmented Generation (RAG) System** that combines dense document vector retrieval with real-time live internet grounding and large language model reasoning.

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
- **🌓 Dynamic Dark / Light Themes:** Modern glassmorphic interface with custom typography (*Plus Jakarta Sans* & *JetBrains Mono*).
- **📥 1-Click Report Export:** Download formatted executive briefing reports in Markdown format.

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

1. Push your repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select your repository.
4. Set Main file path to: `rag.py`.
5. Under **Advanced Settings ➔ Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your_actual_key_here"
   ```
6. Click **Deploy!** 🚀

---

## 📜 License
Distributed under the MIT License.
