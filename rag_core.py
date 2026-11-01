"""Shared Hybrid RAG helpers used by Streamlit UI and Voice Agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
META_PATH = DATA_DIR / "index_meta.json"

def build_system_prompt(
    *,
    enable_pdf: bool = True,
    enable_web: bool = True,
    enable_gemini: bool = True,
    voice_mode: bool = False,
) -> str:
    """Build prompt text based on which answer sources the user enabled."""
    enabled = []
    if enable_pdf:
        enabled.append("1. [DOCUMENT CONTEXT]: uploaded PDF via FAISS retrieval")
    if enable_web:
        enabled.append("2. [WEB SEARCH CONTEXT]: live internet search results")
    if enable_gemini:
        enabled.append("3. [AI GENERAL KNOWLEDGE]: Gemini's own pre-trained knowledge")

    sources_block = "\n".join(enabled) if enabled else "No sources enabled."

    rules = []
    if not enable_pdf:
        rules.append("- Document/PDF retrieval is DISABLED. Ignore any document evidence.")
    if not enable_web:
        rules.append("- Internet search is DISABLED. Do not invent web findings.")
    if not enable_gemini:
        rules.append(
            "- Gemini general knowledge is DISABLED. Answer ONLY from the enabled "
            "document/web context below. If that context is insufficient, say you "
            "cannot answer from the enabled sources."
        )
    else:
        rules.append(
            "- You may use Gemini general knowledge where document/web context is thin "
            "(only if Gemini knowledge is listed as enabled)."
        )
    rules_block = "\n".join(rules)

    if voice_mode:
        return (
            "You are a voice-friendly Hybrid RAG assistant.\n"
            f"Enabled sources:\n{sources_block}\n\n"
            f"Rules:\n{rules_block}\n\n"
            "Speak naturally: 2-5 short sentences, no markdown headings, no bullet lists, no emoji.\n"
            "Lead with the strongest enabled source.\n\n"
            "DOCUMENT CONTEXT:\n{doc_context}\n\n"
            "WEB SEARCH CONTEXT:\n{web_context}\n\n"
            "USER QUESTION:\n{question}"
        )

    sections = ["### 🎯 Executive Takeaway:\n- Direct final answer to the query.\n"]
    if enable_pdf:
        sections.insert(
            0,
            "### 📄 Document Evidence:\n"
            "- What the document confirms. If missing: 'Not documented in the provided file.'\n\n",
        )
    if enable_web:
        insert_at = 1 if enable_pdf else 0
        sections.insert(
            insert_at,
            "### 🌐 Web & Market Intelligence:\n"
            "- External facts from search. If none: 'No external web data required.'\n\n",
        )
    if enable_gemini:
        insert_at = len(sections) - 1
        sections.insert(
            insert_at,
            "### 🤖 AI Technical Synthesis:\n"
            "- Technical context / best practices from Gemini knowledge.\n\n",
        )

    return (
        "You are an Enterprise Hybrid AI Copilot with multi-source reasoning.\n"
        f"Enabled sources:\n{sources_block}\n\n"
        f"Rules:\n{rules_block}\n\n"
        "OBJECTIVE: Thorough, authoritative answer using ONLY the enabled sources.\n\n"
        "MANDATORY FORMATTING — use these markdown sections:\n\n"
        + "".join(sections)
        + "------------------------------------\n"
        "DOCUMENT CONTEXT:\n{doc_context}\n\n"
        "WEB SEARCH CONTEXT:\n{web_context}\n\n"
        "USER QUESTION:\n{question}"
    )


# Defaults keep backward compatibility for CLI scripts
SYSTEM_PROMPT = build_system_prompt()
VOICE_SYSTEM_PROMPT = build_system_prompt(voice_mode=True)


def search_web_ddg(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Live web search via DuckDuckGo."""
    try:
        from ddgs import DDGS

        results = list(DDGS().text(query, max_results=max_results))
        formatted = []
        for r in results:
            formatted.append(
                {
                    "title": r.get("title", "Web Source"),
                    "href": r.get("href", "#"),
                    "body": r.get("body", ""),
                }
            )
        return formatted
    except Exception:
        return []


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def save_vectorstore(
    vectorstore: FAISS,
    *,
    file_name: str,
    pages: int,
    chunks: int,
    chars: int,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    META_PATH.write_text(
        json.dumps(
            {
                "name": file_name,
                "pages": pages,
                "chunks": chunks,
                "chars": chars,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_vectorstore(api_key: str) -> tuple[FAISS | None, dict[str, Any]]:
    if not INDEX_DIR.exists() or not META_PATH.exists():
        return None, {}
    try:
        embeddings = get_embeddings(api_key)
        store = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        return store, meta
    except Exception:
        return None, {}


def clear_persisted_index() -> None:
    if META_PATH.exists():
        META_PATH.unlink()
    if INDEX_DIR.exists():
        for path in INDEX_DIR.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(INDEX_DIR.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        INDEX_DIR.rmdir()


def retrieve_doc_context(
    vectorstore: FAISS | None,
    question: str,
    top_k: int = 4,
    *,
    enabled: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    if not enabled:
        return "Document/PDF retrieval disabled by the user.", []
    if vectorstore is None:
        return "No document is uploaded.", []

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.invoke(question)
    if not relevant_docs:
        return "No relevant document chunks found.", []

    doc_context = "\n\n".join(
        f"[Document Page {d.metadata.get('page', 0) + 1}]:\n{d.page_content}"
        for d in relevant_docs
    )
    sources = [
        {"page": d.metadata.get("page", 0), "content": d.page_content}
        for d in relevant_docs
    ]
    return doc_context, sources


def retrieve_web_context(
    question: str,
    *,
    enabled: bool = True,
    max_results: int = 3,
) -> tuple[str, list[dict[str, str]]]:
    if not enabled:
        return "Internet search disabled by the user.", []

    web_results = search_web_ddg(question, max_results=max_results)
    if not web_results:
        return "No relevant web results found.", []

    web_context = "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in web_results
    )
    return web_context, web_results


def answer_hybrid(
    question: str,
    *,
    api_key: str,
    model_name: str = "gemini-3.5-flash",
    vectorstore: FAISS | None = None,
    enable_pdf: bool = True,
    enable_web_search: bool = True,
    enable_gemini: bool = True,
    top_k: int = 4,
    voice_mode: bool = False,
) -> dict[str, Any]:
    """Run hybrid RAG and return answer text plus sources."""
    doc_context, doc_sources = retrieve_doc_context(
        vectorstore, question, top_k=top_k, enabled=enable_pdf
    )
    web_context, web_sources = retrieve_web_context(
        question, enabled=enable_web_search
    )

    prompt_template = ChatPromptTemplate.from_template(
        build_system_prompt(
            enable_pdf=enable_pdf,
            enable_web=enable_web_search,
            enable_gemini=enable_gemini,
            voice_mode=voice_mode,
        )
    )
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=api_key,
    )
    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "doc_context": doc_context,
            "web_context": web_context,
            "question": question,
        }
    )
    return {
        "answer": answer,
        "sources": {"docs": doc_sources, "web": web_sources},
        "doc_context": doc_context,
        "web_context": web_context,
    }


def resolve_gemini_api_key() -> str:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    )
