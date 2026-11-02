"""DocuMind — Streamlit UI for chatting with PDFs."""
import os

import streamlit as st
from dotenv import load_dotenv

from src.llm import GeminiClient
from src.rag import RAGPipeline

load_dotenv()

st.set_page_config(page_title="DocuMind — Chat with your PDFs", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> GeminiClient:
    return GeminiClient(api_key=api_key)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("indexed_files", [])


init_state()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📄 DocuMind")
    st.caption("RAG-powered PDF assistant · Gemini + FAISS")

    # The server-side key (from .env or Streamlit secrets) is never sent to the browser.
    server_key = os.getenv("GEMINI_API_KEY", "")
    if not server_key:
        try:  # Streamlit Community Cloud secrets
            server_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:  # no secrets file locally
            server_key = ""
    user_key = st.text_input(
        "Gemini API key" + (" (optional)" if server_key else ""), type="password",
        help="Get a free key at https://aistudio.google.com/apikey",
        placeholder="Using the app's built-in key" if server_key else "Paste your key",
    )
    api_key = user_key or server_key

    with st.expander("⚙️ Retrieval settings"):
        chunk_size = st.slider("Chunk size (chars)", 300, 2000, 1000, 100)
        overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
        top_k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("🔍 Build knowledge base", width="stretch", disabled=not files):
        if not api_key:
            st.error("Please add your Gemini API key first.")
        else:
            client = get_client(api_key)
            pipeline = RAGPipeline(client.embed, client.generate, chunk_size, overlap, top_k)
            total = 0
            progress = st.progress(0.0, text="Indexing…")
            for i, f in enumerate(files, start=1):
                total += pipeline.ingest(f, f.name)
                progress.progress(i / len(files), text=f"Indexed {f.name}")
            st.session_state.pipeline = pipeline
            st.session_state.indexed_files = [f.name for f in files]
            st.session_state.messages = []
            st.success(f"Indexed {len(files)} file(s) into {total} chunks.")

    if st.session_state.indexed_files:
        st.markdown("**Knowledge base**")
        for name in st.session_state.indexed_files:
            st.markdown(f"- {name}")

    if st.button("🗑️ Clear chat", width="stretch"):
        st.session_state.messages = []

# ---------------- Main chat ----------------
st.header("Ask anything about your documents")

if st.session_state.pipeline is None:
    st.info("👈 Upload one or more PDFs and click **Build knowledge base** to begin.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("e.g. Summarise the key findings", disabled=st.session_state.pipeline is None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = st.session_state.pipeline.ask(question, st.session_state.messages[:-1])
                st.markdown(answer.text)
                with st.expander(f"📚 Sources ({len(answer.sources)})"):
                    for chunk, score in answer.sources:
                        st.markdown(f"**{chunk.source} — page {chunk.page}** · similarity `{score:.2f}`")
                        st.caption(chunk.text[:400] + ("…" if len(chunk.text) > 400 else ""))
                st.session_state.messages.append({"role": "assistant", "content": answer.text})
            except Exception as exc:  # surface API errors nicely
                st.error(f"Something went wrong: {exc}")
