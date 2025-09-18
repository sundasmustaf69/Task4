# task4_streamlit_gemini.py
import streamlit as st
import google.generativeai as genai
import os
import tempfile
from typing import List

# -----------------------
# Page config + CSS
# -----------------------
st.set_page_config(page_title="Task 4 — RAG Chat (Gemini)", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #f6f8fb; }
.user { background:#e6f2ff; padding:10px; border-radius:10px; margin-bottom:5px;}
.ai { background:#f3f7e8; padding:10px; border-radius:10px; margin-bottom:5px;}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Session state init
# -----------------------
if "docstore" not in st.session_state:
    st.session_state.docstore: List[str] = []
if "messages" not in st.session_state:
    st.session_state.messages: List[dict] = []

# -----------------------
# Sidebar: Settings + KB Upload
# -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("🔑 Gemini API Key", type="password")
    model_name = st.selectbox("🧠 Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    temperature = st.slider("🌡 Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("📝 Max Tokens", 100, 2048, 512, 50)

    st.markdown("---")
    st.subheader("Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF / TXT", accept_multiple_files=True)
    if st.button("➕ Add files"):
        if uploaded_files:
            for uf in uploaded_files:
                fname = uf.name.lower()
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name
                text = ""
                if fname.endswith(".pdf"):
                    from pypdf import PdfReader
                    reader = PdfReader(tmp_path)
                    text = "\n".join([p.extract_text() or "" for p in reader.pages])
                else:
                    with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                if text.strip():
                    st.session_state.docstore.append(text)
                os.remove(tmp_path)
            st.success(f"✅ Added {len(uploaded_files)} file(s) to KB.")
        else:
            st.warning("No files uploaded.")

    if st.button("🗑 Clear KB"):
        st.session_state.docstore = []
        st.success("Knowledge base cleared.")

    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared.")

# -----------------------
# Stop if API key missing
# -----------------------
if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=api_key)
llm = genai.GenerativeModel(model_name)

# -----------------------
# Simple Semantic Search
# -----------------------
def semantic_search(query, top_k=3):
    if not st.session_state.docstore:
        return []
    query_words = query.lower().split()
    scored_docs = []
    for doc in st.session_state.docstore:
        score = sum(doc.lower().count(word) for word in query_words)
        scored_docs.append((score, doc))
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    top_docs = [doc for score, doc in scored_docs if score > 0]
    return top_docs[:top_k] if top_docs else ["No relevant info found."]

# -----------------------
# RAG Pipeline
# -----------------------
def rag_pipeline(question):
    context_docs = semantic_search(question)
    context = "\n".join(context_docs)
    prompt = f"""
You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer in a clear and human-like way:
"""
    response = llm.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )
    return response.text

# -----------------------
# Main Chat UI
# -----------------------
st.title("🤖 Task 4 — RAG Chat (Gemini)")

user_input = st.text_area("💬 Ask your question:", height=120)
if st.button("Ask ✨") and user_input:
    answer = rag_pipeline(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user'>🧑 <b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai'>🤖 <b>AI:</b> {msg['content']}</div>", unsafe_allow_html=True)
