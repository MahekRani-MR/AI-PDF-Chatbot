import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from fpdf import FPDF

from utils.vectorstore import create_vectorstore
from utils.rag_pipeline import create_rag_chain

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Dark Violet / Pink Theme (matches reference)
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', -apple-system, sans-serif; }

    :root {
        --bg: #0b0a18;
        --panel: #12111f;
        --panel-2: #171526;
        --border: #262238;
        --violet: #8b5cf6;
        --violet-soft: rgba(139, 92, 246, 0.16);
        --pink: #ec4899;
        --text: #eceafb;
        --text-dim: #9791b3;
    }

    .stApp {
        background:
            radial-gradient(circle at 20% -10%, rgba(139,92,246,0.16) 0%, transparent 45%),
            radial-gradient(circle at 85% 5%, rgba(236,72,153,0.10) 0%, transparent 40%),
            var(--bg);
    }

    .block-container {
        max-width: 1050px;
        padding-top: 1.2rem;
        padding-bottom: 5.5rem;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--bg);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        font-family: 'Poppins', sans-serif;
        color: var(--text);
    }

    /* Sticky chat-controls block pinned to the top of the sidebar */
    section[data-testid="stSidebar"] .st-key-chat_controls {
        position: sticky;
        top: -1rem;
        z-index: 5;
        background: var(--bg);
        padding: 0.4rem 0 14px 0;
        margin-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    .sb-section-label {
        font-family: 'Poppins', sans-serif;
        font-size: 12px;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: var(--text-dim);
        font-weight: 700;
        margin: 4px 0 8px 0;
    }

    /* Compact buttons inside the chat-controls panel */
    .st-key-chat_controls .stButton button {
        padding: 6px 10px !important;
        font-size: 13px !important;
    }

    .stFileUploader section {
        background: var(--panel) !important;
        border: 1.5px dashed rgba(139,92,246,0.55) !important;
        border-radius: 14px !important;
    }

    /* ---------- Hero card ---------- */
    .hero {
        text-align: center;
        padding: 34px 24px 30px 24px;
        margin-bottom: 24px;
        border-radius: 24px;
        background: linear-gradient(160deg, #171233 0%, #1a1130 55%, #1f1230 100%);
        border: 1px solid rgba(139,92,246,0.28);
        box-shadow: 0 0 0 1px rgba(139,92,246,0.06), 0 20px 50px rgba(0,0,0,0.45);
    }

    .badge {
        display: inline-block;
        background: rgba(139,92,246,0.14);
        color: #cbb8ff;
        padding: 8px 20px;
        border-radius: 24px;
        font-size: 13.5px;
        font-weight: 700;
        margin-bottom: 20px;
        border: 1px solid rgba(139,92,246,0.4);
    }

    .title-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        margin-bottom: 4px;
    }

    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 46px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #a78bfa 0%, #c084fc 45%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        color: var(--text-dim);
        font-size: 16.5px;
        margin-top: 12px;
    }

    /* ---------- Mode cards ---------- */
    .mode-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 4px solid var(--violet);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 22px;
    }

    .mode-title {
        color: #c4b5fd;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .mode-text {
        color: var(--text-dim);
        font-size: 14.5px;
    }

    .pdf-card {
        background: var(--panel);
        border: 1px solid rgba(52,211,153,0.35);
        border-left: 4px solid #34d399;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 22px;
    }

    .pdf-title {
        color: #6ee7b7;
        font-weight: 700;
        font-size: 17px;
    }

    .pdf-name {
        color: var(--text);
        margin-top: 5px;
        font-weight: 500;
        font-size: 14.5px;
    }

    /* ---------- Sidebar config cards ---------- */
    .cfg-card {
        background: var(--panel);
        border: 1px solid var(--border);
        padding: 14px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        transition: border-color 0.2s ease;
    }

    .cfg-card:hover {
        border-color: rgba(139,92,246,0.5);
    }

    .cfg-label {
        color: var(--text-dim);
        font-size: 11px;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }

    .cfg-value {
        color: var(--text);
        font-size: 15px;
        font-weight: 600;
        margin-top: 3px;
    }

    /* ---------- Source chip ---------- */
    .source-card {
        display: inline-block;
        background: var(--violet-soft);
        border: 1px solid rgba(139,92,246,0.4);
        color: #cbb8ff;
        padding: 6px 14px;
        border-radius: 20px;
        margin-top: 10px;
        margin-right: 6px;
        font-size: 12.5px;
        font-weight: 600;
    }

    /* ---------- Chat messages ---------- */
    div[data-testid="stChatMessage"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    div[data-testid="stChatMessageContent"] {
        color: var(--text);
        font-size: 15px;
        line-height: 1.65;
    }

    div[data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #22d3ee, #0ea5e9) !important;
        border-radius: 10px !important;
    }

    /* ---------- Chat input (pill bar) ---------- */
    div[data-testid="stChatInput"] {
        display: flex !important;
        align-items: center !important;
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 999px !important;
        padding: 6px 8px 6px 20px !important;
        margin-bottom: 34px;
        box-sizing: border-box !important;
    }

    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] div[class*="Input"],
    div[data-testid="stChatInput"] div[data-baseweb] {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        flex: 1 1 auto !important;
        width: 100% !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: var(--text) !important;
        font-size: 14.5px !important;
        padding: 8px 0 !important;
        margin: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stChatInput"] button {
        border-radius: 999px !important;
        margin-right: 2px !important;
        flex: 0 0 auto !important;
    }

    div[data-testid="stChatInput"] textarea:focus,
    div[data-testid="stChatInput"] textarea:focus-visible,
    div[data-testid="stChatInput"] textarea:invalid,
    div[data-testid="stChatInput"] textarea:required,
    div[data-testid="stChatInput"] textarea:-moz-ui-invalid {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        outline: none !important;
        border-color: var(--violet) !important;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.2) !important;
    }

    /* ---------- Buttons ---------- */
    .stButton button {
        background: linear-gradient(90deg, #8b5cf6, #ec4899) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 999px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(139,92,246,0.4) !important;
    }

    /* Secondary (download) buttons — outline style, not gradient */
    .stDownloadButton button {
        background: var(--panel-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        font-weight: 600 !important;
        border-radius: 999px !important;
        padding: 6px 10px !important;
        font-size: 13px !important;
    }

    .stDownloadButton button:hover {
        border-color: var(--violet) !important;
        color: #cbb8ff !important;
    }

    /* ---------- Restore list ---------- */
    .saved-chat-meta {
        color: var(--text-dim);
        font-size: 11px;
        margin: -6px 0 8px 4px;
    }

    /* ---------- Alerts ---------- */
    .stAlert { border-radius: 12px; }

    /* ---------- Fixed footer ---------- */
    .footer {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 999;
        background: rgba(11, 10, 24, 0.92);
        backdrop-filter: blur(10px);
        border-top: 1px solid var(--border);
        text-align: center;
        color: var(--text-dim);
        padding: 9px 0;
        font-size: 13px;
    }

    .footer span {
        background: linear-gradient(90deg, #a78bfa, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.35); border-radius: 4px; }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}   # {id: {"title", "messages", "ts", "timestamp"}}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ============================================================
# CHAT-MANAGEMENT HELPERS
# ============================================================

def save_current_session():
    """Save (or update) the current conversation into chat_sessions."""

    if not st.session_state.messages:
        return

    first_user_msg = next(
        (m["content"] for m in st.session_state.messages if m["role"] == "user"),
        "New chat"
    )

    title = first_user_msg[:35] + "…" if len(first_user_msg) > 35 else first_user_msg

    st.session_state.chat_sessions[st.session_state.session_id] = {
        "title": title,
        "messages": st.session_state.messages.copy(),
        "ts": datetime.now(),
        "timestamp": datetime.now().strftime("%b %d, %I:%M %p")
    }


def get_chat_txt():
    """Render the current conversation as plain text."""

    if not st.session_state.messages:
        return "No conversation yet."

    lines = []

    for m in st.session_state.messages:

        role = "You" if m["role"] == "user" else "Assistant"

        lines.append(f"{role}: {m['content']}")

        if m.get("sources"):
            pages = ", ".join(str(p) for p in m["sources"])
            lines.append(f"   Sources: page(s) {pages}")

        lines.append("")

    return "\n".join(lines)


def _prepare_pdf_text(text, max_word_len=60):
    """
    fpdf2 raises FPDFException('Not enough horizontal space to render a
    single character') when a single unbroken 'word' (e.g. a long URL,
    hash, or run of text with no spaces) is wider than the page itself,
    since it has no space to wrap on. This forces a break every
    `max_word_len` characters inside any such long token so it always
    has somewhere to wrap.
    """

    text = text.encode("latin-1", "replace").decode("latin-1")

    wrapped_lines = []

    for line in text.split("\n"):

        wrapped_words = []

        for word in line.split(" "):

            while len(word) > max_word_len:
                wrapped_words.append(word[:max_word_len])
                word = word[max_word_len:]

            wrapped_words.append(word)

        wrapped_lines.append(" ".join(wrapped_words))

    return "\n".join(wrapped_lines)


def get_chat_pdf():
    """Render the current conversation as a downloadable PDF (bytes)."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if not st.session_state.messages:

        pdf.set_font("Helvetica", size=11)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, "No conversation yet.")

    else:

        for m in st.session_state.messages:

            role = "You" if m["role"] == "user" else "Assistant"

            pdf.set_font("Helvetica", style="B", size=11)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 7, role)

            pdf.set_font("Helvetica", size=11)
            pdf.set_x(pdf.l_margin)
            safe_content = _prepare_pdf_text(m["content"])
            pdf.multi_cell(0, 7, safe_content)

            if m.get("sources"):
                pages = ", ".join(str(p) for p in m["sources"])
                pdf.set_font("Helvetica", style="I", size=9)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 6, f"Sources: page(s) {pages}")

            pdf.ln(3)

    raw = pdf.output()

    if isinstance(raw, str):
        raw = raw.encode("latin-1")

    return bytes(raw)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # Sticky chat-controls panel (stays pinned at the top)
    # --------------------------------------------------------

    with st.container(key="chat_controls"):

        st.markdown('<div class="sb-section-label">💬 Chat</div>', unsafe_allow_html=True)

        if st.button("🆕 New Chat", use_container_width=True, key="new_chat_btn"):

            if st.session_state.messages:
                save_current_session()

            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        c1, c2 = st.columns(2)

        with c1:
            if st.button("💾 Save", use_container_width=True, key="save_chat_btn"):

                if st.session_state.messages:
                    save_current_session()
                    st.toast("Conversation saved", icon="💾")
                else:
                    st.toast("Nothing to save yet", icon="⚠️")

        with c2:
            if st.button("🗑️ Delete", use_container_width=True, key="delete_current_btn"):

                st.session_state.chat_sessions.pop(st.session_state.session_id, None)
                st.session_state.messages = []
                st.toast("Chat deleted", icon="🗑️")
                st.rerun()

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "📥 TXT",
                data=get_chat_txt(),
                file_name=f"chat_{st.session_state.session_id[:8]}.txt",
                mime="text/plain",
                use_container_width=True,
                disabled=not st.session_state.messages,
                key="download_txt_btn"
            )

        with d2:
            st.download_button(
                "📥 PDF",
                data=get_chat_pdf(),
                file_name=f"chat_{st.session_state.session_id[:8]}.pdf",
                mime="application/pdf",
                use_container_width=True,
                disabled=not st.session_state.messages,
                key="download_pdf_btn"
            )

        with st.expander("🔄 Restore Previous", expanded=False):

            if not st.session_state.chat_sessions:

                st.caption("No saved conversations yet.")

            else:

                sorted_sessions = sorted(
                    st.session_state.chat_sessions.items(),
                    key=lambda kv: kv[1]["ts"],
                    reverse=True
                )

                for sid, session in sorted_sessions:

                    r_col, x_col = st.columns([4, 1])

                    with r_col:

                        if st.button(
                            f"🗂️ {session['title']}",
                            key=f"restore_{sid}",
                            use_container_width=True
                        ):
                            st.session_state.messages = session["messages"].copy()
                            st.session_state.session_id = sid
                            st.rerun()

                    with x_col:

                        if st.button("✕", key=f"delete_saved_{sid}"):
                            del st.session_state.chat_sessions[sid]
                            st.rerun()

                    st.markdown(
                        f'<div class="saved-chat-meta">{session["timestamp"]}</div>',
                        unsafe_allow_html=True
                    )

    # --------------------------------------------------------
    # Upload Document
    # --------------------------------------------------------

    st.markdown("### 📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload a PDF document to ask questions about its contents.",
        label_visibility="collapsed"
    )

    st.divider()

    if not GROQ_API_KEY:
        st.warning("⚠️ GROQ_API_KEY not found. Add it to your .env file.")
    elif uploaded_file:
        st.success("PDF uploaded successfully!")
    else:
        st.info("Upload a PDF to enable document-based questions.")

    st.divider()

    # --------------------------------------------------------
    # Configuration (moved to the bottom of the sidebar)
    # --------------------------------------------------------

    st.markdown("### ⚙️ Configuration")

    st.markdown(
        """
        <div class="cfg-card">
            <div class="cfg-label">Model</div>
            <div class="cfg-value">🦙 Llama 3.1 8B Instant</div>
        </div>
        <div class="cfg-card">
            <div class="cfg-label">Embeddings</div>
            <div class="cfg-value">🧠 MiniLM-L6-v2</div>
        </div>
        <div class="cfg-card">
            <div class="cfg-label">Vector Database</div>
            <div class="cfg-value">📦 FAISS</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# MAIN HEADER (hero card)
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="badge">🚀 Turn PDFs Into Conversations</div>
        <div class="title-row">
            <div class="main-title">🤖 AI PDF Chatbot</div>
        </div>
        <div class="subtitle">Ask general questions or upload a PDF for source-backed answers.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PDF PROCESSING
# ============================================================

if uploaded_file:

    os.makedirs("uploaded_pdfs", exist_ok=True)

    pdf_path = os.path.join("uploaded_pdfs", uploaded_file.name)

    # Process only if a new PDF is uploaded
    if st.session_state.pdf_name != uploaded_file.name:

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("📚 Reading PDF and creating knowledge base..."):

            vectorstore = create_vectorstore(pdf_path)
            st.session_state.chat = create_rag_chain(vectorstore)

        st.session_state.pdf_name = uploaded_file.name

        # Clear old conversation when a new PDF is uploaded
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

        st.success("PDF processed successfully!")

# ============================================================
# MODE INDICATOR
# ============================================================

if uploaded_file:

    st.markdown(
        f"""
        <div class="pdf-card">
            <div class="pdf-title">📄 PDF Mode Active</div>
            <div class="pdf-name">{uploaded_file.name}</div>
            <div class="mode-text" style="margin-top:8px;">
                Questions related to this document will use Retrieval-Augmented Generation (RAG).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="mode-card">
            <div class="mode-title">🌎 General AI Mode</div>
            <div class="mode-text">
                You can ask questions without uploading a PDF.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    avatar = "🧑" if message["role"] == "user" else "🤖"

    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])

        if message.get("sources"):

            chips = "".join(
                f'<span class="source-card">📄 Page {page}</span>'
                for page in message["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)

# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your PDF..." if uploaded_file else "Ask anything..."
)

# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    if not GROQ_API_KEY:

        st.error("GROQ_API_KEY is missing. Add it to your .env file to chat.")

    else:

        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.chat_message("assistant", avatar="🤖"):

            with st.spinner("Thinking..."):

                try:

                    if uploaded_file and st.session_state.chat:

                        # ================================
                        # PDF / RAG MODE
                        # ================================

                        response = st.session_state.chat(question)
                        answer = response["answer"]

                        sources = []

                        for doc in response.get("sources", []):

                            page = doc.metadata.get("page")

                            if page is not None:

                                page_number = page + 1

                                if page_number not in sources:
                                    sources.append(page_number)

                        st.markdown(answer)

                        if sources:
                            chips = "".join(
                                f'<span class="source-card">📄 Page {p}</span>'
                                for p in sources
                            )
                            st.markdown(chips, unsafe_allow_html=True)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            }
                        )

                    else:

                        # ================================
                        # GENERAL AI MODE
                        # ================================

                        llm = ChatGroq(
                            model="llama-3.1-8b-instant",
                            api_key=GROQ_API_KEY,
                            temperature=0.3
                        )

                        response = llm.invoke(question)
                        answer = response.content

                        st.markdown(answer)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": []
                            }
                        )

                    # Keep the saved copy of this session in sync, if it
                    # was already saved once, so Restore reflects the
                    # latest turn without requiring another manual save.
                    if st.session_state.session_id in st.session_state.chat_sessions:
                        save_current_session()

                except Exception as e:

                    st.error(f"Something went wrong: {str(e)}")

# ============================================================
# FOOTER (fixed, always visible below chat input)
# ============================================================

st.markdown(
    """
    <div class="footer">
        🚀 Built with <span>Streamlit</span> · LangChain · FAISS · HuggingFace · Groq Llama 3.1
    </div>
    """,
    unsafe_allow_html=True
)