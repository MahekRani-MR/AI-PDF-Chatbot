import os
import json
import uuid
from datetime import datetime
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

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
# CHAT HISTORY STORAGE
# ============================================================

HISTORY_FILE = "chat_history.json"


def load_chat_history():
    """Load saved conversations from local JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return {}

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_chat_history(history):
    """Save conversations to local JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.error(f"Unable to save chat history: {e}")


def create_new_chat():
    """Create a new empty conversation."""
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pdf_name": None,
        "messages": []
    }


def generate_chat_title(question):
    """Create a simple title from the first question."""
    title = question.strip().replace("\n", " ")

    if len(title) > 40:
        title = title[:40].rstrip() + "..."

    return title if title else "New Chat"


def get_chat_text(chat):
    """Convert a conversation into plain text."""
    lines = []

    lines.append("AI PDF Chatbot")
    lines.append("=" * 60)
    lines.append(f"Chat: {chat.get('title', 'New Chat')}")
    lines.append(f"Created: {chat.get('created_at', '')}")

    if chat.get("pdf_name"):
        lines.append(f"PDF: {chat['pdf_name']}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    for message in chat.get("messages", []):

        role = message.get("role", "")

        if role == "user":
            lines.append("YOU:")
        else:
            lines.append("AI:")

        lines.append(message.get("content", ""))

        sources = message.get("sources", [])

        if sources:
            lines.append(
                "Sources: " +
                ", ".join(f"Page {page}" for page in sources)
            )

        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    return "\n".join(lines)


def create_pdf(chat):
    """Create a downloadable PDF containing the conversation."""

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer
        )
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics

        buffer = BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ChatTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            spaceAfter=20
        )

        user_style = ParagraphStyle(
            "UserMessage",
            parent=styles["BodyText"],
            fontSize=11,
            leading=16,
            spaceAfter=10
        )

        assistant_style = ParagraphStyle(
            "AssistantMessage",
            parent=styles["BodyText"],
            fontSize=11,
            leading=16,
            spaceAfter=10
        )

        normal_style = ParagraphStyle(
            "NormalText",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=8
        )

        story = []

        story.append(
            Paragraph(
                "AI PDF Chatbot",
                title_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Chat:</b> {chat.get('title', 'New Chat')}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Created:</b> {chat.get('created_at', '')}",
                normal_style
            )
        )

        if chat.get("pdf_name"):
            story.append(
                Paragraph(
                    f"<b>PDF:</b> {chat['pdf_name']}",
                    normal_style
                )
            )

        story.append(Spacer(1, 15))

        for message in chat.get("messages", []):

            role = message.get("role")

            if role == "user":
                heading = "<b>You:</b>"
                style = user_style
            else:
                heading = "<b>AI:</b>"
                style = assistant_style

            content = message.get("content", "")

            # Escape HTML characters
            content = (
                content
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )

            story.append(
                Paragraph(
                    heading,
                    style
                )
            )

            story.append(
                Paragraph(
                    content,
                    style
                )
            )

            sources = message.get("sources", [])

            if sources:
                source_text = (
                    "<b>Sources:</b> " +
                    ", ".join(
                        f"Page {page}"
                        for page in sources
                    )
                )

                story.append(
                    Paragraph(
                        source_text,
                        normal_style
                    )
                )

            story.append(Spacer(1, 10))

        document.build(story)

        buffer.seek(0)

        return buffer

    except ImportError:
        return None


# ============================================================
# SESSION STATE
# ============================================================

if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_chat_history()

if "current_chat_id" not in st.session_state:

    new_chat = create_new_chat()

    st.session_state.all_chats[new_chat["id"]] = new_chat
    st.session_state.current_chat_id = new_chat["id"]

if "messages" not in st.session_state:
    current_chat = st.session_state.all_chats[
        st.session_state.current_chat_id
    ]

    st.session_state.messages = current_chat.get(
        "messages",
        []
    )

if "chat" not in st.session_state:
    st.session_state.chat = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_current_chat():
    return st.session_state.all_chats[
        st.session_state.current_chat_id
    ]


def sync_current_chat():
    """Synchronize session messages with current saved chat."""

    current_chat = get_current_chat()

    current_chat["messages"] = st.session_state.messages

    current_chat["pdf_name"] = st.session_state.pdf_name

    st.session_state.all_chats[
        st.session_state.current_chat_id
    ] = current_chat

    save_chat_history(
        st.session_state.all_chats
    )


def start_new_chat():
    """Start a completely new conversation."""

    new_chat = create_new_chat()

    st.session_state.all_chats[
        new_chat["id"]
    ] = new_chat

    st.session_state.current_chat_id = new_chat["id"]

    st.session_state.messages = []

    st.session_state.chat = None

    st.session_state.pdf_name = None


def restore_chat(chat_id):
    """Restore an existing conversation."""

    chat = st.session_state.all_chats[chat_id]

    st.session_state.current_chat_id = chat_id

    st.session_state.messages = chat.get(
        "messages",
        []
    )

    st.session_state.pdf_name = chat.get(
        "pdf_name"
    )

    # RAG chain cannot automatically be restored
    # from saved chat history.
    st.session_state.chat = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, sans-serif;
}

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
        radial-gradient(
            circle at 20% -10%,
            rgba(139,92,246,0.16) 0%,
            transparent 45%
        ),
        radial-gradient(
            circle at 85% 5%,
            rgba(236,72,153,0.10) 0%,
            transparent 40%
        ),
        var(--bg);
}

.block-container {
    max-width: 1050px;
    padding-top: 1.2rem;
    padding-bottom: 5.5rem;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: var(--bg);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Poppins', sans-serif;
    color: var(--text);
}

.stFileUploader section {
    background: var(--panel) !important;
    border: 1.5px dashed rgba(139,92,246,0.55) !important;
    border-radius: 14px !important;
}

/* Hero */

.hero {
    text-align: center;
    padding: 34px 24px 30px 24px;
    margin-bottom: 24px;
    border-radius: 24px;
    background:
        linear-gradient(
            160deg,
            #171233 0%,
            #1a1130 55%,
            #1f1230 100%
        );
    border: 1px solid rgba(139,92,246,0.28);
    box-shadow:
        0 0 0 1px rgba(139,92,246,0.06),
        0 20px 50px rgba(0,0,0,0.45);
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

.title-icon {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    background: linear-gradient(
        135deg,
        #a78bfa,
        #8b5cf6
    );
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 6px 20px rgba(139,92,246,0.45);
}

.main-title {
    font-family: 'Poppins', sans-serif;
    font-size: 46px;
    font-weight: 800;
    margin: 0;
    background:
        linear-gradient(
            90deg,
            #a78bfa 0%,
            #c084fc 45%,
            #ec4899 100%
        );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle {
    color: var(--text-dim);
    font-size: 16.5px;
    margin-top: 12px;
}

/* Cards */

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

/* Sidebar configuration */

.cfg-card {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 14px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
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

/* Chat history */

.history-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 9px 11px;
    margin-bottom: 7px;
}

.history-title {
    color: var(--text);
    font-size: 13px;
    font-weight: 600;
}

.history-date {
    color: var(--text-dim);
    font-size: 10px;
    margin-top: 3px;
}

/* Source */

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

/* Chat */

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
    background: #232236 !important;
    border-radius: 10px !important;
}

div[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(
        135deg,
        #a78bfa,
        #ec4899
    ) !important;
    border-radius: 10px !important;
}

/* Chat input */

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
}

div[data-testid="stChatInput"] button {
    border-radius: 999px !important;
    margin-right: 2px !important;
    flex: 0 0 auto !important;
}

div[data-testid="stChatInput"] textarea:focus,
div[data-testid="stChatInput"] textarea:focus-visible {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}

div[data-testid="stChatInput"]:focus-within {
    outline: none !important;
    border-color: var(--violet) !important;
    box-shadow:
        0 0 0 3px rgba(139,92,246,0.2) !important;
}

/* Buttons */

.stButton button {
    background:
        linear-gradient(
            90deg,
            #8b5cf6,
            #ec4899
        ) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 999px !important;
    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease;
}

.stButton button:hover {
    transform: translateY(-1px);
    box-shadow:
        0 6px 18px rgba(139,92,246,0.4) !important;
}

/* Alerts */

.stAlert {
    border-radius: 12px;
}

/* Footer */

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
    background:
        linear-gradient(
            90deg,
            #a78bfa,
            #ec4899
        );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

/* Scrollbar */

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(139,92,246,0.35);
    border-radius: 4px;
}

</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "🆕 New Chat",
        use_container_width=True
    ):
        start_new_chat()
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    st.markdown("### 🕘 Chat History")

    chats = list(
        st.session_state.all_chats.values()
    )

    chats = sorted(
        chats,
        key=lambda x: x.get("created_at", ""),
        reverse=True
    )

    if not chats:

        st.caption("No saved conversations yet.")

    else:

        for saved_chat in chats:

            chat_id = saved_chat["id"]

            title = saved_chat.get(
                "title",
                "New Chat"
            )

            created_at = saved_chat.get(
                "created_at",
                ""
            )

            col1, col2 = st.columns(
                [5, 1]
            )

            with col1:

                if st.button(
                    f"💬 {title}",
                    key=f"restore_{chat_id}",
                    use_container_width=True
                ):

                    restore_chat(chat_id)

                    st.rerun()

            with col2:

                if st.button(
                    "🗑️",
                    key=f"delete_{chat_id}"
                ):

                    del st.session_state.all_chats[
                        chat_id
                    ]

                    # If deleting current chat,
                    # create a new chat.
                    if (
                        chat_id ==
                        st.session_state.current_chat_id
                    ):

                        new_chat = create_new_chat()

                        st.session_state.all_chats[
                            new_chat["id"]
                        ] = new_chat

                        st.session_state.current_chat_id = (
                            new_chat["id"]
                        )

                        st.session_state.messages = []

                        st.session_state.chat = None

                        st.session_state.pdf_name = None

                    save_chat_history(
                        st.session_state.all_chats
                    )

                    st.rerun()

    st.divider()

    # --------------------------------------------------------
    # UPLOAD PDF
    # --------------------------------------------------------

    st.markdown("### 📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload a PDF document to ask questions about its contents.",
        label_visibility="collapsed"
    )

    st.divider()

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    st.markdown("### ⚙️ Configuration")

    st.markdown(
        """
        <div class="cfg-card">
            <div class="cfg-label">Model</div>
            <div class="cfg-value">
                🦙 Llama 3.1 8B Instant
            </div>
        </div>

        <div class="cfg-card">
            <div class="cfg-label">Embeddings</div>
            <div class="cfg-value">
                🧠 MiniLM-L6-v2
            </div>
        </div>

        <div class="cfg-card">
            <div class="cfg-label">Vector Database</div>
            <div class="cfg-value">
                📦 FAISS
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # --------------------------------------------------------
    # DOWNLOAD CURRENT CHAT
    # --------------------------------------------------------

    current_chat = get_current_chat()

    if st.session_state.messages:

        st.markdown("### 📥 Export Chat")

        chat_text = get_chat_text(
            current_chat
        )

        st.download_button(
            label="📄 Download TXT",
            data=chat_text,
            file_name="ai_pdf_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

        pdf_file = create_pdf(
            current_chat
        )

        if pdf_file:

            st.download_button(
                label="📕 Download PDF",
                data=pdf_file,
                file_name="ai_pdf_chat.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.divider()

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if not GROQ_API_KEY:

        st.warning(
            "⚠️ GROQ_API_KEY not found."
        )

    elif uploaded_file:

        st.success(
            "PDF uploaded successfully!"
        )

    else:

        st.info(
            "Upload a PDF to enable "
            "document-based questions."
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="badge">
            ✨ AI-Powered Assistant
        </div>

        <div class="title-row">

            <div class="title-icon">
                🤖
            </div>

            <div class="main-title">
                AI PDF Chatbot
            </div>

        </div>

        <div class="subtitle">
            Ask general questions or upload a PDF
            for source-backed answers.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PDF PROCESSING
# ============================================================

if uploaded_file:

    os.makedirs(
        "uploaded_pdfs",
        exist_ok=True
    )

    pdf_path = os.path.join(
        "uploaded_pdfs",
        uploaded_file.name
    )

    # Process only when a different PDF is uploaded

    if (
        st.session_state.pdf_name
        != uploaded_file.name
    ):

        with open(
            pdf_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        with st.spinner(
            "📚 Reading PDF and creating knowledge base..."
        ):

            vectorstore = create_vectorstore(
                pdf_path
            )

            st.session_state.chat = (
                create_rag_chain(
                    vectorstore
                )
            )

        st.session_state.pdf_name = (
            uploaded_file.name
        )

        # Attach PDF to current conversation

        current_chat = get_current_chat()

        current_chat["pdf_name"] = (
            uploaded_file.name
        )

        # We don't clear the chat here.
        # This allows the restored conversation
        # to remain visible.

        sync_current_chat()

        st.success(
            "PDF processed successfully!"
        )


# ============================================================
# MODE INDICATOR
# ============================================================

if uploaded_file:

    st.markdown(
        f"""
        <div class="pdf-card">

            <div class="pdf-title">
                📄 PDF Mode Active
            </div>

            <div class="pdf-name">
                {uploaded_file.name}
            </div>

            <div
                class="mode-text"
                style="margin-top:8px;"
            >
                Questions related to this document
                will use Retrieval-Augmented Generation
                (RAG).
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="mode-card">

            <div class="mode-title">
                🌎 General AI Mode
            </div>

            <div class="mode-text">
                Ask anything without uploading a PDF.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CURRENT CHAT
# ============================================================

for message in st.session_state.messages:

    avatar = (
        "🧑"
        if message["role"] == "user"
        else "🤖"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            chips = "".join(
                f"""
                <span class="source-card">
                    📄 Page {page}
                </span>
                """
                for page in message["sources"]
            )

            st.markdown(
                chips,
                unsafe_allow_html=True
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your PDF..."
    if uploaded_file
    else
    "Ask anything..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    if not GROQ_API_KEY:

        st.error(
            "GROQ_API_KEY is missing."
        )

    else:

        # ----------------------------------------------------
        # FIRST QUESTION BECOMES CHAT TITLE
        # ----------------------------------------------------

        current_chat = get_current_chat()

        if not st.session_state.messages:

            current_chat["title"] = (
                generate_chat_title(
                    question
                )
            )

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        user_message = {
            "role": "user",
            "content": question,
            "sources": []
        }

        st.session_state.messages.append(
            user_message
        )

        with st.chat_message(
            "user",
            avatar="🧑"
        ):

            st.markdown(
                question
            )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    # ========================================
                    # PDF / RAG MODE
                    # ========================================

                    if (
                        uploaded_file
                        and st.session_state.chat
                    ):

                        response = (
                            st.session_state.chat(
                                question
                            )
                        )

                        answer = response[
                            "answer"
                        ]

                        sources = []

                        for doc in response.get(
                            "sources",
                            []
                        ):

                            page = (
                                doc.metadata.get(
                                    "page"
                                )
                            )

                            if page is not None:

                                page_number = (
                                    page + 1
                                )

                                if (
                                    page_number
                                    not in sources
                                ):

                                    sources.append(
                                        page_number
                                    )

                        st.markdown(
                            answer
                        )

                        if sources:

                            chips = "".join(
                                f"""
                                <span class="source-card">
                                    📄 Page {p}
                                </span>
                                """
                                for p in sources
                            )

                            st.markdown(
                                chips,
                                unsafe_allow_html=True
                            )

                        assistant_message = {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }

                        st.session_state.messages.append(
                            assistant_message
                        )

                    # ========================================
                    # GENERAL AI MODE
                    # ========================================

                    else:

                        llm = ChatGroq(
                            model="llama-3.1-8b-instant",
                            api_key=GROQ_API_KEY,
                            temperature=0.3
                        )

                        response = llm.invoke(
                            question
                        )

                        answer = response.content

                        st.markdown(
                            answer
                        )

                        assistant_message = {
                            "role": "assistant",
                            "content": answer,
                            "sources": []
                        }

                        st.session_state.messages.append(
                            assistant_message
                        )

                    # ========================================
                    # SAVE CONVERSATION
                    # ========================================

                    sync_current_chat()

                except Exception as e:

                    st.error(
                        f"Something went wrong: {str(e)}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🚀 Built with
        <span>
            Streamlit · LangChain · FAISS · HuggingFace · Groq Llama 3.1
        </span>
    </div>
    """,
    unsafe_allow_html=True
)