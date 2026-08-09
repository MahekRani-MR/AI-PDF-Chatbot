# 🤖 AI PDF Chatbot

AI PDF Chatbot is a simple RAG-based application that allows users to upload PDF documents and ask questions about their content.

The chatbot uses **FAISS** for vector search, **Hugging Face embeddings** for converting text into vectors, and **Groq** for generating answers.

## 🚀 Features

- 📄 Upload and process PDF documents
- 🔍 Semantic search using FAISS
- 🧠 Hugging Face sentence embeddings
- 💬 Ask questions about uploaded PDFs
- 🤖 AI-generated answers using Groq
- 📚 Displays the source page for answers
- 💡 Supports general questions even when no PDF is uploaded
- 🖥️ Simple and interactive Streamlit interface

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- LangChain Hugging Face
- LangChain Groq
- FAISS
- Sentence Transformers
- PyPDF
- Groq API

## 🔄 How It Works

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Splitting
    ↓
Hugging Face Embeddings
    ↓
FAISS Vector Store
    ↓
User Question
    ↓
Similarity Search
    ↓
Relevant Document Chunks
    ↓
Groq LLM
    ↓
Generated Answer