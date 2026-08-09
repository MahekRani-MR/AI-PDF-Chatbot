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
```

## 📁 Project Structure

```text
AI-PDF-Chatbot/
│
├── app.py
├── requirements.txt
├── Pipfile
├── Pipfile.lock
├── README.md
├── .gitignore
│
├── utils/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── pdf_loader.py
│   ├── rag_chain.py
│   ├── rag_pipeline.py
│   ├── ui.py
│   └── vectorstore.py
│
├── uploaded_pdfs/
└── vectorstore/
```

## 📌 Main Files

- app.py – Main Streamlit application and user interface.
- pdf_loader.py – Loads and extracts content from PDF files.
- embeddings.py – Loads the Hugging Face embedding model.
- vectorstore.py – Splits documents and creates the FAISS vector store.
- rag_pipeline.py – Handles document retrieval and response generation.
- rag_chain.py – Contains the LLM and prompt configuration.
- requirements.txt – Contains the required Python packages.


## ⚙️ Installation

Clone the repository:

``` powersell
git clone https://github.com/MahekRani-MR/AI-PDF-Chatbot.git
``` 

Move into the project directory:

``` powershell
cd AI-PDF-Chatbot
```

Install the required dependencies:

``` powershell
pip install -r requirements.txt
```

## 🔑 API Key Setup

This project uses the Groq API for generating responses.

Create a .env file in the project directory:

```powershell
GROQ_API_KEY=your_groq_api_key
```

Replace your_groq_api_key with your actual Groq API key.

Never upload your .env file or expose your API key publicly.

The .env file is included in .gitignore.

## ▶️ Run the Application

Start the Streamlit application:

``` powershell
streamlit run app.py
```

The application will open in your browser at the local Streamlit URL shown in the terminal.

## 💬 How to Use

- Open the AI PDF Chatbot application.
- Upload a PDF using the sidebar.
- Wait for the document to be processed.
- Enter your question in the chat box.
- The application searches the uploaded document for relevant information.
- Relevant document chunks are retrieved using FAISS.
- The retrieved information is passed to the Groq language model.
- The chatbot generates an answer.
- The relevant source page is displayed with the answer.
- General questions can also be asked without uploading a PDF.

## 🧠 RAG Pipeline

```text
The project follows a Retrieval-Augmented Generation (RAG) approach.

PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Hugging Face Embeddings
     ↓
FAISS Vector Database
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

RAG allows the chatbot to retrieve relevant information from the uploaded document before generating an answer.
```

## 📌 Project Purpose

This project was developed as a practical implementation of Retrieval-Augmented Generation (RAG) using Python, LangChain, FAISS, Hugging Face embeddings, Groq, and Streamlit.

The main purpose is to provide an easy-to-use chatbot that can interact with information contained in PDF documents.

## 🔮 Future Improvements

- Support for multiple PDF documents
- Improved chat history and conversation memory
- Better source and citation display
- Support for additional document formats
- Improved document management
- Further UI customization
- Deployment on Streamlit Cloud
- Improved response speed and retrieval accuracy

## 📄 License

```text
This project is licensed under the MIT License.

See the LICENSE file for more information.
```