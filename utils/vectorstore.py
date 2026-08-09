from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.pdf_loader import load_pdf
from utils.embeddings import get_embedding_model


def create_vectorstore(pdf_path: str):
    documents = load_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore