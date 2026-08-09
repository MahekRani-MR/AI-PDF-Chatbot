from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """
    Loads a PDF and returns a list of LangChain Documents.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents