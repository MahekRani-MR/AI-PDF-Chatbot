from utils.pdf_loader import load_pdf
from utils.embeddings import get_embedding_model
from utils.vectorstore import create_vectorstore
from utils.rag_pipeline import create_rag_chain

pdf_path = "uploaded_pdfs/Reflexion.pdf"

documents = load_pdf(pdf_path)

embedding_model = get_embedding_model()

vectorstore = create_vectorstore(
    documents,
    embedding_model
)

chat = create_rag_chain(vectorstore)

response = chat("What is Reflexion?")

print("\nAnswer:\n")
print(response["answer"])

print("\nSource Page:")
print(response["sources"][0].metadata["page"] + 1)