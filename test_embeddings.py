from utils.pdf_loader import load_pdf
from utils.embeddings import get_embedding_model

documents = load_pdf("uploaded_pdfs/Reflexion.pdf")

embedding_model = get_embedding_model()

print("Embedding model loaded successfully!")
print(f"Loaded {len(documents)} pages.")