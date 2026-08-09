from utils.pdf_loader import load_pdf
from utils.embeddings import get_embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load PDF
documents = load_pdf("uploaded_pdfs/Reflexion.pdf")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} pages.")
print(f"Created {len(chunks)} chunks.")

# Load embedding model
embedding_model = get_embedding_model()

# Create FAISS vector database
vectorstore = FAISS.from_documents(chunks, embedding_model)

print("FAISS vector store created successfully!")

# Test retrieval
results = vectorstore.similarity_search("What is Reflexion?", k=2)

print("\nTop Search Result:\n")
print(results[0].page_content)