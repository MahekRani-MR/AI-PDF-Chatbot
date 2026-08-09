from utils.pdf_loader import load_pdf

docs = load_pdf("uploaded_pdfs/Reflexion.pdf")

print(f"Total Pages: {len(docs)}")

print("\nFirst Page:\n")

print(docs[0].page_content[:1000])