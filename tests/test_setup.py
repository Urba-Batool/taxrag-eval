import os, shutil
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pytesseract
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()

print("Tesseract on PATH:", shutil.which("tesseract"))
print("API key loaded:", bool(os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")))

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
vec = model.encode(["sales tax on goods"])
print("Embedding shape:", vec.shape)  # expect (1, 384)

index = faiss.IndexFlatIP(vec.shape[1])
index.add(vec)
print("FAISS vectors:", index.ntotal)  # expect 1

pdfs = [f for f in os.listdir("data/raw") if f.endswith(".pdf")]
print("PDFs found:", pdfs)
if pdfs:
    doc = fitz.open(os.path.join("data/raw", pdfs[0]))
    print("First PDF pages:", len(doc))
    print("Sample text:", doc[0].get_text()[:200])