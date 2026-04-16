import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from config import DOCUMENTS_DIR, CHUNK_SIZE

def extract_pdf_to_txt(pdf_path, output_dir=DOCUMENTS_DIR):
    """
    Extract text from a PDF and split it into chunks.
    Supports both text PDFs and scanned PDFs (using OCR).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_text = ""

    # --- Step 1: Try normal text extraction ---
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"
    except Exception as e:
        print(f"⚠️ PyPDF2 failed to extract text: {e}")

    # --- Step 2: Fallback to OCR if no text found ---
    if not all_text.strip():
        print("⚠️ No text found, using OCR...")
        try:
            images = convert_from_path(pdf_path)
            for img in images:
                text = pytesseract.image_to_string(img)
                if text.strip():
                    all_text += text + "\n\n"
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")

    if not all_text.strip():
        print("❌ No text could be extracted from the PDF.")
        return

    # --- Step 3: Split text into chunks ---
    words = all_text.split()
    for i in range(0, len(words), CHUNK_SIZE):
        chunk_text = " ".join(words[i:i + CHUNK_SIZE])
        output_file = os.path.join(output_dir, f"chunk_{i // CHUNK_SIZE + 1}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(chunk_text)

    print(f"✅ Extracted {len(words) // CHUNK_SIZE + 1} text chunks to {output_dir}/")
  

def extract_pdf_to_chunks(pdf_path, chunk_size=500):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # Simple chunking
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

