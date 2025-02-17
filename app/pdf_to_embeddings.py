import fitz  # pymupdf (for PDF processing)
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle  # To store original text data

# Download NLTK tokenizer data
nltk.download("punkt")

# 1. Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Extract 3-sentence chunks from a PDF file
def extract_sentence_chunks_from_pdf(pdf_path, chunk_size=3):
    doc = fitz.open(pdf_path)
    
    # Extract text from all pages
    text = " ".join(page.get_text("text") for page in doc)
    
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Group sentences into chunks of 'chunk_size' (default: 3)
    sentence_chunks = [
        " ".join(sentences[i : i + chunk_size])  # Combine every 3 sentences
        for i in range(0, len(sentences), chunk_size)
    ]

    return sentence_chunks

# 3. Convert sentence chunks to embeddings and store in FAISS
def create_and_store_embeddings(pdf_path, db_path, text_path):
    sentence_chunks = extract_sentence_chunks_from_pdf(pdf_path, chunk_size=3)
    
    if not sentence_chunks:
        print("❌ No text found in the PDF file.")
        return

    # Print chunks for debugging
    # for i, chunk in enumerate(sentence_chunks, 1):
    #     print(f"Chunk {i}:")
    #     print(chunk)
    #     print("-----")

    # Convert chunks to embeddings
    embeddings = model.encode(sentence_chunks, convert_to_numpy=True)

    # 4. Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity comparison
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, db_path)

    # Save sentence chunks linked to embeddings
    with open(text_path, "wb") as f:
        pickle.dump(sentence_chunks, f)

    print(f"✅ Sentence chunk embeddings saved to {db_path}")
    print(f"✅ Sentence chunk text data saved to {text_path}")

# Run preprocessing step
create_and_store_embeddings("Nima_Nourinejad_Cover_Letter.pdf", "embeddings.index", "chunks.pkl")
