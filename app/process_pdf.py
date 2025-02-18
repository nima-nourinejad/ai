# import fitz  # PyMuPDF for PDF processing
# import nltk
# from nltk.tokenize import sent_tokenize
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# import faiss
# import numpy as np
# import pickle

# # Download NLTK tokenizer data
# nltk.download("punkt")

# # Load the embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def extract_sentence_chunks_from_pdf(pdf_path, chunk_size=3):
#     """Extracts sentence chunks from the PDF"""
#     doc = fitz.open(pdf_path)
#     text = " ".join(page.get_text("text") for page in doc)
#     sentences = sent_tokenize(text)
    
#     # Group sentences into chunks of 'chunk_size' (default: 3)
#     sentence_chunks = [
#         " ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)
#     ]
#     return sentence_chunks

# def create_and_store_embeddings(pdf_path, db_path, text_path, num_clusters=5):
#     """Creates embeddings for sentence chunks and stores them along with cluster assignments."""
#     sentence_chunks = extract_sentence_chunks_from_pdf(pdf_path, chunk_size=3)
    
#     if not sentence_chunks:
#         print("❌ No text found in the PDF.")
#         return

#     # Convert sentence chunks to embeddings
#     embeddings = model.encode(sentence_chunks, convert_to_numpy=True)

#     # Perform K-means clustering on embeddings
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0)
#     cluster_assignments = kmeans.fit_predict(embeddings)

#     # Create FAISS index for fast search
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity comparison
#     index.add(embeddings)

#     # Save the FAISS index to disk
#     faiss.write_index(index, db_path)

#     # Save the sentence chunks and cluster assignments to disk
#     with open(text_path, "wb") as f:
#         pickle.dump([sentence_chunks, cluster_assignments], f)

#     print(f"✅ Embeddings, clusters, and FAISS index saved to {db_path} and {text_path}.")

# # Call the function to process PDF and save embeddings, clusters
# create_and_store_embeddings("Nima_Nourinejad_Cover_Letter.pdf", "embeddings.index", "chunks.pkl", num_clusters=5)


import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from PyPDF2 import PdfReader

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to read and extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def chunk_text(text, chunk_size=500):
    """Split text into chunks of the specified size."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# Function to embed text chunks and index them using FAISS
def process_pdf(pdf_path):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Split the text into chunks
    chunks = chunk_text(text)

    # Encode the chunks into embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)
    
    # Create a FAISS index for the embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Use L2 distance for similarity
    index.add(embeddings)

    # Perform DBSCAN clustering to assign clusters to the chunks
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(embeddings)
    cluster_assignments = clustering.labels_

    # Save the index, clusters, and chunks for future use
    faiss.write_index(index, "embeddings.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump((chunks, cluster_assignments), f)

# Run the processing on a PDF (you can replace 'your_pdf_file.pdf' with your actual file path)
process_pdf('Nima_Nourinejad_Cover_Letter.pdf')
