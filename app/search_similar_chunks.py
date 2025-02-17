import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS Index
index = faiss.read_index("embeddings.index")

# Load stored sentence chunks and cluster assignments
with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)
    sentence_chunks = data[0]  # First element: chunks
    cluster_assignments = np.array(data[1])  # Second element: clusters

def search_similar_chunks(question, top_k=5):
    """Searches for the most similar sentence chunks to the provided question."""
    # Convert the query to an embedding
    question_embedding = model.encode([question], convert_to_numpy=True)
    
    # Search the FAISS index for the most similar embeddings
    _, nearest_chunk_indices = index.search(question_embedding, top_k)
    
    # Retrieve the corresponding sentence chunks
    top_chunks = [sentence_chunks[i] for i in nearest_chunk_indices[0] if i < len(sentence_chunks)]
    
    # Get the clusters of the top retrieved chunks
    top_chunk_indices = nearest_chunk_indices[0]
    top_clusters = cluster_assignments[top_chunk_indices]
    
    # Group chunks by their clusters (optional: this step depends on how you want to refine the search)
    cluster_to_chunks = {}
    for idx, cluster in zip(top_chunk_indices, top_clusters):
        if cluster not in cluster_to_chunks:
            cluster_to_chunks[cluster] = []
        cluster_to_chunks[cluster].append(sentence_chunks[idx])
    
    return top_chunks, cluster_to_chunks

