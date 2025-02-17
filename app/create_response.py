from search_similar_chunks import search_similar_chunks
from transformers import pipeline

# Load the question-answering model (text generation pipeline)
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def generate_answer(question):
    print("Searching for similar passages...")
    
    # Get the most relevant sentence chunks from the cluster-based search
    similar_passages = search_similar_chunks(question)  # This now uses the cluster-based method
    
    if not similar_passages:
        return "No relevant information found."

    # Join the most similar sentence chunks to form the context for the answer generation
    context = "\n".join(similar_passages)  # Combine the top similar chunks into the context
    
    print("Generating answer...")
    # Use the question-answering model to generate an answer based on the provided context
    response = qa_model(question=question, context=context)

    # Check if the model provided a valid answer
    return response["answer"] if response["answer"].strip() else "The model couldn't generate a clear answer."

# Test the model with an example question
print(generate_answer("What is a school management API?"))
