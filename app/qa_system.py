# from search_similar_chunks import search_similar_chunks
# from transformers import pipeline

# # Load a more powerful question-answering model (BART-large fine-tuned for question-answering)
# qa_model = pipeline("question-answering", model="facebook/bart-large")

# def generate_answer(question):
#     """Generates an answer to the question using the search and QA model."""
#     print("Searching for similar passages...")

#     # Get the most relevant sentence chunks using the cluster-based search
#     similar_passages, _ = search_similar_chunks(question, top_k=5)  # Increase top_k for more context

#     if not similar_passages:
#         return "No relevant information found."

#     # Join the most similar sentence chunks to form the context for the answer generation
#     context = "\n".join(similar_passages)  # Combine the top similar chunks into the context
#     # print("Context for the question:")
#     # print(context)  # Debugging context content
#     print("Generating response...")

#     # Use the question-answering model to generate an answer based on the provided context
#     response = qa_model(question=question, context=context)

#     # Return the answer from the model, or a fallback if the answer is empty
#     return response["answer"] if response["answer"].strip() else "The model couldn't generate a clear answer."

# # Test the model with an example question
# print(generate_answer("School Management API"))

from search_similar_chunks import search_similar_chunks
from transformers import pipeline

# Load the question-answering model (DistilBERT fine-tuned for question-answering)
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def generate_answer(question):
    """Generates an answer to the question using the search and QA model."""
    print("Searching for similar passages...")

    # Get the most relevant sentence chunks using the cluster-based search
    similar_passages, _ = search_similar_chunks(question)  # This uses the cluster-based method

    if not similar_passages:
        return "No relevant information found."

    # Join the most similar sentence chunks to form the context for the answer generation
    context = "\n".join(similar_passages)  # Combine the top similar chunks into the context
    
    print("Generating answer...")
    # Use the question-answering model to generate an answer based on the provided context
    response = qa_model(question=question, context=context)

    # Return the answer from the model, or a fallback if the answer is empty
    return response["answer"] if response["answer"].strip() else "The model couldn't generate a clear answer."

# Test the model with an example question
print(generate_answer("What is the database used in book haven?"))
