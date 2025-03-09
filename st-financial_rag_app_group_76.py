import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import faiss
import re
import numpy as np

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample financial data (replace with real data extraction logic)
documents = [
    "Total Revenue in 2023 was $10M.",
    "Net Profit Margin improved by 15% in Q4 2023.",
    "Operating Expenses decreased by 8% in H1 2023.",
]
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

# FAISS setup
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.numpy())

# BM25 setup
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Guardrail - Input Validation
def validate_query(query):
    banned_keywords = ["attack", "hack", "leak"]
    if any(word in query.lower() for word in banned_keywords):
        return False
    return True

# Guardrail - Output Filtering
def filter_output(response):
    safe_patterns = [r'\$\d+', r'\d+%']  # Match financial values
    if any(re.search(pattern, response) for pattern in safe_patterns):
        return response
    return "Response filtered for quality assurance."

# Search Logic
def search_documents(query):
    if not validate_query(query):
        return "Invalid query detected. Please ask relevant financial questions."

    # BM25 Search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # FAISS Search
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    faiss_scores, faiss_indices = index.search(query_embedding.numpy(), k=3)

    # Combine results
    combined_results = list(zip(documents, bm25_scores))
    sorted_results = sorted(combined_results, key=lambda x: x[1], reverse=True)

    # Select top result
    best_result = sorted_results[0][0] if sorted_results[0][1] > 0 else "No relevant information found."

    # Output filter
    return filter_output(best_result)

# Streamlit UI
st.title("Financial Data Extraction RAG Model")
query = st.text_input("Ask a financial question:")

if st.button("Submit"):
    if query.strip():
        result = search_documents(query)
        st.write("**Answer:**", result)
        st.write("**Confidence Score:**", np.round(np.max(bm25.get_scores(query.split())) / 10, 2))
    else:
        st.warning("Please enter a valid query.")

# Testing Section
st.header("Testing & Validation")
test_questions = [
    "What was the total revenue in 2023?",  # High-confidence
    "Did the net profit margin change in Q3 2023?",  # Low-confidence
    "What is the capital of France?"  # Irrelevant question
]
for q in test_questions:
    st.subheader(f"Test Question: {q}")
    st.write("**Answer:**", search_documents(q))
