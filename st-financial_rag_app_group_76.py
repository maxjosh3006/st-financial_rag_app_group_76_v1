import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import re

# Load Models
bm25_tokenizer = lambda text: text.lower().split()
bm25_corpus = []  # Preprocess and populate with your financial data
bm25 = BM25Okapi([bm25_tokenizer(doc) for doc in bm25_corpus])

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# FAISS Index
dimension = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)
doc_embeddings = embedding_model.encode(bm25_corpus, convert_to_tensor=True)
index.add(np.array(doc_embeddings))

# Guardrail Functions
def validate_input(query):
    if not re.match(r"^[a-zA-Z0-9 ?!.,']+$", query):
        return "Invalid query. Please enter a valid financial question."
    return query

def filter_output(response):
    if "France" in response:  # Example filter for irrelevant content
        return "This question is irrelevant to financial data."
    return response

# Multi-Stage Retrieval Pipeline
def multi_stage_retrieval(query):
    query_validated = validate_input(query)
    if "Invalid" in query_validated:
        return query_validated

    # Stage 1: BM25 Search
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:10]

    # Stage 2: Embedding Search
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    _, top_faiss_indices = index.search(query_embedding.unsqueeze(0).numpy(), 10)

    # Combine Results
    combined_results = set(top_bm25_indices) | set(top_faiss_indices[0])
    candidate_docs = [bm25_corpus[idx] for idx in combined_results]

    # Stage 3: Cross-Encoder Re-ranking
    ranked_docs = sorted(candidate_docs, key=lambda doc: cross_encoder.predict((query, doc)), reverse=True)

    return ranked_docs[0] if ranked_docs else "No relevant content found."

# Streamlit UI Development
st.title("Financial Data Retrieval System")
user_query = st.text_input("Enter your financial query:")
if user_query:
    result = multi_stage_retrieval(user_query)
    filtered_response = filter_output(result)
    st.write("**Answer:**", filtered_response)

# Testing
test_queries = [
    "What is BMW's net profit in 2023?",
    "Summarize BMW's cash flow statement trends.",
    "What is the capital of France?"
]

st.sidebar.title("Test Cases")
for query in test_queries:
    st.sidebar.write(f"**Query:** {query}")
    st.sidebar.write(f"**Answer:** {multi_stage_retrieval(query)}")
