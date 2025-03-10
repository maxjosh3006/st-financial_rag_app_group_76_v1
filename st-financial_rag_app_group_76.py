import os
import re
import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Data Preprocessing
def preprocess_text(text):
    # Clean and chunk data
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = [text[i:i+300] for i in range(0, len(text), 300)]
    return chunks

# Embedding and Vector Storage
def embed_text(chunks):
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Multi-Stage Retrieval (BM25 + Embedding Search)
def retrieve_results(query, index, text_chunks):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    _, retrieved_indices = index.search(query_embedding, 5)

    # BM25 for keyword search
    tokenized_chunks = [chunk.split() for chunk in text_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(query.split())

    # Combine scores with re-ranking (simple merge logic)
    results = sorted(
        enumerate(zip(bm25_scores, retrieved_indices[0])),
        key=lambda x: x[1][0] + x[1][1],
        reverse=True
    )
    return [text_chunks[idx] for idx, _ in results[:3]]

# Guardrail Implementation (Input Filtering)
def filter_query(query):
    if any(word in query.lower() for word in ["capital of france", "weather"]):
        return "This query seems irrelevant to financial data. Please ask a relevant financial question."
    return query

# Streamlit UI
def main():
    st.title("BMW Finance N.V. RAG Model")
    user_query = st.text_input("Enter your financial query:")

    if user_query:
        filtered_query = filter_query(user_query)
        if "irrelevant" in filtered_query:
            st.error(filtered_query)
        else:
            text_chunks = preprocess_text(open("BMW_Finance_NV_Annual_Report_2023.txt").read())
            embeddings = embed_text(text_chunks)
            index = create_faiss_index(embeddings)
            results = retrieve_results(filtered_query, index, text_chunks)
            st.write("### Results")
            for result in results:
                st.markdown(f"> {result}")

if __name__ == "__main__":
    main()
