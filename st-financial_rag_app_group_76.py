import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple

# Load Embedding Model (Fine-tuned for Finance)
embedding_model = SentenceTransformer("FinBERT")  # Fine-tuned model for finance data

def retrieve_chunks(query: str, documents: List[str], boost_keywords: List[str] = []) -> Tuple[List[str], float]:
    """
    Dynamic context expansion with BM25 boosting.
    """
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.split())

    # Boost keywords for financial relevance
    for keyword in boost_keywords:
        scores += bm25.get_scores(keyword.split())

    # Dynamic context expansion if confidence score is low
    best_chunk_idx = np.argmax(scores)
    confidence_score = scores[best_chunk_idx] / max(scores)

    if confidence_score < 0.3:  # Threshold for low confidence
        return documents, confidence_score  # Return all documents for wider context

    return [documents[best_chunk_idx]], confidence_score


def extract_answer(query: str, retrieved_text: str) -> str:
    """
    Extracts answer using embeddings for improved precision.
    """
    embedded_query = embedding_model.encode(query)
    embedded_chunks = embedding_model.encode([retrieved_text])

    similarity = np.dot(embedded_query, embedded_chunks.T) / (np.linalg.norm(embedded_query) * np.linalg.norm(embedded_chunks))
    return retrieved_text if similarity > 0.5 else "Answer not confidently found."


def guardrails(response: str, confidence_score: float) -> str:
    """
    Guardrails to ensure output reliability.
    """
    if confidence_score < 0.3:
        return "Confidence too low. Please refine your query or expand the search scope."
    return response

# Streamlit UI
st.title("Financial Statement Q&A")
query = st.text_input("Enter your financial question:")

if query:
    # Sample Financial Data (replace with your data)
    documents = [
        "Total Receivables from BMW Group companies: EUR 2,345,678 as of 2023.",
        "Net Income for 2023 was EUR 1,234,567.",
        "Total Revenue reached EUR 5,678,910 in 2023.",
    ]

    # Retrieve and Extract Answer
    retrieved_chunks, score = retrieve_chunks(query, documents, boost_keywords=["Receivables", "Net Income", "BMW Group"])
    retrieved_text = "\n".join(retrieved_chunks)
    answer = extract_answer(query, retrieved_text)

    # Apply Guardrails
    final_response = guardrails(answer, score)

    st.write(f"🔍 **Confidence Score:** {score * 100:.2f}%")
    st.write(f"📝 **Answer:** {final_response}")
