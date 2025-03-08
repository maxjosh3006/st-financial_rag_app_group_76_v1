import re
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
from thefuzz import process
from rank_bm25 import BM25Okapi

# Initialize models and data
classification_model = SentenceTransformer('all-MiniLM-L6-v2')
keyword_embeddings = classification_model.encode([
    "revenue", "profit", "expenses", "cash flow", "net income"
])

def calculate_confidence(retrieval_confidence, table_confidence, weighting_factor=0.6):
    return round(min(weighting_factor * retrieval_confidence + (1 - weighting_factor) * table_confidence, 100), 2)

def extract_financial_value(tables, query):
    number_pattern = re.compile(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+[MBK]?)')
    for table in tables:
        for row in table:
            row_text = " ".join(str(cell) for cell in row if cell)
            if process.extractOne(query, [row_text], score_cutoff=70):
                numbers = [cell for cell in row if number_pattern.match(str(cell))]
                if len(numbers) >= 2:
                    return numbers[:2], 90
    return ["No valid financial data found"], 0

def classify_query(query, threshold=0.6):
    query_embedding = classification_model.encode(query)
    similarity_scores = util.cos_sim(query_embedding, keyword_embeddings).squeeze().tolist()
    return "relevant" if max(similarity_scores) >= threshold else "irrelevant"

def retrieve_with_guardrails(query, bm25_threshold=20, faiss_threshold=-0.7):
    retrieved_chunks, retrieval_confidence = multistage_retrieve(query)
    if retrieval_confidence < bm25_threshold or retrieval_confidence < faiss_threshold:
        return ["⚠️ No reliable data found. Try rephrasing your query."], 0
    return retrieved_chunks, retrieval_confidence

def parallel_bm25_scoring(query, k=10):
    with ThreadPoolExecutor() as executor:
        bm25_scores = list(executor.map(lambda x: bm25.get_scores(x.split()), [query]))
    return np.argsort(bm25_scores)[-k:]

def display_financial_data(data):
    if data[0] != "No valid financial data found":
        st.markdown(f"""
        ### 📊 Extracted Financial Data
        - **2023:** {data[0]}
        - **2022:** {data[1]}
        """)
    else:
        st.warning("⚠️ No valid financial data found. Try rephrasing your query.")

# Streamlit UI
st.title("📈 Financial Statement Analysis Tool")
query = st.text_input("Enter your financial query:")

if st.button("Analyze"):
    if classify_query(query) == "irrelevant":
        st.error("❌ Irrelevant query. Please ask about financial data.")
    else:
        retrieved_data, confidence = retrieve_with_guardrails(query)
        extracted_data, extraction_confidence = extract_financial_value(retrieved_data, query)
        final_confidence = calculate_confidence(confidence, extraction_confidence)

        st.success(f"✅ Confidence Score: {final_confidence}%")
        display_financial_data(extracted_data)
