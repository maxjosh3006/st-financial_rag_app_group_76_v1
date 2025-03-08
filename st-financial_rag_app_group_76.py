import streamlit as st
import pdfplumber
import faiss
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from thefuzz import process
from sklearn.preprocessing import MinMaxScaler

# ✅ Load PDF
def load_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def extract_tables_from_pdf(pdf_path):
    extracted_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                extracted_tables.append(table)
    return extracted_tables

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size // 2)]

# ✅ Load Data
pdf_path = "BMW_Finance_NV_Annual_Report_2023.pdf"
pdf_text = load_pdf(pdf_path)
tables = extract_tables_from_pdf(pdf_path)
text_chunks = chunk_text(pdf_text)

# ✅ Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

tokenized_chunks = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# ✅ Enhanced Confidence Scoring
def calculate_confidence(bm25_score, faiss_score):
    return round((0.4 * bm25_score + 0.6 * faiss_score), 2)

# ✅ Multi-Stage Retrieval
def multistage_retrieve(query, k=5, bm25_k=10):
    query_embedding = embedding_model.encode([query])
    bm25_scores = bm25.get_scores(query.split())

    # 🔹 Normalize BM25 Scores
    bm25_scores = MinMaxScaler(feature_range=(0, 100)).fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()
    top_bm25_indices = np.argsort(bm25_scores)[-bm25_k:]

    # 🔹 FAISS Vector Search
    filtered_embeddings = np.array([chunk_embeddings[i] for i in top_bm25_indices])
    faiss_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    faiss_index.add(filtered_embeddings)

    _, faiss_ranks = faiss_index.search(query_embedding, k)
    top_faiss_indices = [top_bm25_indices[i] for i in faiss_ranks[0]]

    # 🔹 Final Scores
    final_scores = {}
    for i in set(top_bm25_indices) | set(top_faiss_indices):
        bm25_score = bm25_scores[i] if i in top_bm25_indices else 0
        faiss_score = -np.linalg.norm(query_embedding - chunk_embeddings[i]) * 100
        final_scores[i] = calculate_confidence(bm25_score, faiss_score)

    top_chunks = sorted(final_scores, key=final_scores.get, reverse=True)[:k]
    return [text_chunks[i] for i in top_chunks], max(final_scores.values())

# ✅ Enhanced Financial Data Extraction
def extract_financial_value(tables, query):
    financial_patterns = r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:million|billion|bn|m|k|USD|€|£)"
    possible_headers = []

    for table in tables:
        for row in table:
            row_text = " ".join(str(cell) for cell in row if cell)
            possible_headers.append(row_text)

    best_match, score = process.extractOne(query, possible_headers, score_cutoff=70) or (None, 0)
    if not best_match:
        return ["No valid financial data found"], 0

    for table in tables:
        for row in table:
            row_text = " ".join(str(cell) for cell in row if cell)
            if best_match in row_text:
                numbers = [cell for cell in row if re.match(financial_patterns, str(cell))]
                if len(numbers) >= 2:
                    return numbers[:2], score

    return ["No valid financial data found"], 0

# ✅ Streamlit UI
st.title("📊 Financial Statement Q&A")
query = st.text_input("Enter your financial question:")

if query:
    retrieved_chunks, retrieval_confidence = multistage_retrieve(query)
    retrieved_text = "\n".join(retrieved_chunks)
    financial_values, table_confidence = extract_financial_value(tables, query)

    final_confidence = calculate_confidence(retrieval_confidence, table_confidence)

    # 🔹 Color-coded Confidence Display
    if final_confidence >= 70:
        st.success(f"**🔍 Confidence Score:** {final_confidence}%")
    elif final_confidence >= 40:
        st.warning(f"**🔍 Confidence Score:** {final_confidence}%")
    else:
        st.error(f"**🔍 Confidence Score:** {final_confidence}%")

    if financial_values and financial_values[0] != "No valid financial data found":
        st.write("### 📊 Extracted Financial Data")
        st.info(f"**2023:** {financial_values[0]}, **2022:** {financial_values[1]}")
    else:
        st.warning("⚠️ No valid financial data found. Try rephrasing your query for better results.")
