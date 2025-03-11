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

# ✅ Extract Tables
def extract_tables_from_pdf(pdf_path):
    extracted_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                extracted_tables.append(table)
    return extracted_tables

# ✅ Chunk Text
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)]

# ✅ Load Data
pdf_path = "BMW_Finance_NV_Annual_Report_2023.pdf"
pdf_text = load_pdf(pdf_path)
tables = extract_tables_from_pdf(pdf_path)
text_chunks = chunk_text(pdf_text)

# ✅ Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

# ✅ Initialize FAISS
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# ✅ Initialize BM25
tokenized_chunks = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# ✅ Multi-Stage Retrieval with Normalized Confidence Scores
def multistage_retrieve(query, k=5, bm25_k=20, alpha=0.7): 
    query_embedding = embedding_model.encode([query])
    bm25_scores = bm25.get_scores(query.split())
    
    # Normalize BM25 Scores
    bm25_scores = np.array(bm25_scores)
    if len(bm25_scores) > 0:
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9) * 100

    top_bm25_indices = np.argsort(bm25_scores)[-bm25_k:]

    filtered_embeddings = np.array([chunk_embeddings[i] for i in top_bm25_indices])
    faiss_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
    faiss_index.add(filtered_embeddings)

    _, faiss_ranks = faiss_index.search(query_embedding, k)
    top_faiss_indices = [top_bm25_indices[i] for i in faiss_ranks[0]]

    final_scores = {}
    for i in set(top_bm25_indices) | set(top_faiss_indices):
        bm25_score = bm25_scores[i] if i in top_bm25_indices else 0
        faiss_score = np.dot(query_embedding, chunk_embeddings[i])
        final_scores[i] = alpha * bm25_score + (1 - alpha) * faiss_score

    if final_scores:
        top_chunks = sorted(final_scores, key=final_scores.get, reverse=True)[:k]
        retrieval_confidence = float(max(final_scores.values()))  # Ensure float value
        if np.isnan(retrieval_confidence):
            retrieval_confidence = 0.0
    else:
        top_chunks = []
        retrieval_confidence = 0.0

    valid_chunks = [i for i in top_chunks if i < len(text_chunks)]
    return [text_chunks[i] for i in valid_chunks], round(retrieval_confidence, 2)


# ✅ Improved Financial Data Extraction with Confidence Scaling

def extract_financial_value(tables, query):
    possible_headers = [
        " ".join(str(cell).strip().lower() for cell in row if cell)
        for table in tables
        for row in table
        if any(cell for cell in row)
    ]

    extraction_result = process.extractOne(query.lower(), possible_headers, score_cutoff=50)

    if extraction_result:
        best_match, score = extraction_result
    else:
        return ["No valid financial data found"], 0

    extracted_numbers = []
    for table in tables:
        for row in table:
            row_text = " ".join(str(cell).strip().lower() for cell in row if cell)
            if best_match in row_text:
                numbers = [cell for cell in row if re.match(r"\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?", str(cell))]
                extracted_numbers.extend(numbers)

    if len(extracted_numbers) >= 2:
        extracted_confidence = round(score * (len(extracted_numbers) / 5), 2)  # Adjust scaling
        return extracted_numbers[:2], extracted_confidence

    return ["No valid financial data found"], 0


# ✅ Query Classification for Irrelevant Queries
classification_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
relevant_keywords = ["revenue", "profit", "expenses", "income", "assets", "liabilities", "equity", 
                     "earnings", "financial performance", "cash flow", "balance sheet", "receivables", 
                     "accounts receivable", "trade receivables", "total receivables"]

keyword_embeddings = classification_model.encode(relevant_keywords)

def classify_query(query, threshold=0.4):
    query_embedding = classification_model.encode(query)
    similarity_scores = util.cos_sim(query_embedding, keyword_embeddings).squeeze().tolist()
    return "relevant" if max(similarity_scores) >= threshold else "irrelevant"

from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler(feature_range=(0, 100))

def calculate_confidence(retrieval_confidence, table_confidence, weight=0.6):
    confidence_values = np.array([[retrieval_confidence, table_confidence]])
    scaled_confidences = scaler.fit_transform(confidence_values)
    
    retrieval_scaled = scaled_confidences[0, 0]
    table_scaled = scaled_confidences[0, 1]

    final_score = (weight * retrieval_scaled) + ((1 - weight) * table_scaled)
    return round(final_score, 2)

# ✅ Streamlit UI
st.title("📊 Financial Statement Q&A")
query = st.text_input("Enter your financial question:", key="financial_query")

if query:
    query_type = classify_query(query)

    if query_type == "irrelevant":
        st.warning("⚠️ This appears to be an irrelevant question.")
        st.write("**🔍 Confidence Score:** 0%")
    else:
        retrieved_chunks, retrieval_confidence = multistage_retrieve(query)
        retrieved_text = "\n".join(retrieved_chunks) if retrieved_chunks else "No relevant data found."

        financial_values, table_confidence = extract_financial_value(tables, query)

        retrieval_confidence = float(retrieval_confidence) if retrieval_confidence else 0.0
        table_confidence = float(table_confidence) if table_confidence else 0.0

        #final_confidence = round((0.6 * retrieval_confidence) + (0.4 * table_confidence), 2)
         final_confidence = calculate_confidence(retrieval_confidence, table_confidence)
        st.write("### ✅ Retrieved Context")
        st.success(retrieved_text)
        st.write(f"### 🔍 Final Confidence Score: {final_confidence}%")

        #if financial_values and financial_values[0] != "No valid financial data found":
            #st.write("### 📊 Extracted Financial Data")
            #st.info(f"**2023:** {financial_values[0]}, **2022:** {financial_values[1]}")
        #else:
            #st.warning("⚠️ No valid financial data found. Try rephrasing your query.")

# ✅ Testing & Validation
if st.sidebar.button("Run Test Queries"):
    st.sidebar.header("🔍 Testing & Validation")

    test_queries = [
        ("Total Receivables from BMW Group companies", "High Confidence"),
        ("Net Income", "Low Confidence"),
        ("What is the capital of France?", "Irrelevant")
    ]

    for test_query, confidence_level in test_queries:
        query_type = classify_query(test_query)

        if query_type == "irrelevant":
            st.sidebar.write(f"**🔹 Query:** {test_query} (❌ Irrelevant)")
            st.sidebar.write("**🔍 Confidence Score:** 0%")
            st.sidebar.write("⚠️ No relevant financial data available.")
            continue

        retrieved_chunks, retrieval_confidence = multistage_retrieve(test_query)
        if retrieved_chunks and isinstance(retrieved_chunks, list):
            retrieved_text = "\n".join(retrieved_chunks)
        else:
            retrieved_text = "No relevant data found or retrieval error occurred."
             # 🔹 Extract financial values from tables
        financial_values, table_confidence = extract_financial_value(tables, test_query)
             # 🔹 Calculate final confidence
        final_confidence = round((0.6 * retrieval_confidence) + (0.4 * table_confidence), 2)
             # 🔹 Display results
        st.sidebar.write(f"**🔹 Query:** {test_query}")
        st.sidebar.write(f"**🔍 Confidence Score:** {final_confidence}%")
        if retrieved_text.strip():
            st.sidebar.write("### ✅ Retrieved Context")
            st.sidebar.success(retrieved_text)
        else:
            st.sidebar.warning("⚠️ No relevant financial context retrieved.")
