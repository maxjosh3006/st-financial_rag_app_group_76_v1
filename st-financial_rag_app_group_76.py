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

# ✅ Enhanced Table Extraction with Improved Parsing
def extract_tables_from_pdf(pdf_path):
    extracted_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                extracted_tables.append(table)  # Store all tables
    return extracted_tables


def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)]


# ✅ Load Data
pdf_path = "BMW_Finance_NV_Annual_Report_2023.pdf"
pdf_text = load_pdf(pdf_path)
tables = extract_tables_from_pdf(pdf_path)
text_chunks = chunk_text(pdf_text)

# ✅ Embedding Model
# ✅ Step 4: Set Up Multi-Stage Retrieval (BM25 + FAISS)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

tokenized_chunks = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# ✅ Improved Multi-Stage Retrieval with Better Confidence Calculation
# ✅ Multi-Stage Retrieval with Improved Scoring

import math

def multistage_retrieve(query, k=5, bm25_k=20, alpha=0.7): 
    query_embedding = embedding_model.encode([query])
    bm25_scores = bm25.get_scores(query.split())
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
        if math.isnan(retrieval_confidence):
            retrieval_confidence = 0.0
    else:
        top_chunks = []
        retrieval_confidence = 0.0

    valid_chunks = [i for i in top_chunks if i < len(text_chunks)]
    return [text_chunks[i] for i in valid_chunks], round(retrieval_confidence, 2)


# ✅ Step 6: Retrieve Financial Values from Tables


# ✅ Improved Financial Data Extraction with Flexible Matching
def extract_financial_value(tables, query):
    possible_headers = [
        " ".join(str(cell).strip().lower() for cell in row if cell)
        for table in tables
        for row in table
        if any(cell for cell in row)
    ]

    extraction_result = process.extractOne(query.lower(), possible_headers, score_cutoff=50)  # Lowered cutoff

    if extraction_result:
        best_match, score = extraction_result
    else:
        return ["No valid financial data found"], 0

    for table in tables:
        for row in table:
            row_text = " ".join(str(cell).strip().lower() for cell in row if cell)
            if best_match in row_text:
                numbers = [
                    cell for cell in row
                    if re.match(r"\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?", str(cell))
                ]
                if len(numbers) >= 2:
                    return numbers[:2], round(score, 2)

    return ["No valid financial data found"], 0

# ✅ Irrelevant Query Handling

# Load the embedding model (same as used for FAISS)
classification_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define keywords for known financial topics
relevant_keywords = [
    "revenue", "profit", "expenses", "income", "assets", "liabilities", "equity", "earnings",
    "financial performance", "cash flow", "balance sheet", "receivables", "accounts receivable",
    "trade receivables", "total receivables"
]

def classify_query(query, threshold=0.4):  # Lowered threshold for flexible matching
    query_embedding = classification_model.encode(query)
    similarity_scores = util.cos_sim(query_embedding, keyword_embeddings).squeeze().tolist()

    if max(similarity_scores) >= threshold:
        return "relevant"
    return "irrelevant"

# Encode relevant keywords for similarity checks
keyword_embeddings = classification_model.encode(relevant_keywords)

scaler = MinMaxScaler(feature_range=(0, 100))
query = st.text_input("Enter your financial question:", key="financial_query")

def calculate_confidence(retrieval_confidence, table_confidence, weight=0.6):
    return round((weight * retrieval_confidence) + ((1 - weight) * table_confidence), 2)

financial_values, table_confidence = extract_financial_value(tables, query)

# Ensure table_confidence is a valid number
try:
    table_confidence = float(table_confidence)
except (ValueError, TypeError):
    table_confidence = 0.0  # Default to 0 if invalid

final_confidence = calculate_confidence(retrieval_confidence, table_confidence)


# ✅ Streamlit UI
st.title("📊 Financial Statement Q&A")
#query = st.text_input("Enter your financial question:")
result = multistage_retrieve(query)
print(f"DEBUG: multistage_retrieve() Result - {result}")

if result and len(result) == 2:
    retrieved_chunks, retrieval_confidence = result
else:
    retrieved_chunks, retrieval_confidence = [], 0.0

try:
    retrieval_confidence = float(retrieval_confidence)
except (ValueError, TypeError):
    retrieval_confidence = 0.0
if query:
    query_type = classify_query(query)  # 🔹 Classify the query first

    if query_type == "irrelevant":
        st.warning("⚠️ This appears to be an irrelevant question.")
        st.write("**🔍 Confidence Score:** 0%")
    else:
        # Proceed with retrieval if query is relevant
        retrieved_chunks = multistage_retrieve(query)
        print(f"Type of retrieved_chunks: {type(retrieved_chunks)}")
        print(f"Content of retrieved_chunks: {retrieved_chunks}")
        if retrieved_chunks and isinstance(retrieved_chunks, list):
           retrieved_text = "\n".join(retrieved_chunks)
        else:
           retrieved_text = "No relevant data found or retrieval error occurred."

        financial_values, table_confidence = extract_financial_value(tables, query)
        print (financial_values)

         # Improved Confidence Calculation
        final_confidence = calculate_confidence(retrieval_confidence, table_confidence)

        # Show confidence scores separately
        st.write("### ✅ Retrieved Context")
        st.success(retrieved_text)
        st.write(f"### 🔍 Final Confidence Score: {final_confidence}%")

        if financial_values and financial_values[0] != "No valid financial data found":
            st.write("### 📊 Extracted Financial Data")
            st.info(f"**2023:** {financial_values[0]}, **2022:** {financial_values[1]}")
        else:
            st.warning("⚠️ No valid financial data found. Try rephrasing your query for better results.")

# ✅ Testing & Validation - Triggered by Button for Cleaner UI
if st.sidebar.button("Run Test Queries"):
    st.sidebar.header("🔍 Testing & Validation")

    test_queries = [
        ("Total Receivables from BMW Group companies", "High Confidence"),
        ("Net Income" , "Low Confidence"),
        ("What is the capital of France?", "Irrelevant")
    ]

    for test_query, confidence_level in test_queries:
        query_type = classify_query(test_query)

        if query_type == "irrelevant":
            st.sidebar.write(f"**🔹 Query:** {test_query} (❌ Irrelevant)")
            st.sidebar.write("**🔍 Confidence Score:** 0%")
            continue  # Skip retrieval steps for irrelevant queries

        retrieved_chunks, retrieval_confidence = multistage_retrieve(test_query)
        retrieved_text = "\n".join(retrieved_chunks)
        financial_values, table_confidence = extract_financial_value(tables, test_query)

        #final_confidence = round((retrieval_confidence + table_confidence) / 2, 2)
        final_confidence = calculate_confidence(retrieval_confidence, table_confidence)

        st.sidebar.write(f"**🔹 Query:** {test_query}")
        st.sidebar.write(f"**🔍 Confidence Score:** {final_confidence}%")
