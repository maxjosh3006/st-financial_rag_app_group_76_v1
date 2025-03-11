import streamlit as st
import pdfplumber
import faiss
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from thefuzz import process
from sklearn.preprocessing import MinMaxScaler
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

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

# ✅ Extract Precise Sentences
def extract_relevant_sentences(retrieved_chunks, query, max_sentences=3):
    sentences = []
    for chunk in retrieved_chunks:
        if not chunk:  # 🔹 Skip empty chunks to prevent errors
            continue
        chunk_sentences = sent_tokenize(chunk)
        for sentence in chunk_sentences:
            if any(word.lower() in sentence.lower() for word in query.split()):
                sentences.append(sentence)
    return " ".join(sentences[:max_sentences]) if sentences else "No precise data found."

# ✅ Multi-Stage Retrieval
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
        retrieval_confidence = float(max(final_scores.values()))
    else:
        top_chunks = []
        retrieval_confidence = 0.0  # Default confidence

    valid_chunks = [i for i in top_chunks if i < len(text_chunks)]
    retrieved_chunks = [text_chunks[i] for i in valid_chunks] if valid_chunks else []

    # Ensure we always return a valid string
    precise_context = extract_relevant_sentences(retrieved_chunks, query) if retrieved_chunks else "No relevant data found."

    return precise_context, round(retrieval_confidence, 2)

# ✅ Query Classification
classification_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
relevant_keywords = ["revenue", "profit", "expenses", "income", "assets", "liabilities", "equity", 
                     "earnings", "financial performance", "cash flow", "balance sheet", "receivables", 
                     "accounts receivable", "trade receivables", "total receivables"]

keyword_embeddings = classification_model.encode(relevant_keywords)

def classify_query(query, threshold=0.4):
    query_embedding = classification_model.encode(query)
    similarity_scores = util.cos_sim(query_embedding, keyword_embeddings).squeeze().tolist()
    return "relevant" if max(similarity_scores) >= threshold else "irrelevant"

scaler = MinMaxScaler(feature_range=(0, 100))

def calculate_confidence(retrieval_confidence):
    return round(scaler.fit_transform(np.array([[retrieval_confidence]]))[0, 0], 2)

# ✅ Streamlit UI
st.title("📊 Financial Statement Q&A")
query = st.text_input("Enter your financial question:", key="financial_query")

if query:
    query_type = classify_query(query)

    if query_type == "irrelevant":
        st.warning("❌ This appears to be an irrelevant question.")
        st.write("**🔍 Confidence Score:** 0%")
    else:
        if test_query.strip():
           retrieved_text, retrieval_confidence = multistage_retrieve(test_query)
        else:
           retrieved_text, retrieval_confidence = "No query provided.", 0.0

        final_confidence = calculate_confidence(retrieval_confidence)

        st.write(f"### 🔍 Confidence Score: {final_confidence}%")
        
        if final_confidence >= 50:  # High confidence
            st.success(f"**✅ Relevant Information:**\n\n {retrieved_text}")
        else:  # Low confidence
            st.warning(f"⚠️ **Low Confidence Data:**\n\n {retrieved_text}")

# ✅ Testing & Validation
if st.sidebar.button("Run Test Queries"):
    st.sidebar.header("🔍 Testing & Validation")

    test_queries = [
        "Total Receivables from BMW Group companies",
        "Net Income",
        "What is the capital of France?"
    ]

    for test_query in test_queries:
        query_type = classify_query(test_query)

        if query_type == "irrelevant":
            st.sidebar.write(f"**🔹 Query:** {test_query} (❌ Irrelevant)")
            st.sidebar.write("**🔍 Confidence Score:** 0%")
            st.sidebar.write("⚠️ No relevant financial data available.")
            continue

        retrieved_text, retrieval_confidence = multistage_retrieve(test_query)
        final_confidence = calculate_confidence(retrieval_confidence)

        st.sidebar.write(f"**🔹 Query:** {test_query}")
        st.sidebar.write(f"**🔍 Confidence Score:** {final_confidence}%")

        if final_confidence >= 50:
            st.sidebar.success(f"✅ **Relevant Information:**\n\n {retrieved_text}")
        else:
            st.sidebar.warning(f"⚠️ **Low Confidence Data:**\n\n {retrieved_text}")
