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

# ✅ Ensure NLTK's Punkt tokenizer is available
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

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

# ✅ Improved Context Extraction (More Precise)
def extract_relevant_sentences(retrieved_chunks, query, max_sentences=3):
    sentences = []
    for chunk in retrieved_chunks:
        if not chunk or not chunk.strip():  # 🔹 Skip empty chunks
            continue
        chunk_sentences = sent_tokenize(chunk)  # ✅ Tokenize into sentences

        # ✅ Keep only sentences with financial data (numbers) or matching query terms
        for sentence in chunk_sentences:
            if re.search(r"\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?", sentence) or any(word.lower() in sentence.lower() for word in query.split()):
                sentences.append(sentence)

    return " ".join(sentences[:max_sentences]) if sentences else "No relevant data found."

# ✅ Hallucination Filtering (Output-Side)
def filter_hallucinations(response, query, confidence_threshold=50):
    """
    Filters hallucinated or misleading responses.
    - If confidence is low and response lacks financial terms, flag it.
    """
    financial_keywords = ["revenue", "profit", "income", "assets", "liabilities", "equity", "expenses", "financial", "cash flow"]
    
    if confidence_threshold < 50 and not any(word in response.lower() for word in financial_keywords):
        return "⚠️ The retrieved answer may not be reliable. Please verify with official financial statements."
    
    return response

# ✅ Multi-Stage Retrieval with Context Filtering & Hallucination Handling
def multistage_retrieve(query, k=5, bm25_k=20, alpha=0.7): 
    if not query or not query.strip():
        return "No query provided.", 0.0

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

    # ✅ Apply refined sentence extraction for better precision
    precise_context = extract_relevant_sentences(retrieved_chunks, query)
    
    # ✅ Apply hallucination filter
    final_response = filter_hallucinations(precise_context, query, retrieval_confidence)

    return final_response, round(retrieval_confidence, 2)

# ✅ Streamlit UI
st.title("📊 Financial Statement Q&A")
query = st.text_input("Enter your financial question:", key="financial_query")

if query:
    retrieved_text, retrieval_confidence = multistage_retrieve(query)
    st.write(f"### 🔍 Confidence Score: {retrieval_confidence}%")

    if retrieval_confidence >= 50:  # High confidence
        st.success(f"**✅ Relevant Information:**\n\n {retrieved_text}")
    else:  # Low confidence
        st.warning(f"⚠️ **Low Confidence Data:**\n\n {retrieved_text}")

# ✅ Testing & Validation
if st.sidebar.button("Run Test Queries"):
    st.sidebar.header("🔍 Testing & Validation")

    test_queries = [
        ("What is the Trade receivables from BMW Group companies? Provide precise response within max 3 sentences", "High Confidence"),
        ("How did changes in interest rates impact BMW Finance N.V.'s profitability in 2023? Provide precise response within max 3 sentences", "Low Confidence"),
        ("What is the capital of France?", "Irrelevant")
    ]

    for test_query, confidence_level in test_queries:
        retrieved_text, retrieval_confidence = multistage_retrieve(test_query)
        st.sidebar.write(f"**🔹 Query:** {test_query}")
        st.sidebar.write(f"**🔍 Confidence Score:** {retrieval_confidence}%")

        if retrieval_confidence >= 70:
            st.sidebar.success(f"✅ **Relevant Information:**\n\n {retrieved_text}")
        else:
            st.sidebar.warning(f"⚠️ **Low Confidence Data:**\n\n {retrieved_text}")
