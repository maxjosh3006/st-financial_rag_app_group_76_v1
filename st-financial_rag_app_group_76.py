import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import re

# ✅ Load Models
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ✅ Extract Financial Text from PDF with Improved Chunking & Table Extraction
def extract_financial_text(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chunks = re.split(r'(?=\b(?:Revenue|Income|Expenses|Receivables|Assets|Profit|Net Income|Earnings|Operating Activities|Cash Flow)\b)', text)
                extracted_text.extend([" ".join(chunk.split()[:400]) for chunk in chunks])  # Increased chunk size for better context
    return extracted_text

# ✅ Improved Table Extraction for Financial Values
def extract_tables_from_pdf(pdf_path):
    extracted_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = " ".join(str(cell) for cell in row if cell)
                    extracted_data.append(row_text.strip())
    return extracted_data

# ✅ Data Loading
pdf_path = "BMW_Finance_NV_Annual_Report_2023.pdf"
bm25_corpus = extract_financial_text(pdf_path) + extract_tables_from_pdf(pdf_path)

# ✅ BM25 Setup with Improved Tokenization & Keyword Weighting
bm25_tokenizer = lambda text: text.lower().split()
bm25 = BM25Okapi([bm25_tokenizer(doc) for doc in bm25_corpus])

# ✅ Embedding and FAISS Setup with Expanded Context
def embed_with_context(chunks):
    return [f"{chunk[:150]} ... {chunk[-150:]}" for chunk in chunks]  # Added context for FAISS

bm25_corpus_with_context = embed_with_context(bm25_corpus)
doc_embeddings = embed_model.encode(bm25_corpus_with_context, convert_to_tensor=True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# ✅ Enhanced Financial Value Extraction with Regex
def extract_financial_values(text):
    pattern = r'\b(?:cash flow|operating activities|investing activities|financing activities|net profit)\b.*?([\d,]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return matches if matches else ["No valid financial data found"]

# ✅ Guardrail for Irrelevant Queries
def is_financial_query(query):
    financial_keywords = [
        "revenue", "profit", "net income", "earnings", "expenses",
        "assets", "liabilities", "equity", "cash flow", "net profit",
        "operating activities", "investing activities", "financing activities"
    ]
    return any(keyword in query.lower() for keyword in financial_keywords)

# ✅ Multi-Stage Retrieval with Enhanced Guardrails and Confidence Calculation
def multi_stage_retrieval(query):
    if not is_financial_query(query):
        return "Irrelevant Query Detected.", [], 0

    # Stage 1: BM25 Search with Weighted Keywords
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:5]

    # Stage 2: FAISS Search with Expanded Context
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    _, top_faiss_indices = index.search(query_embedding.unsqueeze(0).numpy(), 5)

    # Combine Results with Weighted Confidence Score
    combined_results = set(top_bm25_indices) | set(top_faiss_indices[0])
    candidate_docs = [bm25_corpus[idx] for idx in combined_results]

    # Precision Scoring Logic
    top_result = max(candidate_docs, key=lambda doc: bm25_scores[bm25_corpus.index(doc)])
    confidence_score = round(max(bm25_scores) / 10, 2)

    financial_values = extract_financial_values(top_result)

    if financial_values and confidence_score >= 20:
        return top_result, financial_values, confidence_score
    else:
        return "No valid financial data found.", [], 0

# ✅ Streamlit UI
def main():
    st.title("Financial Insights RAG System")
    st.write("Ask any financial-related question based on available reports.")

    query = st.text_input("Enter your financial question:")

    if st.button("Submit"):
        result, financial_values, score = multi_stage_retrieval(query)
        st.write(f"**Answer:** {result}")
        if financial_values:
            st.info(f"**Extracted Data:** {', '.join(financial_values)}")
        st.write(f"**Confidence Score:** {score:.2f}")

    # ✅ Testing Framework with Button for Results
    if st.button("Run Test Cases"):
        test_queries = [
            "What is BMW's net profit in 2023?",
            "What are the total receivables in 2022?",
            "What is the capital of France?",
            "What is Cash flow from operating activities?"
        ]

        for query in test_queries:
            result, financial_values, score = multi_stage_retrieval(query)
            st.write(f"**Query:** {query}")
            st.write(f"**Answer:** {result}")
            if financial_values:
                st.info(f"**Extracted Data:** {', '.join(financial_values)}")
            st.write(f"**Confidence Score:** {score:.2f}")

if __name__ == "__main__":
    main()
