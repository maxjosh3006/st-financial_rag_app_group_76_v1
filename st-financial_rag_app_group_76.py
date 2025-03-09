import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import re

# ✅ Load Models
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ✅ Extract Financial Text from PDF with Improved Chunking
def extract_financial_text(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chunks = re.split(r'(?=\b(?:Revenue|Income|Expenses|Receivables|Assets|Liabilities|Cash Flow|income|loss)\b)', text)
                extracted_text.extend(chunks)
    return extracted_text

# ✅ Data Loading
pdf_path = "BMW_Finance_NV_Annual_Report_2023.pdf"
bm25_corpus = extract_financial_text(pdf_path)

# ✅ BM25 Setup with Improved Tokenization
bm25_tokenizer = lambda text: text.lower().split()
bm25 = BM25Okapi([bm25_tokenizer(doc) for doc in bm25_corpus])

# ✅ Embedding and FAISS Setup
doc_embeddings = embed_model.encode(bm25_corpus, convert_to_tensor=True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# ✅ Enhanced Financial Value Extraction
def extract_financial_values(text):
    pattern = r'\b(?:\$|€|£)?[\d,.]+(?:\s?(bn|m|million|billion))?\b'
    return re.findall(pattern, text, re.IGNORECASE)

# ✅ Guardrail for Irrelevant Queries
def is_financial_query(query):
    financial_keywords = [
        "revenue", "profit", "expenses", "income", "assets", "liabilities", "equity", "cash flow", "net income", "income" ,"loss"
    ]
    return any(keyword in query.lower() for keyword in financial_keywords)

# ✅ Multi-Stage Retrieval with Enhanced Guardrails and Confidence Calculation
def multi_stage_retrieval(query):
    if not is_financial_query(query):
        return "Irrelevant Query Detected.", [], 0

    # Stage 1: BM25 Search
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:5]

    # Stage 2: FAISS Search
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    _, top_faiss_indices = index.search(query_embedding.unsqueeze(0).numpy(), 5)

    # Combine Results with Weighted Confidence Score
    combined_results = set(top_bm25_indices) | set(top_faiss_indices[0])
    candidate_docs = [bm25_corpus[idx] for idx in combined_results]

    # Precision Scoring Logic
    top_result = max(candidate_docs, key=lambda doc: bm25_scores[bm25_corpus.index(doc)])
    confidence_score = round(max(bm25_scores) / 10, 2)

    financial_values = extract_financial_values(top_result)

    if financial_values and confidence_score >= 30:
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
            "What is Income/(loss) before taxation?",
            "What is Interest and interest related income?",
            "What is the capital of France?"
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
