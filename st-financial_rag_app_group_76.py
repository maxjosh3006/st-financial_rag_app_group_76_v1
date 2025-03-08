import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load financial data
financial_data = pd.read_csv("financial_statements.csv")

# Chunking strategy (heading-based)
def custom_chunking(text, chunk_size=100):
    chunks = []
    current_chunk = []
    for line in text.split("\n"):
        if line.strip().startswith("#") or "Income Statement" in line:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        current_chunk.append(line)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# BM25 Retrieval for initial candidate selection
bm25 = BM25Okapi([chunk for chunk in custom_chunking(financial_data.to_string())])

# Embedding Model for Re-ranking
embedding_model = SentenceTransformer("ProsusAI/finbert")

# FAISS Index for Re-ranking
documents = custom_chunking(financial_data.to_string())
embeddings = embedding_model.encode(documents, convert_to_tensor=True)
faiss_index = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="ProsusAI/finbert"))

# Guardrail for irrelevant queries
def query_guardrail(query):
    valid_terms = ["net income", "revenue", "expenses", "assets", "liabilities"]
    return any(term in query.lower() for term in valid_terms)

# Retrieval Pipeline
def retrieve_answer(query):
    if not query_guardrail(query):
        return "❌ Irrelevant query detected. Please ask financial questions."
    
    # Stage 1: BM25 Retrieval
    bm25_results = bm25.get_top_n(query.split(), documents, n=5)
    
    # Stage 2: Re-ranking using FinBERT
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    re_ranked_docs = sorted(bm25_results, key=lambda doc: util.pytorch_cos_sim(query_embedding, embedding_model.encode(doc))[0].item(), reverse=True)

    # Stage 3: Summarization (Optional)
    best_answer = re_ranked_docs[0]  # Top-ranked document
    return best_answer

# Sample Queries
print(retrieve_answer("Total Receivables from BMW Group companies"))
print(retrieve_answer("Net Income"))
print(retrieve_answer("What is the capital of France?"))
