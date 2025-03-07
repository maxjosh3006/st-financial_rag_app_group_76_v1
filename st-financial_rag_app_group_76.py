# ✅ Load Financial PDF & Process Data

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
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)]

# ✅ Load Data
pdf_path = "/BMW_Finance_NV_Annual_Report_2023.pdf"
pdf_text = load_pdf(pdf_path)
tables = extract_tables_from_pdf(pdf_path)
text_chunks = chunk_text(pdf_text)

# ✅ Set Up Multi-Stage Retrieval (BM25 + FAISS)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

tokenized_chunks = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# ✅ Multi-Stage Retrieval with Confidence Score
def multistage_retrieve(query, k=5, bm25_k=10, alpha=0.5):
    """Retrieval + Confidence Score: Uses BM25, FAISS, and re-ranking."""
    query_embedding = embedding_model.encode([query])

    # 🔹 Stage 1: BM25 Keyword Search
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_indices = np.argsort(bm25_scores)[-bm25_k:]
    bm25_confidence = np.max(bm25_scores)  # Get highest BM25 score

    # 🔹 Stage 2: FAISS Vector Search
    filtered_embeddings = np.array([chunk_embeddings[i] for i in top_bm25_indices])
    faiss_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    faiss_index.add(filtered_embeddings)

    _, faiss_ranks = faiss_index.search(query_embedding, k)
    top_faiss_indices = [top_bm25_indices[i] for i in faiss_ranks[0]]

    # 🔹 Stage 3: Re-Ranking (BM25 + FAISS Scores)
    final_scores = {}
    faiss_confidence = 0  # Initialize FAISS confidence
    for i in set(top_bm25_indices) | set(top_faiss_indices):
        bm25_score = bm25_scores[i] if i in top_bm25_indices else 0
        faiss_score = -np.linalg.norm(query_embedding - chunk_embeddings[i])  # L2 distance
        final_scores[i] = alpha * bm25_score + (1 - alpha) * faiss_score
        faiss_confidence = max(faiss_confidence, faiss_score)  # Get highest FAISS score

    # Normalize confidence (scale to 0-100)
    bm25_confidence = (bm25_confidence / max(bm25_scores)) * 100 if max(bm25_scores) > 0 else 0
    faiss_confidence = (faiss_confidence + 1) * 50  # Scale from -1 to 1 into 0-100

    # Final Confidence Score
    final_confidence = (bm25_confidence + faiss_confidence) / 2

    # Get Top K Chunks
    top_chunks = sorted(final_scores, key=final_scores.get, reverse=True)[:k]
    return [text_chunks[i] for i in top_chunks], round(final_confidence, 2)

# ✅ Retrieve Financial Values from Tables
def extract_financial_value(tables, query):
    """Find financial values using fuzzy matching with table match confidence."""
    possible_headers = []

    for table in tables:
        for row in table:
            row_text = " ".join(str(cell) for cell in row if cell)
            possible_headers.append(row_text)

    # 🔹 Find Best-Matching Row for the Query
    extraction_result = process.extractOne(query, possible_headers, score_cutoff=80)

    if extraction_result:
        best_match, score = extraction_result
    else:
        return ["No valid financial data found"], 0  # No match → Confidence = 0

    # 🔹 Extract Correct Numbers from the Matched Row
    for table in tables:
        for row in table:
            row_text = " ".join(str(cell) for cell in row if cell)
            if best_match in row_text:
                numbers = [cell for cell in row if re.match(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", str(cell))]
                if len(numbers) >= 2:
                    return numbers[:2], round(score, 2)  # Return values + confidence

    return ["No valid financial data found"], 0

# ✅ Streamlit UI
st.set_page_config(page_title="Financial Query System", layout="wide")

st.title("📊 Financial Statement Q&A")
st.subheader("Ask financial questions based on the latest company reports")

query = st.text_input("Enter your financial question:")

if query:
    retrieved_chunks, retrieval_confidence = multistage_retrieve(query)
    retrieved_text = "\n".join(retrieved_chunks)
    financial_values, table_confidence = extract_financial_value(tables, query)

    final_confidence = round((retrieval_confidence + table_confidence) / 2, 2)

    st.write("### ✅ Retrieved Context")
    st.success(retrieved_text)

    if financial_values and financial_values[0] != "No valid financial data found":
        st.write("### 📊 Extracted Financial Data")
        st.info(f"**2023:** {financial_values[0]}, **2022:** {financial_values[1]}")
    else:
        st.warning("No valid financial data found for this query.")

    st.write(f"### 🔍 Confidence Score: {final_confidence}%")

# ✅ Testing & Validation
st.sidebar.header("🔍 Testing & Validation")

test_queries = [
    ("Total Receivables from BMW Group companies", "High Confidence"),
    ("Revenue Growth over 5 years", "Low Confidence"),
    ("What is the capital of France?", "Irrelevant")
]

for test_query, confidence_level in test_queries:
    st.sidebar.subheader(f"📝 Test: {test_query} ({confidence_level})")
    
    retrieved_chunks, retrieval_confidence = multistage_retrieve(test_query)
    retrieved_text = "\n".join(retrieved_chunks)
    financial_values, table_confidence = extract_financial_value(tables, test_query)

    final_confidence = round((retrieval_confidence + table_confidence) / 2, 2)

    st.sidebar.write(f"**🔹 Retrieved Context:** {retrieved_text[:500]}...")
    st.sidebar.write(f"**🔍 Confidence Score:** {final_confidence}%")

    if financial_values and financial_values[0] != "No valid financial data found":
        st.sidebar.write(f"📊 **Extracted Financial Data:** 2023: {financial_values[0]}, 2022: {financial_values[1]}")
    else:
        st.sidebar.warning("⚠️ No valid financial data found.")

st.sidebar.write("✅ Testing Complete")
