import streamlit as st
import pdfplumber
import re
from thefuzz import process

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

# ✅ Improved Financial Data Extraction with Flexible Matching
def extract_financial_value(tables, query):
    possible_headers = []
    for table in tables:
        for row in table:
            row_text = " ".join(str(cell) for cell in row if cell)
            possible_headers.append(row_text)

    extraction_result = process.extractOne(query, possible_headers, score_cutoff=85)

    if extraction_result:
        best_match, score = extraction_result
    else:
        return ["No valid financial data found"], 0

    for table in tables:
        for row in table:
            row_text = " ".join(str(cell) for cell in row if cell)
            if best_match in row_text:
                numbers = [cell for cell in row if re.match(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", str(cell))]
                if len(numbers) >= 2:
                    return numbers[:2], round(score, 2)

    return ["No valid financial data found"], 0

# ✅ Streamlit UI
st.title("📊 Financial Data Extractor")
query = st.text_input("Enter your financial data query:")

pdf_path = "BMW_Finance_NV_Annual_Report_2023.pdf"
tables = extract_tables_from_pdf(pdf_path)

if query:
    financial_values, table_confidence = extract_financial_value(tables, query)

    if financial_values and financial_values[0] != "No valid financial data found":
        st.write("### 📊 Extracted Financial Data")
        st.info(f"**2023:** {financial_values[0]}, **2022:** {financial_values[1]}")
    else:
        st.warning("⚠️ No valid financial data found. Try rephrasing your query for better results.")
