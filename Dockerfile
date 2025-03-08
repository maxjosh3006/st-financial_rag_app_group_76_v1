# Build stage
FROM python:3.12.9-slim as builder

WORKDIR /st-financial_rag_app_group_76

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.12.9-slim

WORKDIR /st-financial_rag_app

# Copy necessary files from the builder stage
COPY --from=builder /st-financial_rag_app_group_76/ /st-financial_rag_app_group_76/

# Expose the Streamlit port
EXPOSE 8501

# Set the entrypoint to run Streamlit
ENTRYPOINT ["streamlit", "run", "st-financial_rag_app_group_76.py"]
