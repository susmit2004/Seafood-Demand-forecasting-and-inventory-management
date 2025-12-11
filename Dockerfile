# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/processed data/raw models/saved_models results

# Make startup script executable
RUN chmod +x start.sh 2>/dev/null || echo "start.sh will be available at runtime"

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Run both FastAPI and Streamlit
CMD ["bash", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
