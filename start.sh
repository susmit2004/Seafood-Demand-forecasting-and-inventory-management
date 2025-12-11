#!/bin/bash
# Startup script for Seafood Forecasting Application

echo "Starting Seafood Forecasting Application..."

# Start FastAPI in the background
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
streamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0

# Wait for background processes
wait

