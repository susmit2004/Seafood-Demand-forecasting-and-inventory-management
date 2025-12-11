# 🐟 Seafood Demand Forecasting

AI-powered demand forecasting system for seafood distribution with inventory optimization.

## 🚀 Features

- **AI-Powered Forecasting**: XGBoost and LightGBM models for accurate predictions
- **Real-time Dashboard**: Beautiful glassmorphism UI with interactive charts
- **Data Analysis**: Upload CSV files for instant insights and next-year forecasts
- **Inventory Optimization**: Smart recommendations to reduce waste and prevent stockouts
- **MLOps Pipeline**: Complete CI/CD with automated testing and deployment

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.9
- **Frontend**: Streamlit Dashboard
- **ML Models**: XGBoost, LightGBM, Scikit-learn
- **Database**: PostgreSQL
- **MLOps**: MLflow, Docker, GitHub Actions
- **Monitoring**: Health checks, logging, alerts

## 📦 Quick Start

### Method 1: Local Development (Recommended for first-time setup)

#### Step 1: Install Dependencies
```bash
# Create virtual environment (optional but recommended)
python -m venv seafood_env

# Activate virtual environment
# On Windows:
seafood_env\Scripts\activate
# On Linux/Mac:
source seafood_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Prepare Data (Optional)
```bash
# Place your CSV file in data/raw/ directory
# If you don't have data, the app will create sample data automatically
```

#### Step 3: Run the Application

**Option A: Run Both Services Separately (Recommended)**

Terminal 1 - Start FastAPI Backend:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 - Start Streamlit Dashboard (Run from project root):
```bash
# Method 1: Using the helper script (easiest - recommended)
python run_dashboard.py

# Method 2: Direct streamlit command
streamlit run app/dashboard.py --server.port 8501
```

**Option B: Use the Quick Setup Script**
```bash
python run.py
```
This will start the FastAPI server. Open another terminal and run:
```bash
# Easiest method:
python run_dashboard.py

# Or directly:
streamlit run app/dashboard.py
```

#### Step 4: Access the Application
- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

### Method 2: Using Docker

#### Build and Run with Docker
```bash
# Build the Docker image
docker build -t seafood-forecasting .

# Run the container
docker run -p 8000:8000 -p 8501:8501 seafood-forecasting
```

Then access:
- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI API**: http://localhost:8000

---

### Method 3: Using Docker Compose (Full Stack)

#### Start All Services (PostgreSQL, MLflow, MinIO, Web App)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access the services:
- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI API**: http://localhost:8000
- **MLflow UI**: http://localhost:5000
- **MinIO Console**: http://localhost:9001

---

## 🔧 Common Commands

### Data Pipeline (Optional - if you have raw data)
```bash
python scripts/data_pipeline.py
```

### Train Models (Optional)
```bash
python scripts/train_model.py
```

### Run Tests
```bash
pytest tests/
```

### Check Application Health
```bash
curl http://localhost:8000/health
```
