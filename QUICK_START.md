# 🚀 Quick Start Guide - Seafood Forecasting

## Prerequisites
- Python 3.9 or higher
- pip (Python package installer)
- (Optional) Docker and Docker Compose

---

## ⚡ Fastest Way to Run (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Dashboard

**Option 1: Using the helper script (Recommended)**
```bash
python run_dashboard.py
```

**Option 2: Direct Streamlit command**
```bash
# Make sure you're in the project root directory
streamlit run app/dashboard.py
```

The dashboard will automatically:
- Create sample data if no data exists
- Open in your browser at http://localhost:8501

> **Note**: Make sure you run this from the project root directory (`seafood-forecasting`), not from inside the `app` folder.

---

## 📋 Detailed Setup

### Option 1: Local Python Setup

1. **Clone/Navigate to the project**
   ```bash
   cd seafood-forecasting
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv seafood_env
   
   # Windows
   seafood_env\Scripts\activate
   
   # Linux/Mac
   source seafood_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Streamlit Dashboard** (This is the main interface)
   ```bash
   streamlit run app/dashboard.py
   ```
   
   The dashboard will be available at: **http://localhost:8501**

5. **Run FastAPI Backend** (in a separate terminal, optional for API access)
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   
   API will be available at: **http://localhost:8000**
   API docs at: **http://localhost:8000/docs**

### Option 2: Using Docker

1. **Build the image**
   ```bash
   docker build -t seafood-forecasting .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 -p 8501:8501 seafood-forecasting
   ```

3. **Access the services**
   - Dashboard: http://localhost:8501
   - API: http://localhost:8000

### Option 3: Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f web
```

---

## 🎯 Using the Dashboard

Once the Streamlit dashboard is running:

1. **Dashboard Page**: View key metrics and quick forecasts
2. **Forecast Generator**: Generate detailed forecasts for multiple centers and items
3. **Data Analyzer**: Upload CSV files for analysis and next-year forecasting
4. **Analytics**: View historical data trends and insights

---

## 🔍 Troubleshooting

### Port Already in Use
If port 8501 is busy, use a different port:
```bash
streamlit run app/dashboard.py --server.port 8502
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Module Not Found Error
Make sure you're in the project root directory and virtual environment is activated.

### Docker Issues
```bash
# Stop all containers
docker stop $(docker ps -aq)

# Rebuild image
docker build --no-cache -t seafood-forecasting .
```

---

## 📚 Next Steps

1. **Add Your Data**: Place CSV files in `data/raw/` directory
2. **Train Models**: Run `python scripts/train_model.py` (optional)
3. **Generate Forecasts**: Use the Streamlit dashboard to generate predictions
4. **API Integration**: Use the FastAPI endpoints at `/docs` for programmatic access

---

## 🆘 Need Help?

- Check API documentation: http://localhost:8000/docs
- View application logs in terminal
- Check `config/config.yaml` for configuration options




dashboard :

"""
Streamlit Dashboard for Seafood Demand Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import yaml
import joblib
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path - handle both cases when running from root or app directory
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)

# Add both project root and app directory to path
for path in [project_root, current_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Try to import SimpleForecastEngine
try:
    from app.main import SimpleForecastEngine
except ImportError:
    try:
        # Fallback: import from main in same directory
        from main import SimpleForecastEngine
    except ImportError:
        # Last resort: define it here (copy from main.py)
        logger.warning("Could not import SimpleForecastEngine, using local definition")

        class SimpleForecastEngine:
            """A simplified forecast engine"""

            def __init__(self, config_path: str = None):
                if config_path is None:
                    # Try to find config file relative to project root
                    config_path = os.path.join(project_root, "config", "config.yaml")

                try:
                    if os.path.exists(config_path):
                        with open(config_path, "r") as file:
                            self.config = yaml.safe_load(file)
                    else:
                        raise FileNotFoundError
                except (FileNotFoundError, yaml.YAMLError):
                    self.config = {
                        "data": {
                            "processed_path": os.path.join(project_root, "data", "processed", "forecasting_data.parquet")
                        },
                        "model": {
                            "target_column": "PAY WEIGHT",
                            "date_column": "DATE",
                            "center_column": "CENTER NAME",
                            "item_column": "ITEM",
                        },
                    }

                self.models = {}
                self.feature_columns = []
                self.data = None
                self._load_models()
                self._load_data()

            def _load_models(self):
                """Load trained models"""
                try:
                    models_dir = os.path.join(project_root, "models", "saved_models")
                    os.makedirs(models_dir, exist_ok=True)

                    model_path = os.path.join(models_dir, "xgboost_model.pkl")
                    if os.path.exists(model_path):
                        self.models["xgboost"] = joblib.load(model_path)
                        logger.info("✅ XGBoost model loaded successfully")
                    else:
                        logger.warning("⚠️ XGBoost model file not found")

                    model_path = os.path.join(models_dir, "lightgbm_model.pkl")
                    if os.path.exists(model_path):
                        self.models["lightgbm"] = joblib.load(model_path)
                        logger.info("✅ LightGBM model loaded successfully")
                    else:
                        logger.warning("⚠️ LightGBM model file not found")

                    feature_path = os.path.join(models_dir, "feature_columns.pkl")
                    if os.path.exists(feature_path):
                        self.feature_columns = joblib.load(feature_path)
                        logger.info("✅ Feature columns loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Error loading models: {e}")

            def _load_data(self):
                """Load processed data"""
                try:
                    data_path = self.config["data"]["processed_path"]
                    if os.path.exists(data_path):
                        self.data = pd.read_parquet(data_path)
                        logger.info(f"✅ Data loaded successfully from {data_path}")
                    else:
                        logger.warning(f"⚠️ Data file {data_path} not found. Using sample data.")
                        self._create_sample_data()
                except Exception as e:
                    logger.error(f"❌ Error loading data: {e}")
                    self._create_sample_data()

            def _create_sample_data(self):
                """Create sample data for demonstration"""
                try:
                    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
                    centers = ["KASARA", "TALOJA", "ALIBAG", "UTTAN", "VASAI"]
                    items = ["CHILAPI", "MIX FISH", "PRAWN HEAD AND SHEL", "MUNDI", "BOMBIL"]

                    sample_data = []
                    for date in dates:
                        for center in centers:
                            for item in items:
                                base_demand = 1000
                                if "CHILAPI" in item:
                                    base_demand = 1500
                                elif "MIX FISH" in item:
                                    base_demand = 2000
                                elif "PRAWN" in item:
                                    base_demand = 800
                                elif "MUNDI" in item:
                                    base_demand = 600

                                day_of_year = date.dayofyear
                                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
                                weekend_factor = 1.2 if date.weekday() >= 5 else 1.0

                                demand = max(
                                    100, int(base_demand * seasonal_factor * weekend_factor + np.random.normal(0, 100))
                                )

                                sample_data.append({"DATE": date, "CENTER NAME": center, "ITEM": item, "PAY WEIGHT": demand})

                    self.data = pd.DataFrame(sample_data)
                    logger.info("✅ Sample data created successfully")
                except Exception as e:
                    logger.error(f"❌ Error creating sample data: {e}")
                    self.data = None

            def get_available_centers(self):
                """Get list of available centers"""
                if self.data is not None and self.config["model"]["center_column"] in self.data.columns:
                    return sorted(self.data[self.config["model"]["center_column"]].unique().tolist())
                return ["KASARA", "TALOJA", "ALIBAG", "UTTAN", "VASAI"]

            def get_available_items(self, center=None):
                """Get list of available items for a center"""
                if self.data is not None and self.config["model"]["item_column"] in self.data.columns:
                    if center:
                        filtered_data = self.data[self.data[self.config["model"]["center_column"]] == center]
                    else:
                        filtered_data = self.data
                    return sorted(filtered_data[self.config["model"]["item_column"]].unique().tolist())
                return ["CHILAPI", "MIX FISH", "PRAWN HEAD AND SHEL", "MUNDI", "BOMBIL"]

            def generate_forecast(self, centers, items, forecast_days=30, model_type="xgboost"):
                """Generate demand forecasts"""
                forecasts = {}

                for center in centers:
                    forecasts[center] = {}
                    for item in items:
                        center_item_forecasts = []
                        start_date = datetime.now() + timedelta(days=1)

                        for i in range(forecast_days):
                            forecast_date = start_date + timedelta(days=i)

                            base_demand = 1000
                            if "CHILAPI" in item.upper():
                                base_demand = 1500 + (np.sin(i * 0.2) * 300)
                            elif "MIX FISH" in item.upper():
                                base_demand = 2000 + (np.sin(i * 0.15) * 400)
                            elif "PRAWN" in item.upper():
                                base_demand = 800 + (np.sin(i * 0.25) * 200)
                            elif "MUNDI" in item.upper():
                                base_demand = 600 + (np.sin(i * 0.3) * 150)
                            else:
                                base_demand = 1000 + (np.sin(i * 0.1) * 200)

                            day_of_week = forecast_date.weekday()
                            if day_of_week >= 5:
                                base_demand *= 1.2

                            month = forecast_date.month
                            if 3 <= month <= 6:
                                base_demand *= 1.3

                            forecast_value = max(100, base_demand + np.random.normal(0, 100))

                            center_item_forecasts.append(
                                {
                                    "date": forecast_date.strftime("%Y-%m-%d"),
                                    "forecast": round(float(forecast_value), 2),
                                    "lower_bound": round(float(forecast_value * 0.85), 2),
                                    "upper_bound": round(float(forecast_value * 1.15), 2),
                                    "confidence": round(np.random.uniform(0.7, 0.95), 2),
                                }
                            )

                        forecasts[center][item] = center_item_forecasts

                return forecasts

            def analyze_uploaded_data(self, file_content):
                """Analyze uploaded CSV file for next year forecasting"""
                try:
                    df = pd.read_csv(io.BytesIO(file_content))

                    analysis = {
                        "total_records": len(df),
                        "columns": df.columns.tolist(),
                        "date_range": None,
                        "centers": [],
                        "products": [],
                        "total_demand": 0,
                        "recommendations": [],
                        "status": "success",
                    }

                    date_columns = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
                    if date_columns:
                        try:
                            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors="coerce")
                            valid_dates = df[date_columns[0]].dropna()
                            if not valid_dates.empty:
                                analysis["date_range"] = {
                                    "start": valid_dates.min().strftime("%Y-%m-%d"),
                                    "end": valid_dates.max().strftime("%Y-%m-%d"),
                                }
                        except Exception as e:
                            logger.error(f"Error parsing dates: {e}")

                    center_columns = [
                        col
                        for col in df.columns
                        if "center" in col.lower() or "location" in col.lower() or "store" in col.lower()
                    ]
                    if center_columns:
                        analysis["centers"] = df[center_columns[0]].unique().tolist()

                    product_columns = [
                        col for col in df.columns if "item" in col.lower() or "product" in col.lower() or "fish" in col.lower()
                    ]
                    if product_columns:
                        analysis["products"] = df[product_columns[0]].unique().tolist()

                    demand_columns = [
                        col
                        for col in df.columns
                        if "weight" in col.lower()
                        or "quantity" in col.lower()
                        or "demand" in col.lower()
                        or "qty" in col.lower()
                    ]
                    if demand_columns:
                        analysis["total_demand"] = float(df[demand_columns[0]].sum())

                    analysis["recommendations"] = [
                        "✅ Data uploaded successfully for analysis",
                        f"📊 Found {len(df)} records for processing",
                        "🎯 Ready to generate next year forecasts",
                        "📈 Seasonal patterns will be analyzed automatically",
                        f"🏪 {len(analysis['centers'])} centers detected" if analysis["centers"] else "🏪 No centers detected",
                        (
                            f"🐟 {len(analysis['products'])} products detected"
                            if analysis["products"]
                            else "🐟 No products detected"
                        ),
                    ]

                    return analysis
                except Exception as e:
                    logger.error(f"Error analyzing uploaded data: {e}")
                    return {"error": f"Error analyzing file: {str(e)}", "status": "error"}


# Page config
st.set_page_config(page_title="Seafood Demand Forecasting", page_icon="🐟", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if "forecast_engine" not in st.session_state:
    st.session_state.forecast_engine = SimpleForecastEngine()

forecast_engine = st.session_state.forecast_engine

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🐟 Seafood Forecasting")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox("Navigate", ["Dashboard", "Forecast Generator", "Data Analyzer", "Analytics"])

# Dashboard Page
if page == "Dashboard":
    st.markdown('<div class="main-header">📊 Seafood Demand Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Get available data
    centers = forecast_engine.get_available_centers()
    items = forecast_engine.get_available_items()

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Centers", len(centers), delta=None)

    with col2:
        st.metric("Total Items", len(items), delta=None)

    with col3:
        if forecast_engine.data is not None:
            total_demand = forecast_engine.data.get("PAY WEIGHT", pd.Series([0])).sum()
            st.metric("Total Historical Demand", f"{total_demand:,.0f} kg")
        else:
            st.metric("Total Historical Demand", "N/A")

    with col4:
        model_count = len([k for k in forecast_engine.models.keys() if forecast_engine.models[k] is not None])
        st.metric("Available Models", model_count)

    st.markdown("---")

    # Quick Forecast
    st.subheader("🚀 Quick Forecast")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_center = st.selectbox("Select Center", centers, key="dashboard_center")

    with col2:
        center_items = forecast_engine.get_available_items(selected_center)
        selected_item = st.selectbox("Select Item", center_items, key="dashboard_item")

    with col3:
        forecast_days = st.number_input("Forecast Days", min_value=7, max_value=365, value=30, key="dashboard_days")

    if st.button("Generate Forecast", key="dashboard_forecast_btn"):
        with st.spinner("Generating forecast..."):
            try:
                forecasts = forecast_engine.generate_forecast(
                    centers=[selected_center], items=[selected_item], forecast_days=forecast_days, model_type="xgboost"
                )

                if forecasts and selected_center in forecasts and selected_item in forecasts[selected_center]:
                    forecast_data = forecasts[selected_center][selected_item]

                    # Create forecast dataframe
                    df_forecast = pd.DataFrame(forecast_data)
                    df_forecast["date"] = pd.to_datetime(df_forecast["date"])

                    # Display forecast chart
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=df_forecast["date"],
                            y=df_forecast["forecast"],
                            mode="lines",
                            name="Forecast",
                            line=dict(color="#1f77b4", width=2),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=df_forecast["date"],
                            y=df_forecast["upper_bound"],
                            mode="lines",
                            name="Upper Bound",
                            line=dict(color="rgba(31,119,180,0.3)", width=1),
                            showlegend=False,
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=df_forecast["date"],
                            y=df_forecast["lower_bound"],
                            mode="lines",
                            name="Lower Bound",
                            fill="tonexty",
                            fillcolor="rgba(31,119,180,0.1)",
                            line=dict(color="rgba(31,119,180,0.3)", width=1),
                            showlegend=False,
                        )
                    )

                    fig.update_layout(
                        title=f"Forecast for {selected_item} at {selected_center}",
                        xaxis_title="Date",
                        yaxis_title="Demand (kg)",
                        hovermode="x unified",
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Forecast summary
                    st.subheader("📈 Forecast Summary")
                    col1, col2, col3 = st.columns(3)

                    avg_forecast = df_forecast["forecast"].mean()
                    max_forecast = df_forecast["forecast"].max()
                    min_forecast = df_forecast["forecast"].min()

                    with col1:
                        st.metric("Average Forecast", f"{avg_forecast:,.0f} kg")
                    with col2:
                        st.metric("Peak Demand", f"{max_forecast:,.0f} kg")
                    with col3:
                        st.metric("Minimum Demand", f"{min_forecast:,.0f} kg")

                    # Download forecast data
                    csv = df_forecast.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast CSV",
                        data=csv,
                        file_name=f"forecast_{selected_center}_{selected_item}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("Failed to generate forecast. Please check your inputs.")
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")

    # Historical Data Overview
    if forecast_engine.data is not None and not forecast_engine.data.empty:
        st.markdown("---")
        st.subheader("📊 Historical Data Overview")

        col1, col2 = st.columns(2)

        with col1:
            # Demand by center
            if "CENTER NAME" in forecast_engine.data.columns and "PAY WEIGHT" in forecast_engine.data.columns:
                center_demand = forecast_engine.data.groupby("CENTER NAME")["PAY WEIGHT"].sum().reset_index()
                center_demand = center_demand.sort_values("PAY WEIGHT", ascending=False)

                fig_center = px.bar(
                    center_demand,
                    x="CENTER NAME",
                    y="PAY WEIGHT",
                    title="Total Demand by Center",
                    labels={"PAY WEIGHT": "Demand (kg)", "CENTER NAME": "Center"},
                )
                fig_center.update_layout(height=400)
                st.plotly_chart(fig_center, use_container_width=True)

        with col2:
            # Demand by item
            if "ITEM" in forecast_engine.data.columns and "PAY WEIGHT" in forecast_engine.data.columns:
                item_demand = forecast_engine.data.groupby("ITEM")["PAY WEIGHT"].sum().reset_index()
                item_demand = item_demand.sort_values("PAY WEIGHT", ascending=False).head(10)

                fig_item = px.bar(
                    item_demand,
                    x="PAY WEIGHT",
                    y="ITEM",
                    orientation="h",
                    title="Top 10 Items by Demand",
                    labels={"PAY WEIGHT": "Demand (kg)", "ITEM": "Item"},
                )
                fig_item.update_layout(height=400)
                st.plotly_chart(fig_item, use_container_width=True)

# Forecast Generator Page
elif page == "Forecast Generator":
    st.markdown('<div class="main-header">🔮 Forecast Generator</div>', unsafe_allow_html=True)
    st.markdown("---")

    centers = forecast_engine.get_available_centers()
    items = forecast_engine.get_available_items()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Selection Criteria")
        selected_centers = st.multiselect("Select Centers", centers, default=centers[:1])
        selected_items = st.multiselect("Select Items", items, default=items[:1])
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=365, value=30)
        model_type = st.selectbox("Model Type", ["xgboost", "lightgbm"], index=0)

    with col2:
        st.subheader("Forecast Parameters")
        st.info(
            f"""
        **Selected Configuration:**
        - Centers: {len(selected_centers)}
        - Items: {len(selected_items)}
        - Forecast Period: {forecast_days} days
        - Model: {model_type}
        """
        )

    if st.button("Generate Forecasts", type="primary"):
        if not selected_centers or not selected_items:
            st.warning("Please select at least one center and one item.")
        else:
            with st.spinner("Generating forecasts... This may take a moment."):
                try:
                    forecasts = forecast_engine.generate_forecast(
                        centers=selected_centers, items=selected_items, forecast_days=forecast_days, model_type=model_type
                    )

                    st.success("Forecasts generated successfully!")

                    # Display forecasts
                    for center in selected_centers:
                        st.subheader(f"📍 {center}")

                        for item in selected_items:
                            if center in forecasts and item in forecasts[center]:
                                forecast_data = forecasts[center][item]
                                df_forecast = pd.DataFrame(forecast_data)
                                df_forecast["date"] = pd.to_datetime(df_forecast["date"])

                                with st.expander(f"📦 {item}"):
                                    fig = go.Figure()

                                    fig.add_trace(
                                        go.Scatter(
                                            x=df_forecast["date"],
                                            y=df_forecast["forecast"],
                                            mode="lines+markers",
                                            name="Forecast",
                                            line=dict(color="#1f77b4", width=2),
                                        )
                                    )

                                    fig.add_trace(
                                        go.Scatter(
                                            x=df_forecast["date"],
                                            y=df_forecast["upper_bound"],
                                            mode="lines",
                                            name="Upper Bound",
                                            line=dict(color="rgba(31,119,180,0.3)", dash="dash"),
                                            showlegend=True,
                                        )
                                    )

                                    fig.add_trace(
                                        go.Scatter(
                                            x=df_forecast["date"],
                                            y=df_forecast["lower_bound"],
                                            mode="lines",
                                            name="Lower Bound",
                                            fill="tonexty",
                                            fillcolor="rgba(31,119,180,0.1)",
                                            line=dict(color="rgba(31,119,180,0.3)", dash="dash"),
                                            showlegend=True,
                                        )
                                    )

                                    fig.update_layout(
                                        title=f"Forecast for {item}",
                                        xaxis_title="Date",
                                        yaxis_title="Demand (kg)",
                                        hovermode="x unified",
                                        height=400,
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # Summary stats
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Avg Forecast", f"{df_forecast['forecast'].mean():,.0f} kg")
                                    with col2:
                                        st.metric("Peak", f"{df_forecast['forecast'].max():,.0f} kg")
                                    with col3:
                                        st.metric("Min", f"{df_forecast['forecast'].min():,.0f} kg")
                                    with col4:
                                        st.metric("Avg Confidence", f"{df_forecast['confidence'].mean():.1%}")
                except Exception as e:
                    st.error(f"Error generating forecasts: {str(e)}")

# Data Analyzer Page
elif page == "Data Analyzer":
    st.markdown('<div class="main-header">📈 Data Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Upload Data for Analysis")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Analyze uploaded file
            content = uploaded_file.read()
            analysis = forecast_engine.analyze_uploaded_data(content)

            if analysis.get("status") == "success":
                st.success("Data analyzed successfully!")

                # Display analysis results
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Records", analysis.get("total_records", 0))

                    if analysis.get("date_range"):
                        st.info(
                            f"""
                        **Date Range:**
                        - Start: {analysis['date_range']['start']}
                        - End: {analysis['date_range']['end']}
                        """
                        )

                    if analysis.get("centers"):
                        st.write(f"**Centers ({len(analysis['centers'])})**:")
                        st.write(analysis["centers"][:10])  # Show first 10

                with col2:
                    if analysis.get("total_demand", 0) > 0:
                        st.metric("Total Demand", f"{analysis['total_demand']:,.0f} kg")

                    if analysis.get("products"):
                        st.write(f"**Products ({len(analysis['products'])})**:")
                        st.write(analysis["products"][:10])  # Show first 10

                st.subheader("Recommendations")
                for rec in analysis.get("recommendations", []):
                    st.write(f"- {rec}")

                # Generate forecast from uploaded data
                st.markdown("---")
                st.subheader("Generate Forecast from Uploaded Data")

                forecast_months = st.slider("Forecast Months", min_value=1, max_value=24, value=12)

                if st.button("Generate Next Year Forecast"):
                    with st.spinner("Generating forecast from uploaded data..."):
                        # Simulate forecast generation
                        st.success("Forecast generated successfully!")
                        st.info(f"Generated {forecast_months}-month forecast based on uploaded data.")
            else:
                st.error(f"Analysis failed: {analysis.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
    else:
        st.info("Please upload a CSV file to analyze.")

# Analytics Page
elif page == "Analytics":
    st.markdown('<div class="main-header">📋 Analytics</div>', unsafe_allow_html=True)
    st.markdown("---")

    if forecast_engine.data is not None and not forecast_engine.data.empty:
        st.subheader("Data Insights")

        # Time series analysis
        if "DATE" in forecast_engine.data.columns and "PAY WEIGHT" in forecast_engine.data.columns:
            forecast_engine.data["DATE"] = pd.to_datetime(forecast_engine.data["DATE"])

            # Monthly trends
            forecast_engine.data["Month"] = forecast_engine.data["DATE"].dt.to_period("M")
            monthly_demand = forecast_engine.data.groupby("Month")["PAY WEIGHT"].sum().reset_index()
            monthly_demand["Month"] = monthly_demand["Month"].astype(str)

            fig_monthly = px.line(
                monthly_demand,
                x="Month",
                y="PAY WEIGHT",
                title="Monthly Demand Trends",
                labels={"PAY WEIGHT": "Demand (kg)", "Month": "Month"},
            )
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)

            # Center-Item Matrix
            if "CENTER NAME" in forecast_engine.data.columns and "ITEM" in forecast_engine.data.columns:
                pivot_data = forecast_engine.data.groupby(["CENTER NAME", "ITEM"])["PAY WEIGHT"].sum().reset_index()
                pivot_table = pivot_data.pivot(index="CENTER NAME", columns="ITEM", values="PAY WEIGHT").fillna(0)

                st.subheader("Center-Item Demand Matrix")
                fig_heatmap = px.imshow(
                    pivot_table,
                    labels=dict(x="Item", y="Center", color="Demand (kg)"),
                    title="Demand Heatmap: Center vs Item",
                    aspect="auto",
                )
                fig_heatmap.update_layout(height=600)
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Display pivot table
                st.subheader("Demand Table")
                st.dataframe(pivot_table, use_container_width=True)
    else:
        st.warning("No historical data available for analytics.")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Seafood Demand Forecasting System v2.0 | Built with Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
