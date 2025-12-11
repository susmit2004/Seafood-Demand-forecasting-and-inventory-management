"""
Streamlit Dashboard for Jagdamba fisheries Demand Forecasting.
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
from pandas.tseries.offsets import MonthBegin

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
                        logger.info("XGBoost model loaded successfully")
                    else:
                        logger.warning("XGBoost model file not found")

                    model_path = os.path.join(models_dir, "lightgbm_model.pkl")
                    if os.path.exists(model_path):
                        self.models["lightgbm"] = joblib.load(model_path)
                        logger.info("LightGBM model loaded successfully")
                    else:
                        logger.warning("LightGBM model file not found")

                    feature_path = os.path.join(models_dir, "feature_columns.pkl")
                    if os.path.exists(feature_path):
                        self.feature_columns = joblib.load(feature_path)
                        logger.info("Feature columns loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading models: {e}")

            def _load_data(self):
                """Load processed data"""
                try:
                    data_path = self.config["data"]["processed_path"]
                    if os.path.exists(data_path):
                        self.data = pd.read_parquet(data_path)
                        logger.info(f"Data loaded successfully from {data_path}")
                    else:
                        logger.warning(f"Data file {data_path} not found. Using sample data.")
                        self._create_sample_data()
                except Exception as e:
                    logger.error(f"Error loading data: {e}")
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
                    logger.info("Sample data created successfully")
                except Exception as e:
                    logger.error(f"Error creating sample data: {e}")
                    self.data = None

            def get_available_centers(self):
                """Get list of available centers"""
                if self.data is not None and self.config["model"]["center_column"] in self.data.columns:
                    # Remove NaN values and duplicates, then sort
                    centers = self.data[self.config["model"]["center_column"]].dropna().unique().tolist()
                    # Remove any duplicates and filter out empty strings
                    centers = [c for c in centers if c and str(c).strip()]
                    return sorted(list(set(centers)))  # Use set to remove duplicates, then sort
                return ["KASARA", "TALOJA", "ALIBAG", "UTTAN", "VASAI"]

            def get_available_items(self, center=None):
                """Get list of available items for a center"""
                if self.data is not None and self.config["model"]["item_column"] in self.data.columns:
                    if center:
                        filtered_data = self.data[self.data[self.config["model"]["center_column"]] == center]
                    else:
                        filtered_data = self.data
                    # Remove NaN values and duplicates, then sort
                    items = filtered_data[self.config["model"]["item_column"]].dropna().unique().tolist()
                    # Remove any duplicates and filter out empty strings
                    items = [i for i in items if i and str(i).strip()]
                    return sorted(list(set(items)))  # Use set to remove duplicates, then sort
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

            def generate_forecast_from_uploaded_data(self, df, forecast_months=12):
                """Generate aggregated forecast from uploaded CSV data"""
                if df is None or df.empty:
                    return []

                working_df = df.copy()

                date_columns = [
                    col
                    for col in working_df.columns
                    if "date" in col.lower() or "time" in col.lower()
                ]
                demand_columns = [
                    col
                    for col in working_df.columns
                    if "weight" in col.lower()
                    or "quantity" in col.lower()
                    or "demand" in col.lower()
                    or "qty" in col.lower()
                ]
                center_columns = [
                    col
                    for col in working_df.columns
                    if "center" in col.lower() or "location" in col.lower() or "store" in col.lower()
                ]
                item_columns = [
                    col
                    for col in working_df.columns
                    if "item" in col.lower() or "product" in col.lower() or "fish" in col.lower()
                ]

                date_col = date_columns[0] if date_columns else None
                demand_col = demand_columns[0] if demand_columns else None
                center_col = center_columns[0] if center_columns else None
                item_col = item_columns[0] if item_columns else None

                if demand_col is None:
                    return []

                if date_col is None:
                    working_df["__generated_date"] = pd.date_range(
                        start=datetime.now() - timedelta(days=len(working_df)),
                        periods=len(working_df),
                        freq="D",
                    )
                    date_col = "__generated_date"

                working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
                working_df = working_df.dropna(subset=[date_col])

                if working_df.empty:
                    return []

                if center_col is None:
                    working_df["__center"] = "All Locations"
                    center_col = "__center"

                if item_col is None:
                    working_df["__item"] = "All Products"
                    item_col = "__item"

                working_df = working_df[[date_col, center_col, item_col, demand_col]].copy()
                working_df["Month"] = working_df[date_col].dt.to_period("M").dt.to_timestamp()

                grouped = (
                    working_df.groupby([center_col, item_col, "Month"])[demand_col]
                    .sum()
                    .reset_index()
                )

                if grouped.empty:
                    return []

                records = []
                for (center, item), subset in grouped.groupby([center_col, item_col]):
                    subset = subset.sort_values("Month")
                    baseline = subset[demand_col].tail(3).mean()
                    if pd.isna(baseline):
                        baseline = subset[demand_col].mean()
                    if pd.isna(baseline) or baseline <= 0:
                        baseline = max(subset[demand_col].median(), 1)

                    last_month = subset["Month"].max()
                    future_months = pd.date_range(
                        last_month + MonthBegin(),
                        periods=forecast_months,
                        freq="MS",
                    )

                    for idx, month in enumerate(future_months):
                        seasonality = 1 + 0.12 * np.sin((idx / 12) * 2 * np.pi)
                        trend = 1 + 0.015 * idx
                        forecast_value = max(0, baseline * seasonality * trend)

                        records.append(
                            {
                                "Center": center,
                                "Item": item,
                                "Month": month.strftime("%Y-%m"),
                                "Forecast": round(float(forecast_value), 2),
                                "LowerBound": round(float(forecast_value * 0.9), 2),
                                "UpperBound": round(float(forecast_value * 1.1), 2),
                            }
                        )

                return records

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
                        "Data upload completed and ready for analysis.",
                        f"{len(df)} records identified for processing.",
                        "Next-year forecasting pipeline configured.",
                        "Seasonal demand patterns will be evaluated automatically.",
                        f"{len(analysis['centers'])} locations detected." if analysis["centers"] else "No locations detected.",
                        f"{len(analysis['products'])} product groups detected." if analysis["products"] else "No product groups detected.",
                    ]

                    analysis["dataframe"] = df

                    return analysis
                except Exception as e:
                    logger.error(f"Error analyzing uploaded data: {e}")
                    return {"error": f"Error analyzing file: {str(e)}", "status": "error"}


# Page config
icon_path = os.path.join(project_root, "app", "frontend", "static", "images", "Jagdamba fisheries_dashboard_icon.svg")
st.set_page_config(
    page_title="Jagdamba Fisheries Demand Forecasting",
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "forecast_engine" not in st.session_state:
    st.session_state.forecast_engine = SimpleForecastEngine()

forecast_engine = st.session_state.forecast_engine


def get_theme_palette():
    """Return color palette based on configured Streamlit theme."""
    try:
        base_theme = st.get_option("theme.base") or "light"
    except Exception:
        # Fallback if theme option is not available
        base_theme = "light"
    
    if base_theme.lower() == "dark":
        return {
            "background": "#0f172a",
            "surface": "#1e293b",
            "card": "#1f2937",
            "border": "#334155",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "accent": "#60a5fa",
            "accent_alt": "#818cf8",
            "success_bg": "#064e3b",
            "success_border": "#34d399",
            "info_bg": "#1e3a8a",
            "info_border": "#60a5fa",
            "warning_bg": "#78350f",
            "warning_border": "#fbbf24",
            "grid": "#334155",
            "plot_bg": "#1e293b",
            "paper_bg": "#1e293b",
        }
    return {
        "background": "#f4f6fb",
        "surface": "#ffffff",
        "card": "#ffffff",
        "border": "#e2e8f0",
        "text": "#0f172a",
        "muted": "#475569",
        "accent": "#2563eb",
        "accent_alt": "#0ea5e9",
        "success_bg": "#ecfdf3",
        "success_border": "#34d399",
        "info_bg": "#eef2ff",
        "info_border": "#818cf8",
        "warning_bg": "#fff7ed",
        "warning_border": "#fb923c",
        "grid": "#d0d7e3",
        "plot_bg": "#ffffff",
        "paper_bg": "#ffffff",
    }


# Get current palette dynamically
def get_palette():
    """Get current theme palette - call this instead of using PALETTE directly."""
    return get_theme_palette()


def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color string to rgba(...) string with given alpha."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def apply_chart_theme(fig, palette=None, height=400):
    """Apply consistent styling to Plotly charts."""
    if palette is None:
        palette = get_palette()
    
    # Use plot_bg and paper_bg if available, otherwise fall back to surface
    plot_bg = palette.get("plot_bg", palette["surface"])
    paper_bg = palette.get("paper_bg", palette["surface"])
    
    fig.update_layout(
        height=height,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(color=palette["text"], size=12),
        hovermode="x unified",
        legend=dict(
            bgcolor=paper_bg,
            bordercolor=palette["border"],
            borderwidth=1,
            font=dict(color=palette["text"]),
        ),
        margin=dict(l=20, r=20, t=60, b=40),
        title=dict(font=dict(color=palette["text"])),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=palette["grid"],
        linecolor=palette["border"],
        zeroline=False,
        showline=True,
        title=dict(font=dict(color=palette["text"])),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=palette["grid"],
        linecolor=palette["border"],
        zeroline=False,
        showline=True,
        title=dict(font=dict(color=palette["text"])),
    )
    return fig


def render_page_header(title, subtitle="", icon="monitoring"):
    """Render a page header with icon and subtitle."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header__icon material-symbols-rounded">{icon}</div>
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title, icon="insights"):
    """Render section heading with consistent styling."""
    st.markdown(
        f"""
        <div class="section-heading">
            <span class="material-symbols-rounded">{icon}</span>
            <span>{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Custom CSS with improved design - regenerated on each render to support theme switching
PALETTE = get_palette()  # Get current palette
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,0,0&display=swap');

    .stApp {{
        background-color: {PALETTE["background"]};
        color: {PALETTE["text"]};
        font-family: 'Inter', sans-serif;
    }}

    .block-container {{
        padding: 1rem 2.5rem 2.5rem 2.5rem;
        max-width: 1400px;
    }}

    .page-header {{
        background: {PALETTE["surface"]};
        border: 1px solid {PALETTE["border"]};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        gap: 1rem;
        align-items: center;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.08);
    }}

    .page-header h1 {{
        margin: 0;
        font-size: 1.75rem;
        color: {PALETTE["text"]};
    }}

    .page-header p {{
        margin: 0.2rem 0 0 0;
        color: {PALETTE["muted"]};
        font-size: 0.95rem;
    }}

    .page-header__icon {{
        font-size: 2.5rem;
        color: {PALETTE["accent"]};
    }}

    .section-heading {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: {PALETTE["muted"]};
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.75rem;
    }}

    .section-heading .material-symbols-rounded {{
        font-size: 1.35rem;
        color: {PALETTE["accent"]};
        background: {PALETTE["accent"]}1a;
        border-radius: 8px;
        padding: 0.15rem 0.35rem;
    }}

    .metric-card, [data-testid="stMetric"] {{
        background: {PALETTE["card"]};
        border-radius: 14px;
        border: 1px solid {PALETTE["border"]};
        box-shadow: 0 25px 45px rgba(15, 23, 42, 0.08);
        padding: 1.5rem;
    }}

    .stButton>button {{
        width: 100%;
        background: linear-gradient(120deg, {PALETTE["accent"]}, {PALETTE["accent_alt"]});
        color: #fff;
        border: none;
        padding: 0.85rem 1.25rem;
        border-radius: 12px;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 20px 35px rgba(37, 99, 235, 0.25);
    }}

    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 25px 45px rgba(37, 99, 235, 0.35);
    }}

    .sidebar-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid {PALETTE["border"]};
    }}

    .sidebar-header__icon {{
        font-size: 1.75rem;
        color: {PALETTE["accent"]};
    }}

    .sidebar-header h2 {{
        margin: 0;
        font-size: 1.1rem;
        color: {PALETTE["text"]};
    }}

    .sidebar-header span {{
        font-size: 0.85rem;
        color: {PALETTE["muted"]};
    }}

    .content-divider {{
        height: 2px;
        border: none;
        background: linear-gradient(90deg, {PALETTE["accent"]}, transparent);
        margin: 2rem 0;
    }}

    .info-box, .success-box, .warning-box {{
        background: {PALETTE["surface"]};
        border: 1px solid {PALETTE["border"]};
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }}

    .success-box {{
        background: {PALETTE["success_bg"]};
        border-color: {PALETTE["success_border"]};
    }}

    .info-box {{
        background: {PALETTE["info_bg"]};
        border-color: {PALETTE["info_border"]};
    }}

    .warning-box {{
        background: {PALETTE["warning_bg"]};
        border-color: {PALETTE["warning_border"]};
    }}

    .stAlert {{
        border-radius: 12px;
        border: 1px solid {PALETTE["border"]};
    }}

    .footer {{
        text-align: center;
        color: {PALETTE["muted"]};
        padding: 1.5rem 0 2rem 0;
        font-size: 0.9rem;
        margin-top: 2rem;
        border-top: 1px solid {PALETTE["border"]};
    }}

    .stDataFrame, .dataframe {{
        border: 1px solid {PALETTE["border"]};
        border-radius: 12px;
    }}

    .recommendation-list {{
        list-style: none;
        padding-left: 0;
        margin: 0;
    }}

    .recommendation-list li {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
        color: {PALETTE["text"]};
        font-weight: 500;
    }}

    .recommendation-list li .material-symbols-rounded {{
        font-size: 1rem;
        color: {PALETTE["accent"]};
        background: transparent;
        padding: 0;
    }}

    @media (max-width: 1200px) {{
        .block-container {{
            padding: 1rem;
        }}
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding: 0.5rem;
        }}
        .page-header {{
            flex-direction: column;
            align-items: flex-start;
        }}
        .page-header h1 {{
            font-size: 1.35rem;
        }}
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <span class="material-symbols-rounded sidebar-header__icon">donut_small</span>
        <div>
            <h2>Jagdamba fisheries Forecasting</h2>
            <span>Demand Intelligence Suite</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown('<hr class="content-divider">', unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox("Navigate", ["Dashboard", "Forecast Generator", "Data Analyzer", "Analytics"])

# Dashboard Page
if page == "Dashboard":
    render_page_header(
        "Jagdamba Fisheries Demand Forecasting Dashboard",
        "Monitor key demand signals and simulate forward-looking scenarios.",
        icon="leaderboard",
    )
    st.markdown('<hr class="content-divider">', unsafe_allow_html=True)

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

    st.markdown('<hr class="content-divider">', unsafe_allow_html=True)

    # Quick Forecast
    render_section_heading("Quick Forecast", icon="bolt")

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

                    # Display forecast chart - get fresh palette for current theme
                    current_palette = get_palette()
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=df_forecast["date"],
                            y=df_forecast["forecast"],
                            mode="lines",
                            name="Forecast",
                            line=dict(color=current_palette["accent"], width=3),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=df_forecast["date"],
                            y=df_forecast["upper_bound"],
                            mode="lines",
                            name="Upper Bound",
                            line=dict(color=current_palette["accent_alt"], width=1),
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
                            fillcolor=hex_to_rgba(current_palette["accent"], 0.12),
                            line=dict(color=current_palette["accent_alt"], width=1),
                            showlegend=False,
                        )
                    )

                    fig.update_layout(
                        title=f"Forecast for {selected_item} at {selected_center}",
                        xaxis_title="Date",
                        yaxis_title="Demand (kg)",
                    )

                    st.plotly_chart(apply_chart_theme(fig, height=500), width="stretch")

                    # Forecast summary
                    render_section_heading("Forecast Summary", icon="analytics")
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
        st.markdown('<hr class="content-divider">', unsafe_allow_html=True)
        render_section_heading("Historical Data Overview", icon="query_stats")

        col1, col2 = st.columns(2)

        with col1:
            # Demand by center
            if "CENTER NAME" in forecast_engine.data.columns and "PAY WEIGHT" in forecast_engine.data.columns:
                current_palette = get_palette()
                center_demand = forecast_engine.data.groupby("CENTER NAME")["PAY WEIGHT"].sum().reset_index()
                center_demand = center_demand.sort_values("PAY WEIGHT", ascending=False)

                fig_center = px.bar(
                    center_demand,
                    x="CENTER NAME",
                    y="PAY WEIGHT",
                    title="Total Demand by Center",
                    labels={"PAY WEIGHT": "Demand (kg)", "CENTER NAME": "Center"},
                    color_discrete_sequence=[current_palette["accent"]],
                )
                fig_center.update_traces(marker_line_color=current_palette["accent_alt"], marker_line_width=1)
                st.plotly_chart(apply_chart_theme(fig_center), width="stretch")

        with col2:
            # Demand by item
            if "ITEM" in forecast_engine.data.columns and "PAY WEIGHT" in forecast_engine.data.columns:
                current_palette = get_palette()
                item_demand = forecast_engine.data.groupby("ITEM")["PAY WEIGHT"].sum().reset_index()
                item_demand = item_demand.sort_values("PAY WEIGHT", ascending=False).head(10)

                fig_item = px.bar(
                    item_demand,
                    x="PAY WEIGHT",
                    y="ITEM",
                    orientation="h",
                    title="Top 10 Items by Demand",
                    labels={"PAY WEIGHT": "Demand (kg)", "ITEM": "Item"},
                    color_discrete_sequence=[current_palette["accent"]],
                )
                fig_item.update_traces(marker_line_color=current_palette["accent_alt"], marker_line_width=1)
                st.plotly_chart(apply_chart_theme(fig_item), width="stretch")

        if {"DATE", "PAY WEIGHT"}.issubset(forecast_engine.data.columns):
            render_section_heading("Weekly Demand Pattern", icon="calendar_month")
            seasonality_df = forecast_engine.data[["DATE", "PAY WEIGHT"]].copy()
            seasonality_df["DATE"] = pd.to_datetime(seasonality_df["DATE"], errors="coerce")
            seasonality_df = seasonality_df.dropna(subset=["DATE"])

            if not seasonality_df.empty:
                seasonality_df["Weekday"] = seasonality_df["DATE"].dt.day_name()
                weekday_order = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                weekday_summary = (
                    seasonality_df.groupby("Weekday")["PAY WEIGHT"]
                    .mean()
                    .reindex(weekday_order)
                    .dropna()
                    .reset_index()
                )

                if not weekday_summary.empty:
                    current_palette = get_palette()
                    weekday_summary = weekday_summary.rename(columns={"PAY WEIGHT": "Average Demand"})
                    fig_weekday = px.line(
                        weekday_summary,
                        x="Weekday",
                        y="Average Demand",
                        title="Average Demand by Weekday",
                        markers=True,
                        color_discrete_sequence=[current_palette["accent"]],
                    )
                    st.plotly_chart(apply_chart_theme(fig_weekday, height=380), width="stretch")

# Forecast Generator Page
elif page == "Forecast Generator":
    render_page_header(
        "Forecast Generator",
        "Configure tailored demand simulations by location, category, and model.",
        icon="precision_manufacturing",
    )
    st.markdown('<hr class="content-divider">', unsafe_allow_html=True)

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
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info(
            f"""
        **Selected Configuration:**
        - Centers: {len(selected_centers)}
        - Items: {len(selected_items)}
        - Forecast Period: {forecast_days} days
        - Model: {model_type}
        """
        )
        st.markdown('</div>', unsafe_allow_html=True)

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
                        render_section_heading(f"{center}", icon="location_on")

                        for item in selected_items:
                            if center in forecasts and item in forecasts[center]:
                                forecast_data = forecasts[center][item]
                                df_forecast = pd.DataFrame(forecast_data)
                                df_forecast["date"] = pd.to_datetime(df_forecast["date"])

                                with st.expander(f"{item}"):
                                    current_palette = get_palette()
                                    fig = go.Figure()

                                    fig.add_trace(
                                        go.Scatter(
                                            x=df_forecast["date"],
                                            y=df_forecast["forecast"],
                                            mode="lines+markers",
                                            name="Forecast",
                                            line=dict(color=current_palette["accent"], width=2),
                                        )
                                    )

                                    fig.add_trace(
                                        go.Scatter(
                                            x=df_forecast["date"],
                                            y=df_forecast["upper_bound"],
                                            mode="lines",
                                            name="Upper Bound",
                                            line=dict(color=current_palette["accent_alt"], dash="dash"),
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
                                            fillcolor=hex_to_rgba(current_palette["accent"], 0.12),
                                            line=dict(color=current_palette["accent_alt"], dash="dash"),
                                            showlegend=True,
                                        )
                                    )

                                    fig.update_layout(
                                        title=f"Forecast for {item}",
                                        xaxis_title="Date",
                                        yaxis_title="Demand (kg)",
                                    )

                                    st.plotly_chart(apply_chart_theme(fig), width="stretch")

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
    render_page_header(
        "Data Analyzer",
        "Explore uploaded operational data and activate forecasting workflows.",
        icon="database",
    )
    st.markdown('<hr class="content-divider">', unsafe_allow_html=True)

    render_section_heading("Upload Data for Analysis", icon="cloud_upload")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read()
            analysis = forecast_engine.analyze_uploaded_data(content)

            if analysis.get("status") == "success":
                uploaded_df = analysis.pop("dataframe", None)
                st.session_state["uploaded_dataset"] = uploaded_df
                st.session_state.pop("uploaded_forecast_df", None)

                st.success("Data analyzed successfully.")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Records", analysis.get("total_records", 0))

                    if analysis.get("date_range"):
                        st.markdown(
                            f"""
                            <div class="info-box">
                                <strong>Date Range</strong><br/>
                                {analysis['date_range']['start']} to {analysis['date_range']['end']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if analysis.get("centers"):
                        st.write(f"Locations detected ({len(analysis['centers'])}):")
                        st.write(analysis["centers"][:10])

                with col2:
                    if analysis.get("total_demand", 0) > 0:
                        st.metric("Total Demand", f"{analysis['total_demand']:,.0f} kg")

                    if analysis.get("products"):
                        st.write(f"Products detected ({len(analysis['products'])}):")
                        st.write(analysis["products"][:10])

                recommendations = analysis.get("recommendations", [])
                if recommendations:
                    render_section_heading("Recommendations", icon="task_alt")
                    items = "".join(
                        [
                            f"<li><span class='material-symbols-rounded'>task_alt</span>{rec}</li>"
                            for rec in recommendations
                        ]
                    )
                    st.markdown(f"<ul class='recommendation-list'>{items}</ul>", unsafe_allow_html=True)

                if uploaded_df is not None and not uploaded_df.empty:
                    render_section_heading("Uploaded Data Preview", icon="table_chart")
                    st.dataframe(uploaded_df.head(200), width="stretch", height=300)

                st.markdown('<hr class="content-divider">', unsafe_allow_html=True)
                render_section_heading("Generate Forecast from Uploaded Data", icon="show_chart")

                forecast_months = st.slider(
                    "Forecast Months",
                    min_value=1,
                    max_value=24,
                    value=12,
                    key="uploaded_forecast_months",
                )

                if st.button("Generate Next Year Forecast", key="uploaded_forecast_btn"):
                    data_source = uploaded_df
                    if (data_source is None or data_source.empty) and "uploaded_dataset" in st.session_state:
                        data_source = st.session_state.get("uploaded_dataset")

                    if data_source is None or data_source.empty:
                        st.warning("The uploaded dataset is empty or unreadable. Please upload a valid CSV file.")
                    else:
                        with st.spinner("Generating forecast from uploaded data..."):
                            forecast_records = forecast_engine.generate_forecast_from_uploaded_data(
                                data_source, forecast_months
                            )

                            if forecast_records:
                                forecast_df = pd.DataFrame(forecast_records)
                                st.session_state["uploaded_forecast_df"] = forecast_df
                                st.success("Forecast generated successfully.")
                                st.info(f"Created a {forecast_months}-month projection based on the uploaded signals.")
                            else:
                                st.warning("Could not derive forecast due to missing date or demand columns.")

                forecast_df = st.session_state.get("uploaded_forecast_df")
                if forecast_df is not None and not forecast_df.empty:
                    st.markdown('<hr class="content-divider">', unsafe_allow_html=True)
                    render_section_heading("Forecast Output", icon="stacked_line_chart")

                    col_sel1, col_sel2 = st.columns(2)
                    with col_sel1:
                        selected_center = st.selectbox(
                            "Select Location",
                            sorted(forecast_df["Center"].unique()),
                            key="uploaded_center_select",
                        )
                    with col_sel2:
                        filtered_items = sorted(
                            forecast_df[forecast_df["Center"] == selected_center]["Item"].unique()
                        )
                        selected_item = st.selectbox(
                            "Select Product",
                            filtered_items,
                            key="uploaded_item_select",
                        )

                    filtered_df = forecast_df[
                        (forecast_df["Center"] == selected_center) & (forecast_df["Item"] == selected_item)
                    ].copy()

                    if not filtered_df.empty:
                        current_palette = get_palette()
                        filtered_df["MonthStart"] = pd.to_datetime(filtered_df["Month"])

                        fig_uploaded = go.Figure()
                        fig_uploaded.add_trace(
                            go.Scatter(
                                x=filtered_df["MonthStart"],
                                y=filtered_df["Forecast"],
                                name="Forecast",
                                mode="lines+markers",
                                line=dict(color=current_palette["accent"], width=3),
                            )
                        )
                        fig_uploaded.add_trace(
                            go.Scatter(
                                x=filtered_df["MonthStart"],
                                y=filtered_df["UpperBound"],
                                name="Upper Bound",
                                line=dict(color=current_palette["accent_alt"], dash="dot"),
                            )
                        )
                        fig_uploaded.add_trace(
                            go.Scatter(
                                x=filtered_df["MonthStart"],
                                y=filtered_df["LowerBound"],
                                name="Lower Bound",
                                fill="tonexty",
                                fillcolor=hex_to_rgba(current_palette["accent"], 0.12),
                                line=dict(color=current_palette["accent_alt"], dash="dot"),
                            )
                        )
                        fig_uploaded.update_layout(
                            title=f"{selected_item} forecast for {selected_center}",
                            xaxis_title="Month",
                            yaxis_title="Demand (kg)",
                        )
                        st.plotly_chart(apply_chart_theme(fig_uploaded), width="stretch")

                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        with summary_col1:
                            st.metric("Average Forecast", f"{filtered_df['Forecast'].mean():,.0f} kg")
                        with summary_col2:
                            st.metric("Peak Projection", f"{filtered_df['UpperBound'].max():,.0f} kg")
                        with summary_col3:
                            st.metric("Lowest Projection", f"{filtered_df['LowerBound'].min():,.0f} kg")

                        st.dataframe(
                            filtered_df[["Month", "Forecast", "LowerBound", "UpperBound"]],
                            width="stretch",
                        )

                        st.download_button(
                            label="Download Forecast CSV",
                            data=filtered_df.to_csv(index=False),
                            file_name=f"uploaded_forecast_{selected_center}_{selected_item}.csv",
                            mime="text/csv",
                        )
            else:
                st.session_state.pop("uploaded_dataset", None)
                st.session_state.pop("uploaded_forecast_df", None)
                st.error(f"Analysis failed: {analysis.get('error', 'Unknown error')}")
        except Exception as e:
            st.session_state.pop("uploaded_dataset", None)
            st.session_state.pop("uploaded_forecast_df", None)
            st.error(f"Error analyzing file: {str(e)}")
    else:
        st.session_state.pop("uploaded_dataset", None)
        st.session_state.pop("uploaded_forecast_df", None)
        st.info("Please upload a CSV file to analyze.")

# Analytics Page
elif page == "Analytics":
    render_page_header(
        "Analytics",
        "Deep dive into historical performance by month, location, and species.",
        icon="insights",
    )
    st.markdown('<hr class="content-divider">', unsafe_allow_html=True)

    if forecast_engine.data is not None and not forecast_engine.data.empty:
        st.subheader("Data Insights")

        # Time series analysis
        if "DATE" in forecast_engine.data.columns and "PAY WEIGHT" in forecast_engine.data.columns:
            forecast_engine.data["DATE"] = pd.to_datetime(forecast_engine.data["DATE"])

            # Monthly trends
            current_palette = get_palette()
            forecast_engine.data["Month"] = forecast_engine.data["DATE"].dt.to_period("M")
            monthly_demand = forecast_engine.data.groupby("Month")["PAY WEIGHT"].sum().reset_index()
            monthly_demand["Month"] = monthly_demand["Month"].astype(str)

            fig_monthly = px.line(
                monthly_demand,
                x="Month",
                y="PAY WEIGHT",
                title="Monthly Demand Trends",
                labels={"PAY WEIGHT": "Demand (kg)", "Month": "Month"},
                color_discrete_sequence=[current_palette["accent"]],
            )
            st.plotly_chart(apply_chart_theme(fig_monthly), width="stretch")

            # Center-Item Matrix
            if "CENTER NAME" in forecast_engine.data.columns and "ITEM" in forecast_engine.data.columns:
                pivot_data = forecast_engine.data.groupby(["CENTER NAME", "ITEM"])["PAY WEIGHT"].sum().reset_index()
                pivot_table = pivot_data.pivot(index="CENTER NAME", columns="ITEM", values="PAY WEIGHT").fillna(0)

                current_palette = get_palette()
                st.subheader("Center-Item Demand Matrix")
                fig_heatmap = px.imshow(
                    pivot_table,
                    labels=dict(x="Item", y="Center", color="Demand (kg)"),
                    title="Demand Heatmap: Center vs Item",
                    aspect="auto",
                    color_continuous_scale=[
                        [0.0, current_palette["surface"]],
                        [0.5, current_palette["accent_alt"]],
                        [1.0, current_palette["accent"]],
                    ],
                )
                st.plotly_chart(apply_chart_theme(fig_heatmap, height=600), width="stretch")

                # Display pivot table
                st.subheader("Demand Table")
                st.dataframe(pivot_table, width="stretch")
    else:
        st.warning("No historical data available for analytics.")

# Footer
st.markdown('<hr class="content-divider">', unsafe_allow_html=True)
st.markdown(
    """
<div class="footer">
    <p>Jagdamba Fisheries Demand Forecasting System v3.0 | Built By Susmit Naik</p>
</div>
""",
    unsafe_allow_html=True,
)