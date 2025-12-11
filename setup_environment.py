import os
import sys
import subprocess
import importlib

def check_and_install_packages():
    """Check and install required packages"""
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn',
        'xgboost', 'lightgbm', 'prophet', 'statsmodels', 'mlflow',
        'plotly', 'streamlit', 'sqlalchemy', 'jinja2', 'pytest', 'pyyaml', 'joblib'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'models/saved_models',
        'app/api',
        'tests',
        'monitoring',
        'config',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for packages"""
    packages = ['models', 'scripts', 'app', 'monitoring']
    
    for package in packages:
        init_file = os.path.join(package, '__init__.py')
        with open(init_file, 'w') as f:
            if package == 'models':
                f.write('from .model_registry import ModelRegistry\n\n__all__ = ["ModelRegistry"]')
            elif package == 'scripts':
                f.write('from .data_pipeline import DataPipeline\nfrom .train_model import ModelTrainer\nfrom .deploy_model import ModelDeployer\nfrom .utils import ForecastEngine\n\n__all__ = ["DataPipeline", "ModelTrainer", "ModelDeployer", "ForecastEngine"]')
            elif package == 'monitoring':
                f.write('from .drift_detection import DataDriftDetector\n\n__all__ = ["DataDriftDetector"]')
            else:
                f.write('# Package initialization')
        print(f"Created {init_file}")

if __name__ == "__main__":
    print("Setting up seafood forecasting environment...")
    
    print("\n1. Checking and installing packages...")
    check_and_install_packages()
    
    print("\n2. Creating directory structure...")
    create_directory_structure()
    
    print("\n3. Creating package initialization files...")
    create_init_files()
    
    print("\n✓ Environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python scripts/data_pipeline.py")
    print("2. Run: python scripts/train_model.py") 
    print("3. Run: streamlit run app/dashboard.py")
    print("4. Run: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("\n🌐 Streamlit Dashboard: http://localhost:8501")
    print("🌐 FastAPI API: http://localhost:8000/docs")