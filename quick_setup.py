#!/usr/bin/env python3
"""
Quick setup for Seafood Demand Forecasting
"""

import os
import shutil

def create_file(path, content):
    """Create a file with given content"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created {path}")

def main():
    print("🚀 Quick Setup for Seafood Demand Forecasting")
    print("="*50)
    
    # Create directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models/saved_models",
        "tests",
        "monitoring",
        "config",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created {directory}/")
    
    # Create __init__.py files
    init_files = ["models/__init__.py", "scripts/__init__.py", "app/__init__.py", "monitoring/__init__.py"]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# Package initialization\n")
        print(f"📄 Created {init_file}")
    
    print("\n✅ Setup completed!")
    print("\n📋 Next steps:")
    print("1. Place your CSV file at: data/raw/Production_1_Cleaned_Expanded.csv")
    print("2. Run: python scripts/data_pipeline.py")
    print("3. Run: python scripts/train_model.py") 
    print("4. Run: python scripts/deploy_model.py")
    print("5. Run: streamlit run app/dashboard.py")
    print("6. Run: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("\n🌐 Streamlit Dashboard: http://localhost:8501")
    print("🌐 FastAPI API: http://localhost:8000/docs")

if __name__ == "__main__":
    main()