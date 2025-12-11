#!/usr/bin/env python3
"""
Quick start script for Seafood Demand Forecasting
"""

import os
import sys
import subprocess
import webbrowser
from datetime import datetime

def run_step(description, command):
    """Run a step in the pipeline"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    try:
        if command.endswith('.py'):
            # Run Python script
            result = subprocess.run([sys.executable, command], check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                # Filter out Unicode errors from stderr
                filtered_stderr = [line for line in result.stderr.split('\n') if 'UnicodeEncodeError' not in line]
                if filtered_stderr:
                    print(f"STDERR: {' '.join(filtered_stderr)}")
        else:
            # Run shell command
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                filtered_stderr = [line for line in result.stderr.split('\n') if 'UnicodeEncodeError' not in line]
                if filtered_stderr:
                    print(f"STDERR: {' '.join(filtered_stderr)}")
        
        print(f"[SUCCESS] {description} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error in {description}:")
        # Filter Unicode errors from error output
        filtered_stderr = [line for line in e.stderr.split('\n') if 'UnicodeEncodeError' not in line]
        if filtered_stderr:
            print(f"Error output: {' '.join(filtered_stderr)}")
        print(f"Stdout: {e.stdout}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error in {description}: {e}")
        return False

def main():
    print("Seafood Demand Forecasting - Quick Start")
    print("="*50)
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models/saved_models",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists("data/raw/Production_1_Cleaned_Expanded.csv"):
        print("\n[ERROR] Please place your CSV file at: data/raw/Production_1_Cleaned_Expanded.csv")
        print("Then run this script again.")
        return
    
    # Run pipeline steps
    steps = [
        ("Data Processing", "scripts/data_pipeline.py"),
        ("Model Training", "scripts/train_model.py"),
    ]
    
    for step_name, step_command in steps:
        success = run_step(step_name, step_command)
        if not success:
            print(f"\n[ERROR] Pipeline failed at: {step_name}")
            return
    
    # Start the web application
    print(f"\n{'='*60}")
    print("Starting Web Application...")
    print(f"{'='*60}")
    
    try:
        # Start the FastAPI server
        import uvicorn
        
        print("[INFO] Starting services...")
        print("[INFO] FastAPI API will be available at: http://localhost:8000")
        print("[INFO] API documentation at: http://localhost:8000/docs")
        print("[INFO] Streamlit Dashboard at: http://localhost:8501")
        print("[INFO] Press Ctrl+C to stop the servers")
        print("\n[INFO] To start Streamlit dashboard separately, run: streamlit run app/dashboard.py")
        
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] Error starting web server: {e}")

if __name__ == "__main__":
    main()