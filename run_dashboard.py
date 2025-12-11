#!/usr/bin/env python3
"""
Simple script to run the Streamlit dashboard from project root
"""

import os
import sys
import subprocess

# Get the directory where this script is located (project root)
project_root = os.path.dirname(os.path.abspath(__file__))

# Change to project root directory
os.chdir(project_root)

# Add project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Run Streamlit
if __name__ == "__main__":
    dashboard_path = os.path.join(project_root, "app", "dashboard.py")
    print(f"Starting Streamlit dashboard from: {project_root}")
    print(f"Dashboard will be available at: http://localhost:8501")
    print("-" * 60)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path
    ])

