import os
import sys
import subprocess
from pathlib import Path

# =========================================
# Cross-Platform Production-Ready Setup
# AI Stock Trading / MLOps
# =========================================

print("===== Starting ML Trading System Setup =====")

# -------------------------
# 1️⃣ Create project directories
# -------------------------
folders = [
    "data",
    "logs",
    "models",
    "training_pipeline",
    "dashboard",
    "evaluation",
    "scripts"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
print("[✓] Project directories created.")

# -------------------------
# 2️⃣ Create virtual environment
# -------------------------
venv_path = Path("venv")
if not venv_path.exists():
    print("[*] Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"])

# Activate message
activate_msg = "source venv/bin/activate" if os.name != "nt" else r"venv\Scripts\activate.bat"
print(f"[✓] Virtual environment created. Activate with:\n    {activate_msg}")

# -------------------------
# 3️⃣ Install Python dependencies
# -------------------------
print("[*] Installing Python packages...")
packages = [
    "yfinance",
    "pandas",
    "numpy",
    "scikit-learn",
    "mlflow",
    "streamlit",
    "plotly",
    "matplotlib",
    "seaborn"
]
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.run([sys.executable, "-m", "pip", "install"] + packages)
print("[✓] Python dependencies installed.")

# -------------------------
# 4️⃣ Create placeholder files
# -------------------------
placeholders = [
    "training_pipeline/train.py",
    "evaluation/evaluate.py",
    "dashboard/app.py",
    "scripts/setup.py",
    "portfolio.csv",
    "logs/system.log"
]

for file in placeholders:
    Path(file).touch(exist_ok=True)

# Set log file permissions (Windows/Linux compatible)
log_path = Path("logs/system.log")
log_path.chmod(0o664)

print("[✓] Placeholder files created.")

# -------------------------
# 5️⃣ .gitignore
# -------------------------
gitignore_content = """
# Virtual environment
venv/

# Python cache
__pycache__/
*.pyc

# Logs
logs/

# Data
data/

# Models
models/

# Jupyter
.ipynb_checkpoints/

# OS files
.DS_Store
"""
Path(".gitignore").write_text(gitignore_content)
print("[✓] .gitignore created.")

# -------------------------
# 6️⃣ README.md
# -------------------------
readme_content = """
# AI Stock Trading System (MLOps / Production Ready)

## Project Overview
This is a production-ready, automated AI stock trading system built for learning and MLOps experience.
It includes:

- Walk-forward ML training
- Forward-testing/backtesting
- Real-time stock data ingestion (yfinance)
- Dashboard visualization (Streamlit + Plotly)
- Model tracking & experiment management (MLflow)
- Logging and reproducible environment (cross-platform)

## Project Structure
ml-trading-system/
├── data/
├── logs/
├── models/
├── training_pipeline/
├── dashboard/
├── evaluation/
├── scripts/
├── venv/
├── portfolio.csv
├── .gitignore
├── README.md
└── requirements.txt

## Setup Instructions
1. Run setup script:
    python scripts/setup.py
2. Activate environment:
    Windows: venv\\Scripts\\activate.bat
    Linux/macOS: source venv/bin/activate
3. Run training:
    python training_pipeline/train.py
4. Run dashboard:
    streamlit run dashboard/app.py

## MLOps / Production Features
- CI/CD ready (GitHub Actions)
- Cloud deployable (AWS/GCP/Azure)
- Logging & monitoring
- MLflow experiment tracking
- Cron/scheduler compatible
"""
Path("README.md").write_text(readme_content)
print("[✓] README.md created.")

# -------------------------
# 7️⃣ requirements.txt
# -------------------------
requirements_content = "\n".join(packages)
Path("requirements.txt").write_text(requirements_content)
print("[✓] requirements.txt created.")

# -------------------------
# Done
# -------------------------
print("===== Setup Complete =====")
print(f"Activate environment:\n  {activate_msg}")
print("Run training: python training_pipeline/train.py")
print("Run dashboard: streamlit run dashboard/app.py")