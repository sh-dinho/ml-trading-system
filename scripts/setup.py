#!/usr/bin/env python3
"""
Production ML Trading System Initializer
-----------------------------------------
Creates project structure, virtual environment,
installs dependencies, and prepares package layout.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List


# =========================
# CONFIGURATION
# =========================

FOLDERS: List[str] = [
    "config",
    "data/raw",
    "data/processed",
    "src/data",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/utils",
    "src/pipeline",
    "pipelines",
    "dashboard",
    "models",
    "logs",
    "scripts",
    "tests",
]

FILES: Dict[str, str] = {
    "pipelines/train.py": "# Training pipeline entrypoint\n",
    "pipelines/backtest.py": "# Backtesting pipeline entrypoint\n",
    "pipelines/realtime.py": "# Realtime trading pipeline entrypoint\n",
    "dashboard/app.py": "# Streamlit dashboard\n",
    "src/data/fetcher.py": "# Data fetching logic\n",
    "src/data/preprocess.py": "# Data preprocessing logic\n",
    "src/features/engineer.py": "# Feature engineering logic\n",
    "src/models/trainer.py": "# Model training logic\n",
    "src/models/predictor.py": "# Model inference logic\n",
    "src/models/registry.py": "# Model registry (MLflow or local)\n",
    "src/evaluation/metrics.py": "# Evaluation metrics\n",
    "src/utils/logger.py": "# Logging utilities\n",
    "src/utils/paths.py": "# Path utilities\n",
    "src/utils/config_loader.py": "# YAML config loader\n",
    "src/pipeline/train_pipeline.py": "# High-level training pipeline\n",
    "src/pipeline/backtest_pipeline.py": "# High-level backtest pipeline\n",
    "src/pipeline/realtime_pipeline.py": "# High-level realtime pipeline\n",
    "config/data.yaml": "source: yfinance\n",
    "config/model.yaml": "model: RandomForest\n",
    "config/training.yaml": "epochs: 10\n",
    "README.md": "# ML Trading System\n",
    "requirements.txt": """yfinance
pandas
numpy
scikit-learn
mlflow
streamlit
plotly
matplotlib
seaborn
pyyaml
""",
    ".gitignore": """# Python
__pycache__/
*.pyc

# Virtual env
venv/

# Data
data/raw/
data/processed/

# Models
models/
mlruns/

# Logs
logs/

# OS
.DS_Store
Thumbs.db
""",
}


# =========================
# UTILS
# =========================

def log(msg: str):
    print(f"[INIT] {msg}")


def create_folders():
    for folder in FOLDERS:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        log(f"Ensured folder: {folder}")


def create_files():
    for file_path, content in FILES.items():
        path = Path(file_path)
        if not path.exists():
            path.write_text(content)
            log(f"Created file: {file_path}")
        else:
            log(f"File exists: {file_path}")


def create_init_files():
    for root, dirs, files in os.walk("src"):
        init_path = Path(root) / "__init__.py"
        if not init_path.exists():
            init_path.write_text("")
            log(f"Created package file: {init_path}")


def get_venv_python():
    if os.name == "nt":
        return Path("venv/Scripts/python.exe")
    return Path("venv/bin/python")


def create_venv():
    venv_path = Path("venv")
    if not venv_path.exists():
        log("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        log("Virtual environment created.")
    else:
        log("Virtual environment already exists.")


def install_dependencies():
    pip_python = get_venv_python()

    if not pip_python.exists():
        log("Virtual environment not found. Skipping dependency install.")
        return

    try:
        log("Upgrading pip...")
        subprocess.run([pip_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)

        log("Installing dependencies...")
        subprocess.run([pip_python, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

        log("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        log("Dependency installation failed.")
        sys.exit(1)


def print_activation():
    if os.name == "nt":
        cmd = "venv\\Scripts\\activate"
    else:
        cmd = "source venv/bin/activate"

    print("\nActivate environment with:")
    print(f"  {cmd}")


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Initialize ML Trading Project")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--no-install", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()

    log("Initializing project structure...")
    create_folders()
    create_files()
    create_init_files()

    if not args.no_venv:
        create_venv()

    if not args.no_install:
        install_dependencies()

    log("Initialization complete.")
    print_activation()


if __name__ == "__main__":
    main()