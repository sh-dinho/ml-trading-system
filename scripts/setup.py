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
from pathlib import Path
import argparse
from typing import List, Dict


class ProjectInitializer:
    def __init__(self, force: bool = False):
        self.force = force

        # -----------------------------
        # Single source of truth
        # -----------------------------
        self.dependencies: List[str] = [
            # Data ingestion
            "yfinance",
            "pandas",
            "numpy",

            # Feature engineering
            "ta",

            # Machine learning
            "scikit-learn",
            "xgboost",

            # Experiment tracking
            "mlflow",

            # Visualization
            "streamlit",
            "plotly",
            "matplotlib",
            "seaborn",

            # Utilities
            "pyyaml",
            "joblib",
        ]

        self.folders: List[str] = [
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

        self.files: Dict[str, str] = {
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

    # -----------------------------
    # Logging
    # -----------------------------
    def log(self, msg: str):
        print(f"[INIT] {msg}")

    # -----------------------------
    # Folder & File Creation
    # -----------------------------
    def create_folders(self):
        for folder in self.folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
            self.log(f"Ensured folder: {folder}")

    def create_files(self):
        # Auto-generate requirements.txt from dependencies
        self.files["requirements.txt"] = "\n".join(self.dependencies) + "\n"

        for file_path, content in self.files.items():
            path = Path(file_path)
            if path.exists() and not self.force:
                self.log(f"File exists (skipped): {file_path}")
                continue
            path.write_text(content)
            self.log(f"Created file: {file_path}")

    def create_init_files(self):
        for root, dirs, _ in os.walk("src"):
            init_path = Path(root) / "__init__.py"
            if not init_path.exists():
                init_path.write_text("")
                self.log(f"Created package file: {init_path}")

    # -----------------------------
    # Virtual Environment
    # -----------------------------
    def get_venv_python(self) -> Path:
        return Path("venv/Scripts/python.exe") if os.name == "nt" else Path("venv/bin/python")

    def create_venv(self):
        venv_path = Path("venv")
        if venv_path.exists() and not self.force:
            self.log("Virtual environment already exists.")
            return

        self.log("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        self.log("Virtual environment created.")

    def install_dependencies(self):
        pip_python = self.get_venv_python()

        if not pip_python.exists():
            self.log("Virtual environment not found. Skipping dependency install.")
            return

        try:
            self.log("Upgrading pip...")
            subprocess.run([pip_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)

            self.log("Installing dependencies...")
            subprocess.run([pip_python, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

            self.log("Dependencies installed successfully.")
        except subprocess.CalledProcessError:
            self.log("Dependency installation failed.")
            sys.exit(1)

    # -----------------------------
    # Activation Message
    # -----------------------------
    def print_activation(self):
        cmd = "venv\\Scripts\\activate" if os.name == "nt" else "source venv/bin/activate"
        print("\nActivate environment with:")
        print(f"  {cmd}")

    # -----------------------------
    # Main Orchestration
    # -----------------------------
    def initialize_project(self, skip_venv=False, skip_install=False):
        self.log("Initializing project structure...")
        self.create_folders()
        self.create_files()
        self.create_init_files()

        if not skip_venv:
            self.create_venv()

        if not skip_install:
            self.install_dependencies()

        self.log("Initialization complete.")
        self.print_activation()


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Initialize ML Trading Project")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--no-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files/venv")

    args = parser.parse_args()

    initializer = ProjectInitializer(force=args.force)
    initializer.initialize_project(skip_venv=args.no_venv, skip_install=args.no_install)


if __name__ == "__main__":
    main()