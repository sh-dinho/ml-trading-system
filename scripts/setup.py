import os
import sys
import subprocess
from pathlib import Path

CREATE_VENV = True
INSTALL_DEPENDENCIES = True

FOLDERS = [
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
    "tests"
]

FILES = {
    "pipelines/train.py": "# Training pipeline entrypoint\n",
    "pipelines/backtest.py": "# Backtesting pipeline entrypoint\n",
    "pipelines/realtime.py": "# Realtime trading pipeline entrypoint\n",
    "dashboard/app.py": "# Streamlit dashboard\n",
    "src/data/fetcher.py": "# Data fetching logic\n",
    "src/data/preprocess.py": "# Data preprocessing logic\n",
    "src/features/engineer.py": "# Feature engineering orchestrator\n",
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
    "scripts/setup.py": "# Additional setup scripts\n",
    "README.md": "# ML Trading System\n",
    ".gitignore": "*.pyc\n__pycache__/\nvenv/\n",
    "requirements.txt": "yfinance\npandas\nnumpy\nscikit-learn\nmlflow\nstreamlit\nplotly\nmatplotlib\nseaborn\npyyaml\n"
}

def create_folders():
    for folder in FOLDERS:
        path = Path(folder)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"[+] Created folder: {folder}")
        else:
            print(f"[=] Folder exists: {folder}")

def create_files():
    for file, content in FILES.items():
        path = Path(file)
        if not path.exists():
            path.write_text(content)
            print(f"[+] Created file: {file}")
        else:
            print(f"[=] File exists: {file}")

def create_venv():
    if not CREATE_VENV:
        return None

    venv_path = Path("venv")
    if not venv_path.exists():
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("[+] Virtual environment created")
    else:
        print("[=] Virtual environment already exists")

    return "venv\\Scripts\\activate.bat" if os.name == "nt" else "source venv/bin/activate"

def install_dependencies():
    if not INSTALL_DEPENDENCIES:
        return

    print("[*] Upgrading pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print("[*] Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def main():
    print("===== Initializing Project Structure =====")
    create_folders()
    create_files()

    print("\n===== Environment Setup =====")
    activate_cmd = create_venv()
    install_dependencies()

    print("\n===== Initialization Complete =====")
    if activate_cmd:
        print(f"Activate your environment:\n  {activate_cmd}")

if __name__ == "__main__":
    main()