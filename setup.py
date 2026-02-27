import os
import sys
import shutil
import subprocess
from pathlib import Path

OVERWRITE = True

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
    "pipelines/train.py": "",
    "pipelines/backtest.py": "",
    "pipelines/realtime.py": "",
    "dashboard/app.py": "",
    "src/data/fetcher.py": "",
    "src/data/preprocess.py": "",
    "src/features/engineer.py": "",
    "src/models/trainer.py": "",
    "src/models/predictor.py": "",
    "src/models/registry.py": "",
    "src/evaluation/metrics.py": "",
    "src/utils/logger.py": "",
    "src/utils/paths.py": "",
    "src/utils/config_loader.py": "",
    "src/pipeline/train_pipeline.py": "",
    "src/pipeline/backtest_pipeline.py": "",
    "src/pipeline/realtime_pipeline.py": "",
    "config/data.yaml": "source: yfinance",
    "config/model.yaml": "model: RandomForest",
    "config/training.yaml": "epochs: 10",
    "scripts/setup.py": "",
    "README.md": "",
    ".gitignore": "",
    "requirements.txt": ""
}

PACKAGES = [
    "yfinance",
    "pandas",
    "numpy",
    "scikit-learn",
    "mlflow",
    "streamlit",
    "plotly",
    "matplotlib",
    "seaborn",
    "pyyaml"
]

def recreate_folders():
    for folder in FOLDERS:
        path = Path(folder)
        if path.exists() and OVERWRITE:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

def recreate_files():
    for file, content in FILES.items():
        path = Path(file)
        if path.exists() and OVERWRITE:
            path.unlink()
        path.write_text(content)

def create_venv():
    venv_path = Path("venv")
    if venv_path.exists() and OVERWRITE:
        shutil.rmtree(venv_path)
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    return "source venv/bin/activate" if os.name != "nt" else r"venv\Scripts\activate.bat"

def install_dependencies():
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install"] + PACKAGES, check=True)

def main():
    print("===== Setting Up Project =====")
    recreate_folders()
    recreate_files()
    activate_cmd = create_venv()
    install_dependencies()
    print("===== Setup Complete =====")
    print(f"Activate environment:\n  {activate_cmd}")

if __name__ == "__main__":
    main()