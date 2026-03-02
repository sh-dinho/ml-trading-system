from pathlib import Path
import joblib
from datetime import datetime


class ModelRegistry:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, name: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{name}"
        path = self.model_dir / filename
        joblib.dump(model, path)
        return str(path)

    def latest_model_path(self) -> str:
        files = sorted(self.model_dir.glob("*.pkl"))
        if not files:
            raise FileNotFoundError("No models found in registry.")
        return str(files[-1])