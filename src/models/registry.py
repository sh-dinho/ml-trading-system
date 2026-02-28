from pathlib import Path
from typing import Any, Optional
import joblib
import datetime


class ModelRegistry:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def save_model(self, model: Any, name: str, with_timestamp: bool = True) -> Path:
        if with_timestamp:
            stem = Path(name).stem
            suffix = Path(name).suffix or ".pkl"
            filename = f"{stem}_{self._timestamp()}{suffix}"
        else:
            filename = name

        path = self.model_dir / filename
        joblib.dump(model, path)
        return path

    def load_model(self, path: Optional[str] = None, latest: bool = False) -> Any:
        if latest:
            candidates = sorted(self.model_dir.glob("*.pkl"))
            if not candidates:
                raise FileNotFoundError("No models found in registry.")
            path = candidates[-1]
        elif path is not None:
            path = Path(path)
        else:
            raise ValueError("Either `path` must be provided or `latest=True`.")

        return joblib.load(path)