from pathlib import Path
import datetime
import joblib
import json
from typing import Any, Optional, Dict


class ModelRegistry:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp(self):
        return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # -----------------------------
    # Save model + metadata
    # -----------------------------
    def save_model(
        self,
        model: Any,
        name: str,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        with_timestamp: bool = True
    ) -> Path:

        stem = Path(name).stem
        suffix = Path(name).suffix or ".pkl"

        if with_timestamp:
            filename = f"{stem}_{self._timestamp()}{suffix}"
        else:
            filename = f"{stem}{suffix}"

        model_path = self.model_dir / filename
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "model_file": filename,
            "timestamp": self._timestamp(),
            "metrics": metrics or {},
            "config": config or {},
        }

        meta_path = self.model_dir / f"{stem}_{self._timestamp()}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return model_path

    # -----------------------------
    # Load model
    # -----------------------------
    def load_model(self, path: Optional[str] = None, latest: bool = False) -> Any:
        if latest:
            candidates = sorted(self.model_dir.glob("*.pkl"))
            if not candidates:
                raise FileNotFoundError("No models found in registry.")
            path = candidates[-1]

        if path is None:
            raise ValueError("Provide a path or set latest=True.")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        return joblib.load(path)

    # -----------------------------
    # List all models
    # -----------------------------
    def list_models(self):
        return sorted(self.model_dir.glob("*.pkl"))

    # -----------------------------
    # Load metadata for a model
    # -----------------------------
    def load_metadata(self, model_path: Path) -> Dict[str, Any]:
        stem = model_path.stem.split("_")[0]
        meta_files = sorted(self.model_dir.glob(f"{stem}_*.json"))
        if not meta_files:
            return {}
        return json.load(open(meta_files[-1]))