import joblib
import pandas as pd
from pathlib import Path
from src.models.registry import ModelRegistry


class ModelPredictor:
    def __init__(self, model_dir: str = "models"):
        self.registry = ModelRegistry(model_dir)
        self.model = joblib.load(self.registry.latest_model_path())

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)