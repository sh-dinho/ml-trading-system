from typing import Union
import pandas as pd
from pathlib import Path

from .registry import ModelRegistry


class ModelPredictor:
    def __init__(self, model_dir="models", model_path=None):
        self.registry = ModelRegistry(model_dir)

        # Load model
        if model_path:
            self.model = self.registry.load_model(path=model_path)
        else:
            self.model = self.registry.load_model(latest=True)

        # Load metadata if available
        self.metadata = {}
        try:
            self.metadata = self.registry.load_metadata(Path(model_path)) if model_path else {}
        except Exception:
            pass

    # -----------------------------
    # Input normalization
    # -----------------------------
    def _normalize(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            return X.to_frame().T
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame or Series.")
        return X

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, X: Union[pd.DataFrame, pd.Series]):
        X = self._normalize(X)
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, pd.Series]):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("This model does not support probability predictions.")
        X = self._normalize(X)
        return self.model.predict_proba(X)