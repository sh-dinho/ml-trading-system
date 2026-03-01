from typing import Union
import pandas as pd
from .registry import ModelRegistry

class ModelPredictor:
    def __init__(self, model_dir="models", model_path=None):
        self.registry=ModelRegistry(model_dir)
        self.model=self.registry.load_model(path=model_path) if model_path else self.registry.load_model(latest=True)

    def predict(self,X:Union[pd.DataFrame,pd.Series]):
        if isinstance(X,pd.Series): X=X.to_frame().T
        return self.model.predict(X)

    def predict_proba(self,X:Union[pd.DataFrame,pd.Series]):
        if not hasattr(self.model,"predict_proba"):
            raise AttributeError("Model has no predict_proba")
        if isinstance(X,pd.Series): X=X.to_frame().T
        return self.model.predict_proba(X)