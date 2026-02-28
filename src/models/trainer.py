from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import yaml


@dataclass
class TrainConfig:
    model_type: str
    task_type: str
    test_size: float
    shuffle: bool
    random_state: int


class ModelTrainer:
    def __init__(self, config_path: str = "config/model.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.config = TrainConfig(
            model_type=cfg.get("model", "RandomForest"),
            task_type=cfg.get("task", "classification"),  # "classification" or "regression"
            test_size=cfg.get("test_size", 0.2),
            shuffle=cfg.get("shuffle", True),
            random_state=cfg.get("random_state", 42),
        )

    def _build_model(self):
        if self.config.model_type == "RandomForest":
            if self.config.task_type == "classification":
                base_model = RandomForestClassifier(
                    n_estimators=200,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
            else:
                base_model = RandomForestRegressor(
                    n_estimators=200,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", base_model),
            ]
        )
        return pipeline

    def train(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, float]]:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state,
        )

        model = self._build_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if self.config.task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, average="weighted")),
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics = {
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_test, y_pred)),
            }

        return model, metrics