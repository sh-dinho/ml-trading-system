from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


@dataclass
class TrainConfig:
    model_type: str
    task_type: str
    test_size: float
    random_state: int
    window_size: int
    step_size: int


class ModelTrainer:
    def __init__(self, config_path="config/model.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.config = TrainConfig(
            model_type=cfg.get("model", "RandomForest"),
            task_type=cfg.get("task", "classification"),
            test_size=cfg.get("test_size", 0.2),
            random_state=cfg.get("random_state", 42),
            window_size=cfg.get("window_size", 1000),
            step_size=cfg.get("step_size", 100),
        )

    # -----------------------------
    # Model builder
    # -----------------------------
    def _build_model(self):
        if self.config.model_type == "RandomForest":
            if self.config.task_type == "classification":
                base = RandomForestClassifier(
                    n_estimators=300,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
            else:
                base = RandomForestRegressor(
                    n_estimators=300,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
        else:
            raise ValueError(f"Unsupported model type {self.config.model_type}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", base)
        ])

    # -----------------------------
    # Walk-forward validation
    # -----------------------------
    def walk_forward(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, float]]:
        w = self.config.window_size
        step = self.config.step_size

        metrics_list = []
        models = []

        for start in range(0, len(X) - w, step):
            end = start + w
            test_end = end + step

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            X_test = X.iloc[end:test_end]
            y_test = y.iloc[end:test_end]

            if len(X_test) == 0:
                break

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

            metrics_list.append(metrics)
            models.append(model)

        # Aggregate metrics
        final_metrics = {
            key: float(np.mean([m[key] for m in metrics_list]))
            for key in metrics_list[0]
        }

        return models[-1], final_metrics