import numpy as np
import pandas as pd
import optuna

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, mean_squared_error

import lightgbm as lgb
import xgboost as xgb


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    # --------------------------------------------------------
    # Model builders for each family
    # --------------------------------------------------------
    def build_model(self, family, params):
        if family == "rf":
            if self.config.task_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=self.config.random_state,
                    n_jobs=-1
                )

        elif family == "xgb":
            model = xgb.XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample"],
                random_state=self.config.random_state,
                n_jobs=-1
            )

        elif family == "lgb":
            model = lgb.LGBMClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample"],
                random_state=self.config.random_state,
                n_jobs=-1
            )

        elif family == "logreg":
            model = LogisticRegression(max_iter=500)

        else:
            raise ValueError(f"Unknown model family: {family}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

    # --------------------------------------------------------
    # Hyperparameter search space per family
    # --------------------------------------------------------
    def suggest_params(self, trial, family):
        if family == "rf":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }

        if family in ("xgb", "lgb"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample": trial.suggest_float("colsample", 0.5, 1.0),
            }

        if family == "logreg":
            return {}

    # --------------------------------------------------------
    # Objective for Optuna
    # --------------------------------------------------------
    def _objective(self, trial, family, X_train, y_train, X_val, y_val):
        params = self.suggest_params(trial, family)
        model = self.build_model(family, params)

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        if self.config.task_type == "classification":
            return f1_score(y_val, preds, average="weighted")
        else:
            return -mean_squared_error(y_val, preds)

    # --------------------------------------------------------
    # Walk-forward with multi-model sweeps
    # --------------------------------------------------------
    def walk_forward_multi(self, X, y, n_trials=20):
        w = self.config.window_size
        step = self.config.step_size

        families = ["rf", "xgb", "lgb", "logreg"]

        preds_all = pd.Series(index=y.index, dtype=float)
        window_results = []

        for start in range(0, len(X) - w - step, step):
            end = start + w
            test_end = end + step

            X_train, y_train = X.iloc[start:end], y.iloc[start:end]
            X_val, y_val = X.iloc[end:test_end], y.iloc[end:test_end]

            best_family = None
            best_score = -np.inf
            best_params = None

            # Tune each family
            for fam in families:
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda t: self._objective(t, fam, X_train, y_train, X_val, y_val),
                    n_trials=n_trials
                )

                if study.best_value > best_score:
                    best_score = study.best_value
                    best_family = fam
                    best_params = study.best_params

            # Train best model for this window
            model = self.build_model(best_family, best_params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            preds_all.iloc[end:test_end] = preds

            window_results.append({
                "family": best_family,
                "params": best_params,
                "score": best_score
            })

        return preds_all, window_results