from pathlib import Path
import pandas as pd
import numpy as np
import mlflow

from src.models.trainer_multi import ModelTrainer
from src.models.registry import ModelRegistry
from src.models.feature_selection import FeatureSelector


class TrainPipeline:
    def __init__(
        self,
        dataset_dir: str = "data/datasets",
        model_dir: str = "models",
        global_model: bool = True,
        n_trials: int = 20,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.global_model = global_model
        self.n_trials = n_trials

    def _load_all(self) -> pd.DataFrame:
        frames = []
        for file in self.dataset_dir.glob("*.csv"):
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            df["ticker"] = file.stem
            frames.append(df)
        return pd.concat(frames).sort_index()

    def _train_core(self, df: pd.DataFrame, model_name: str, run_name: str):

        # ----------------------------------------------------
        # FIX: Keep only numeric features
        # ----------------------------------------------------
        X = df.drop("target", axis=1)
        X = X.select_dtypes(include=[np.number])   # <── KEY FIX
        y = df["target"]

        trainer = ModelTrainer()

        preds, window_results, importance_list = trainer.walk_forward_multi(
            X, y, n_trials=self.n_trials
        )

        selector = FeatureSelector(min_importance=0.0, min_windows=0.3)
        selected_features = selector.select_stable_features(importance_list)
        ranked = selector.rank_features(importance_list)

        # ----------------------------------------------------
        # FIX: Select numeric-only features safely
        # ----------------------------------------------------
        X_selected = X[selected_features]

        final_family = window_results[-1]["family"]
        final_params = window_results[-1]["params"]

        final_model = trainer.build_model(final_family, final_params)
        final_model.fit(X_selected, y)

        registry = ModelRegistry(str(self.model_dir))
        saved_path = registry.save_model(final_model, model_name)

        imp_path = self.model_dir / f"{model_name}_feature_importance.csv"
        ranked.to_csv(imp_path)

        sel_path = self.model_dir / f"{model_name}_selected_features.csv"
        pd.Series(selected_features).to_csv(sel_path, index=False)

        avg_score = float(sum(w["score"] for w in window_results) / len(window_results))

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("num_windows", len(window_results))
            mlflow.log_param("final_family", final_family)
            mlflow.log_params(final_params)
            mlflow.log_param("num_selected_features", len(selected_features))
            mlflow.log_metric("avg_val_score", avg_score)
            mlflow.log_artifact(str(imp_path))
            mlflow.log_artifact(str(sel_path))

        print(f"\nModel saved at {saved_path}")
        print(f"Final family: {final_family}")
        print(f"Avg validation score: {avg_score:.4f}")
        print(f"Selected features ({len(selected_features)}): {selected_features}")

    def run_global(self):
        df = self._load_all()
        self._train_core(df, "global_multi_wf.pkl", "walk_forward_global_multi")

    def run_for_ticker(self, ticker: str):
        dataset_path = self.dataset_dir / f"{ticker}.csv"
        if not dataset_path.exists():
            print(f"[!] Dataset not found for {ticker}")
            return
        df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
        self._train_core(df, f"{ticker}_multi_wf.pkl", f"walk_forward_{ticker}_multi")

    def run_all(self):
        for file in self.dataset_dir.glob("*.csv"):
            self.run_for_ticker(file.stem)

    def run(self):
        if self.global_model:
            self.run_global()
        else:
            self.run_all()