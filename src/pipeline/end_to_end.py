from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt

from src.training_pipeline.train_pipeline import TrainPipeline
from src.models.predictor import ModelPredictor
from src.backtest.backtester import Backtester, BacktestConfig


class EndToEndPipeline:
    def __init__(
        self,
        dataset_dir: str = "data/datasets",
        model_dir: str = "models",
        backtest_dir: str = "data/backtests",
        n_trials: int = 20,
        global_model: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.backtest_dir = Path(backtest_dir)
        self.backtest_dir.mkdir(parents=True, exist_ok=True)

        self.train_pipeline = TrainPipeline(
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            global_model=global_model,
            n_trials=n_trials,
        )

        cfg = BacktestConfig()
        self.backtester = Backtester(cfg)

    def _plot_equity(self, df: pd.DataFrame, out_path: Path):
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["equity"], label="Strategy")
        if "benchmark_equity" in df.columns:
            plt.plot(df.index, df["benchmark_equity"], label="Benchmark", alpha=0.7)
        plt.title("Equity Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def _save_report(self, metrics, out_path: Path):
        html = "<html><body><h1>Backtest Report</h1><table>"
        for k, v in metrics.items():
            html += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        html += "</table></body></html>"
        with open(out_path, "w") as f:
            f.write(html)

    def run(self):
        with mlflow.start_run(run_name="end_to_end_experiment"):

            print("\n[1] Training walk-forward multi-model tuned model...")
            self.train_pipeline.run()

            print("\n[2] Generating predictions for all tickers...")
            predictor = ModelPredictor(model_dir=str(self.model_dir))

            data = {}
            preds = {}

            for file in self.dataset_dir.glob("*.csv"):
                ticker = file.stem
                df = pd.read_csv(file, index_col=0, parse_dates=True)

                # ----------------------------------------------------
                # FIX: Keep only numeric features for prediction
                # ----------------------------------------------------
                X = df.drop("target", axis=1)
                X = X.select_dtypes(include=[np.number])

                y_pred = predictor.predict(X)

                data[ticker] = df
                preds[ticker] = pd.Series(y_pred, index=df.index)

            print("\n[3] Running portfolio backtest...")
            portfolio_df, metrics = self.backtester.run_portfolio(data, preds)

            out_csv = self.backtest_dir / "portfolio_end_to_end.csv"
            portfolio_df.to_csv(out_csv)

            eq_path = self.backtest_dir / "equity_curve.png"
            self._plot_equity(portfolio_df, eq_path)

            report_path = self.backtest_dir / "performance_report.html"
            self._save_report(metrics, report_path)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.log_artifact(str(out_csv))
            mlflow.log_artifact(str(eq_path))
            mlflow.log_artifact(str(report_path))

            print(f"\nEnd-to-end backtest saved at {out_csv}")
            print(f"Equity curve saved at {eq_path}")
            print(f"Performance report saved at {report_path}")
            print("Final metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    EndToEndPipeline(global_model=True, n_trials=20).run()