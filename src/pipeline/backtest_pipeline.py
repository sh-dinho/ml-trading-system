from pathlib import Path
import pandas as pd

from src.models.predictor import ModelPredictor
from src.backtest.backtester import Backtester, BacktestConfig


class BacktestPipeline:
    def __init__(
        self,
        dataset_dir: str = "data/datasets",
        model_dir: str = "models",
        backtest_dir: str = "data/backtests",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.backtest_dir = Path(backtest_dir)
        self.backtest_dir.mkdir(parents=True, exist_ok=True)

        cfg = BacktestConfig(
            threshold_long=0.0,
            threshold_short=0.0,
            cost_bps=10.0,
        )
        self.backtester = Backtester(cfg)

    def run_for_ticker(self, ticker: str):
        dataset_path = self.dataset_dir / f"{ticker}.csv"
        if not dataset_path.exists():
            print(f"[!] Dataset not found for {ticker}, skipping.")
            return

        df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

        predictor = ModelPredictor(model_dir=str(self.model_dir))
        X = df.drop("target", axis=1)
        preds = predictor.predict(X)
        preds = pd.Series(preds, index=df.index)

        results, metrics = self.backtester.run(df, preds, ticker)

        out_path = self.backtest_dir / f"{ticker}_backtest.csv"
        results.to_csv(out_path)

        print(f"\nBacktest for {ticker}")
        print("Saved to:", out_path)
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    def run_all(self):
        for file in self.dataset_dir.glob("*.csv"):
            ticker = file.stem
            self.run_for_ticker(ticker)


if __name__ == "__main__":
    BacktestPipeline().run_all()