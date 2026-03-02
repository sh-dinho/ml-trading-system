from pathlib import Path
import pandas as pd

from src.models.predictor import ModelPredictor
from src.backtest.backtester import Backtester, BacktestConfig


class BacktestPipeline:
    def __init__(
        self,
        dataset_dir="data/datasets",
        model_dir="models",
        backtest_dir="data/backtests"
    ):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.backtest_dir = Path(backtest_dir)
        self.backtest_dir.mkdir(parents=True, exist_ok=True)

        cfg = BacktestConfig()
        self.backtester = Backtester(cfg)

    # --------------------------------------------------------
    # Portfolio backtest (multi‑ticker)
    # --------------------------------------------------------
    def run_portfolio(self):
        predictor = ModelPredictor(model_dir=str(self.model_dir))

        data = {}
        preds = {}

        # Load all tickers
        for file in self.dataset_dir.glob("*.csv"):
            ticker = file.stem
            df = pd.read_csv(file, index_col=0, parse_dates=True)

            X = df.drop("target", axis=1)
            y_pred = predictor.predict(X)

            data[ticker] = df
            preds[ticker] = pd.Series(y_pred, index=df.index)

        # Run portfolio backtest
        portfolio_df, metrics = self.backtester.run_portfolio(data, preds)

        # Save results
        out_path = self.backtest_dir / "portfolio_backtest.csv"
        portfolio_df.to_csv(out_path)

        print(f"\nPortfolio backtest saved at {out_path}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # --------------------------------------------------------
    # Single‑ticker backtest
    # --------------------------------------------------------
    def run_for_ticker(self, ticker: str):
        dataset_path = self.dataset_dir / f"{ticker}.csv"
        if not dataset_path.exists():
            print(f"[!] Dataset not found for {ticker}")
            return

        df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
        predictor = ModelPredictor(model_dir=str(self.model_dir))

        X = df.drop("target", axis=1)
        preds = pd.Series(predictor.predict(X), index=df.index)

        results, metrics = self.backtester.run(df, preds, ticker)

        out_path = self.backtest_dir / f"{ticker}_backtest.csv"
        results.to_csv(out_path)

        print(f"\nBacktest for {ticker} saved at {out_path}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # --------------------------------------------------------
    # Run all tickers individually
    # --------------------------------------------------------
    def run_all(self):
        for file in self.dataset_dir.glob("*.csv"):
            self.run_for_ticker(file.stem)


if __name__ == "__main__":
    BacktestPipeline().run_portfolio()