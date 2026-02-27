import pandas as pd
from pathlib import Path
import yaml

class TrainingDatasetBuilder:
    def __init__(self, config_path="config/training.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.features_dir = Path("data/features")
        self.processed_dir = Path("data/processed")
        self.dataset_dir = Path("data/datasets")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_numeric(self, df):
        for col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "")
                .str.replace("$", "")
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def load_data(self, ticker):
        features = pd.read_csv(self.features_dir / f"{ticker}.csv", index_col=0, parse_dates=True)
        prices = pd.read_csv(self.processed_dir / f"{ticker}.csv", index_col=0, parse_dates=True)

        features = self._ensure_numeric(features)
        prices = self._ensure_numeric(prices)

        return features, prices

    def create_target(self, prices):
        horizon = self.config["target"]["horizon"]
        method = self.config["target"]["type"]

        close = prices["Close"]

        if method == "future_return":
            return (close.shift(-horizon) / close - 1).rename("target")

        if method == "direction":
            return (close.shift(-horizon) > close).astype(int).rename("target")

        if method == "multi_class":
            future_ret = close.shift(-horizon) / close - 1
            bins = [-1, -0.02, -0.005, 0.005, 0.02, 1]
            labels = [0, 1, 2, 3, 4]
            return pd.cut(future_ret, bins=bins, labels=labels).rename("target")

        raise ValueError("Unknown target type")

    def build_for_ticker(self, ticker):
        features, prices = self.load_data(ticker)
        target = self.create_target(prices)

        dataset = features.join(target).dropna()
        output_path = self.dataset_dir / f"{ticker}.csv"
        dataset.to_csv(output_path)

        print(f"Training dataset saved: {output_path}")

    def build_all(self):
        for file in self.features_dir.glob("*.csv"):
            ticker = file.stem
            self.build_for_ticker(ticker)