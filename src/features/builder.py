import pandas as pd
from pathlib import Path
import yaml

from .indicators import (
    sma, ema, rsi, bollinger_bands,
    returns, log_returns, rolling_volatility
)

class FeatureBuilder:
    def __init__(self, config_path="config/features.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.processed_dir = Path("data/processed")
        self.features_dir = Path("data/features")
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def build_features(self, df: pd.DataFrame):
        # Ensure numeric types
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

        features = pd.DataFrame(index=df.index)

        # Trend indicators
        for window in self.config["sma"]:
            features[f"SMA_{window}"] = sma(df, window)

        for window in self.config["ema"]:
            features[f"EMA_{window}"] = ema(df, window)

        # Momentum
        if self.config["rsi"]["enabled"]:
            w = self.config["rsi"]["window"]
            features[f"RSI_{w}"] = rsi(df, w)

        # Volatility
        if self.config["bollinger"]["enabled"]:
            bb = bollinger_bands(
                df,
                window=self.config["bollinger"]["window"],
                num_std=self.config["bollinger"]["num_std"]
            )
            features = features.join(bb)

        # Price‑action features
        if self.config["returns"]["enabled"]:
            features["returns"] = returns(df)

        if self.config["log_returns"]["enabled"]:
            features["log_returns"] = log_returns(df)

        if self.config["rolling_volatility"]["enabled"]:
            w = self.config["rolling_volatility"]["window"]
            features[f"volatility_{w}"] = rolling_volatility(df, w)

        return features

    def process_all(self):
        for file in self.processed_dir.glob("*.csv"):
            df = pd.read_csv(file, index_col=0, parse_dates=True)

            features = self.build_features(df)

            output_path = self.features_dir / file.name
            features.to_csv(output_path)

            print(f"Features saved: {output_path}")