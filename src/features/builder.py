import pandas as pd
from pathlib import Path
import yaml
from .indicators import sma, ema, rsi, bollinger_bands

class FeatureBuilder:
    def __init__(self, config_path="config/features.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.processed_dir = Path("data/processed")
        self.features_dir = Path("data/features")
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def build_features(self, df: pd.DataFrame):
        features = pd.DataFrame(index=df.index)

        # Trend indicators
        for window in self.config["sma"]:
            features[f"SMA_{window}"] = sma(df, window)

        for window in self.config["ema"]:
            features[f"EMA_{window}"] = ema(df, window)

        # Momentum
        if self.config.get("rsi", False):
            features["RSI_14"] = rsi(df)

        # Volatility
        if self.config.get("bollinger", False):
            bb = bollinger_bands(df)
            features = features.join(bb)

        return features

    def process_all(self):
        for file in self.processed_dir.glob("*.csv"):
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            features = self.build_features(df)
            output_path = self.features_dir / file.name
            features.to_csv(output_path)
            print(f"Features saved: {output_path}")