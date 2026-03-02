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

        # Registry of indicator functions
        self.registry = {
            "sma": self._compute_sma,
            "ema": self._compute_ema,
            "rsi": self._compute_rsi,
            "bollinger": self._compute_bollinger,
            "returns": self._compute_returns,
            "log_returns": self._compute_log_returns,
            "rolling_volatility": self._compute_volatility,
        }

    # -----------------------------
    # Indicator wrappers
    # -----------------------------
    def _compute_sma(self, df, cfg):
        if not cfg.get("enabled", True):
            return {}
        return {f"SMA_{w}": sma(df, w) for w in cfg.get("windows", [])}

    def _compute_ema(self, df, cfg):
        if not cfg.get("enabled", True):
            return {}
        return {f"EMA_{w}": ema(df, w) for w in cfg.get("windows", [])}

    def _compute_rsi(self, df, cfg):
        if not cfg.get("enabled", False):
            return {}
        w = cfg.get("window", 14)
        return {f"RSI_{w}": rsi(df, w)}

    def _compute_bollinger(self, df, cfg):
        if not cfg.get("enabled", False):
            return {}
        w = cfg.get("window", 20)
        n = cfg.get("num_std", 2)
        return bollinger_bands(df, w, n)

    def _compute_returns(self, df, cfg):
        return {"returns": returns(df)} if cfg.get("enabled", False) else {}

    def _compute_log_returns(self, df, cfg):
        return {"log_returns": log_returns(df)} if cfg.get("enabled", False) else {}

    def _compute_volatility(self, df, cfg):
        if not cfg.get("enabled", False):
            return {}
        w = cfg.get("window", 20)
        return {f"volatility_{w}": rolling_volatility(df, w)}

    # -----------------------------
    # Main feature builder
    # -----------------------------
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(pd.to_numeric, errors="coerce")
        features = {}

        for group_name, group_cfg in self.config.items():
            if not isinstance(group_cfg, dict):
                continue
            if not group_cfg.get("enabled", True):
                continue

            for feat_name, cfg in group_cfg.items():
                if feat_name == "enabled":
                    continue
                if feat_name not in self.registry:
                    continue

                result = self.registry[feat_name](df, cfg)
                if isinstance(result, dict):
                    features.update(result)

        return pd.DataFrame(features, index=df.index)

    # -----------------------------
    # Batch processing
    # -----------------------------
    def process_all(self):
        for file in self.processed_dir.glob("*.csv"):
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            features = self.build_features(df)

            out_path = self.features_dir / file.name
            features.to_csv(out_path)

            print(f"Features saved: {out_path}")