import pandas as pd
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import get_logger
from src.utils.paths import ensure_dir
from .indicators import (
    sma, ema, rsi, bollinger_bands,
    returns, log_returns, rolling_volatility
)

logger = get_logger("feature_engineer")


class FeatureEngineer:
    def __init__(self,
                 data_config="config/data.yaml",
                 feature_config="config/features.yaml",
                 max_workers=4):
        # Load configs
        with open(feature_config, "r") as f:
            self.feature_cfg = yaml.safe_load(f)
        with open(data_config, "r") as f:
            self.data_cfg = yaml.safe_load(f)

        # Directories
        self.processed_dir = ensure_dir(Path(self.data_cfg.get("output_dir", "data/processed")))
        self.features_dir = ensure_dir(Path("data/features"))
        self.max_workers = max_workers

    # ---------------- Build features for a single DataFrame ----------------
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(pd.to_numeric, errors="coerce")
        features = pd.DataFrame(index=df.index)

        # Trend indicators
        for window in self.feature_cfg.get("sma", []):
            features[f"SMA_{window}"] = sma(df, window)
        for window in self.feature_cfg.get("ema", []):
            features[f"EMA_{window}"] = ema(df, window)

        # Momentum indicators
        rsi_cfg = self.feature_cfg.get("rsi", {})
        if rsi_cfg.get("enabled", False):
            w = rsi_cfg.get("window", 14)
            features[f"RSI_{w}"] = rsi(df, w)

        # Volatility indicators
        boll_cfg = self.feature_cfg.get("bollinger", {})
        if boll_cfg.get("enabled", False):
            features = features.join(
                bollinger_bands(df,
                                window=boll_cfg.get("window", 20),
                                num_std=boll_cfg.get("num_std", 2))
            )

        # Price-action features
        if self.feature_cfg.get("returns", {}).get("enabled", False):
            features["returns"] = returns(df)
        if self.feature_cfg.get("log_returns", {}).get("enabled", False):
            features["log_returns"] = log_returns(df)
        if self.feature_cfg.get("rolling_volatility", {}).get("enabled", False):
            w = self.feature_cfg["rolling_volatility"].get("window", 20)
            features[f"volatility_{w}"] = rolling_volatility(df, w)

        return features

    # ---------------- Process all CSVs in the directory ----------------
    def process_all(self):
        files = list(self.processed_dir.glob("*.csv"))
        if not files:
            logger.warning(f"No CSVs found in {self.processed_dir}")
            return

        logger.info(f"Building features for {len(files)} tickers using {self.max_workers} workers")

        def process_file(file_path: Path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                features = self.build_features(df)
                output_path = self.features_dir / file_path.name
                features.to_csv(output_path)
                logger.info(f"Features saved: {output_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

        # Parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in files}
            for future in as_completed(futures):
                future.result()