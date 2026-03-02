import pandas as pd
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.paths import ensure_dir
from src.utils.logger import get_logger

from .indicators import (
    sma, ema, rsi, bollinger_bands,
    returns, log_returns, rolling_volatility
)

logger = get_logger("feature_engineer")


class FeatureEngineer:
    def __init__(
        self,
        data_config: str = "config/data.yaml",
        feature_config: str = "config/features.yaml",
        max_workers: int = 4
    ):
        self.feature_cfg = self._load_yaml(feature_config)
        self.data_cfg = self._load_yaml(data_config)

        output_cfg = self.data_cfg.get("output", {})
        self.processed_dir = ensure_dir(Path(output_cfg.get("processed_dir", "data/processed")))
        self.features_dir = ensure_dir(Path(output_cfg.get("features_dir", "data/features")))

        self.max_workers = max_workers

        # Registry of all supported indicators
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
    # Config loading
    # -----------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config {path}: {e}")
            raise

    # -----------------------------
    # Feature computation wrappers
    # -----------------------------
    def _compute_sma(self, df, cfg):
        if not cfg.get("enabled", True):
            return {}
        windows = cfg.get("windows", [])
        return {f"SMA_{w}": sma(df, w) for w in windows}

    def _compute_ema(self, df, cfg):
        if not cfg.get("enabled", True):
            return {}
        windows = cfg.get("windows", [])
        return {f"EMA_{w}": ema(df, w) for w in windows}

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

        # Iterate through groups (trend, momentum, volatility, price_action)
        for group_name, group_cfg in self.feature_cfg.items():
            if not isinstance(group_cfg, dict):
                continue
            if not group_cfg.get("enabled", True):
                continue

            for feature_name, cfg in group_cfg.items():
                if feature_name == "enabled":
                    continue
                if feature_name not in self.registry:
                    logger.warning(f"Unknown feature '{feature_name}' in config.")
                    continue

                try:
                    result = self.registry[feature_name](df, cfg)
                    if isinstance(result, dict):
                        features.update(result)
                except Exception as e:
                    logger.error(f"Failed to compute {feature_name}: {e}")

        return pd.DataFrame(features, index=df.index)

    # -----------------------------
    # Batch processing
    # -----------------------------
    def process_all(self):
        files = list(self.processed_dir.glob("*.csv"))
        if not files:
            logger.warning("No CSV files found for feature engineering.")
            return

        logger.info(f"Building features for {len(files)} files")

        def process_file(file: Path):
            try:
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                features = self.build_features(df)
                out_path = self.features_dir / file.name
                features.to_csv(out_path)
                logger.info(f"Saved features: {out_path}")
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in files}
            for future in as_completed(futures):
                future.result()