import pandas as pd
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.paths import ensure_dir
from src.utils.logger import get_logger

from .indicators import (
    sma, ema, rsi, roc, macd, stochastic,
    bollinger_bands, rolling_volatility, atr,
    returns, log_returns, obv, vroc, cmf
)

logger = get_logger("feature_engineer")


class FeatureEngineer:
    def __init__(self, data_config="config/data.yaml", feature_config="config/features.yaml", max_workers=4):
        with open(feature_config) as f:
            self.feature_cfg = yaml.safe_load(f)

        with open(data_config) as f:
            self.data_cfg = yaml.safe_load(f)

        output_cfg = self.data_cfg.get("output", {})
        self.processed_dir = ensure_dir(Path(output_cfg.get("processed_dir", "data/processed")))
        self.features_dir = ensure_dir(Path(output_cfg.get("features_dir", "data/features")))

        self.max_workers = max_workers

        # Indicator registry
        self.registry = {
            "sma": self._compute_sma,
            "ema": self._compute_ema,
            "rsi": self._compute_rsi,
            "roc": self._compute_roc,
            "macd": self._compute_macd,
            "stochastic": self._compute_stochastic,
            "bollinger": self._compute_bollinger,
            "rolling_volatility": self._compute_volatility,
            "atr": self._compute_atr,
            "returns": self._compute_returns,
            "log_returns": self._compute_log_returns,
            "obv": self._compute_obv,
            "vroc": self._compute_vroc,
            "cmf": self._compute_cmf,
        }

    # ---------------- Indicator wrappers ----------------
    def _compute_sma(self, df, cfg):
        if not cfg.get("enabled", True): return {}
        return {f"SMA_{w}": sma(df, w) for w in cfg.get("windows", [])}

    def _compute_ema(self, df, cfg):
        if not cfg.get("enabled", True): return {}
        return {f"EMA_{w}": ema(df, w) for w in cfg.get("windows", [])}

    def _compute_rsi(self, df, cfg):
        if not cfg.get("enabled", False): return {}
        w = cfg.get("window", 14)
        return {f"RSI_{w}": rsi(df, w)}

    def _compute_roc(self, df, cfg):
        if not cfg.get("enabled", False): return {}
        w = cfg.get("window", 10)
        return {f"ROC_{w}": roc(df, w)}

    def _compute_macd(self, df, cfg):
        if not cfg.get("enabled", False): return {}
        return macd(df, cfg.get("fast", 12), cfg.get("slow", 26), cfg.get("signal", 9))

    def _compute_stochastic(self, df, cfg):
        if not cfg.get("enabled", False): return {}
        return stochastic(df, cfg.get("k_window", 14), cfg.get("d_window", 3))

    def _compute_bollinger(self, df, cfg):
        if not cfg.get("enabled", False): return {}
        return bollinger_bands(df, cfg.get("window", 20), cfg.get("num_std", 2))

    def _compute_volatility(self, df, cfg):
        if not cfg.get("enabled", False): return {}
        w = cfg.get("window", 20)
        return {f"volatility_{w}": rolling_volatility(df, w)}

    def _compute_atr(self, df, cfg):
        if not cfg.get("enabled", False): return {}
        w = cfg.get("window", 14)
        return {f"ATR_{w}": atr(df, w)}

    def _compute_returns(self, df, cfg):
        return {"returns": returns(df)} if cfg.get("enabled", False) else {}

    def _compute_log_returns(self, df, cfg):
        return {"log_returns": log_returns(df)} if cfg.get("enabled", False) else {}

    def _compute_obv(self, df, cfg):
        return {"OBV": obv(df)} if cfg.get("enabled", False) else {}

    def _compute_vroc(self, df, cfg):
        if not cfg.get("enabled", False): return {}
        w = cfg.get("window", 10)
        return {f"VROC_{w}": vroc(df, w)}

    def _compute_cmf(self, df, cfg):
        if not cfg.get("enabled", False): return {}
        w = cfg.get("window", 20)
        return {f"CMF_{w}": cmf(df, w)}

    # ---------------- Main builder ----------------
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(pd.to_numeric, errors="coerce")
        features = {}

        for group_name, group_cfg in self.feature_cfg.items():
            if not isinstance(group_cfg, dict): continue
            if not group_cfg.get("enabled", True): continue

            for feat_name, cfg in group_cfg.items():
                if feat_name == "enabled": continue
                if feat_name not in self.registry:
                    logger.warning(f"Unknown feature '{feat_name}' in config.")
                    continue

                result = self.registry[feat_name](df, cfg)
                if isinstance(result, dict):
                    features.update(result)
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        features[col] = result[col]

        return pd.DataFrame(features, index=df.index)

    # ---------------- Batch processing ----------------
    def process_all(self):
        files = list(self.processed_dir.glob("*.csv"))
        if not files:
            logger.warning("No CSVs to process.")
            return

        logger.info(f"Building features for {len(files)} files")

        def process_file(file):
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            features = self.build_features(df)
            out = self.features_dir / file.name
            features.to_csv(out)
            logger.info(f"Saved features: {out}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(process_file, f): f for f in files}
            for future in as_completed(futures):
                future.result()