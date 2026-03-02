import pandas as pd
from pathlib import Path
import yaml

from src.utils.paths import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("dataset_builder")


class DatasetBuilder:
    def __init__(
        self,
        data_config="config/data.yaml",
        feature_config="config/features.yaml",
        target_config="config/targets.yaml"
    ):
        self.data_cfg = self._load_yaml(data_config)
        self.feature_cfg = self._load_yaml(feature_config)
        self.target_cfg = self._load_yaml(target_config)

        output_cfg = self.data_cfg.get("output", {})
        self.features_dir = Path(output_cfg.get("features_dir", "data/features"))
        self.dataset_dir = ensure_dir(Path("data/datasets"))

    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    # -----------------------------
    # Target computation
    # -----------------------------
    def _compute_target(self, df: pd.DataFrame) -> pd.Series:
        cfg = self.target_cfg
        ttype = cfg.get("type")
        horizon = cfg.get("horizon", 1)

        # Base series
        future_ret = df["returns"].shift(-horizon)
        future_log_ret = df["log_returns"].shift(-horizon)

        # -------------------------
        # Regression targets
        # -------------------------
        if ttype == "future_return":
            return future_ret

        if ttype == "future_log_return":
            return future_log_ret

        # -------------------------
        # Binary direction
        # -------------------------
        if ttype == "direction":
            return (future_ret > 0).astype(int)

        # -------------------------
        # Three-class direction
        # -------------------------
        if ttype == "direction_3class":
            up = cfg["thresholds"]["up"]
            down = cfg["thresholds"]["down"]

            return pd.cut(
                future_ret,
                bins=[-float("inf"), down, up, float("inf")],
                labels=[0, 1, 2]  # down, flat, up
            ).astype(int)

        # -------------------------
        # Future volatility
        # -------------------------
        if ttype == "future_volatility":
            window = cfg.get("vol_window", 20)
            vol = df["returns"].rolling(window).std()
            return vol.shift(-horizon)

        # -------------------------
        # Volatility buckets
        # -------------------------
        if ttype == "volatility_bucket":
            window = cfg.get("vol_window", 20)
            vol = df["returns"].rolling(window).std().shift(-horizon)

            low = cfg["volatility_buckets"]["low"]
            med = cfg["volatility_buckets"]["medium"]

            return pd.cut(
                vol,
                bins=[-float("inf"), low, med, float("inf")],
                labels=[0, 1, 2]  # low, medium, high
            ).astype(int)

        # -------------------------
        # MACD crossovers
        # -------------------------
        if ttype == "macd_cross":
            macd_line = df.filter(like="MACD_").iloc[:, 0]
            signal_line = df.filter(like="MACD_signal").iloc[:, 0]

            prev_diff = (macd_line - signal_line).shift(1)
            curr_diff = macd_line - signal_line

            return (
                (prev_diff < 0) & (curr_diff > 0)
            ).astype(int) - (
                (prev_diff > 0) & (curr_diff < 0)
            ).astype(int)

        # -------------------------
        # Bollinger breakout
        # -------------------------
        if ttype == "bollinger_breakout":
            upper = df.filter(like="BB_upper").iloc[:, 0]
            lower = df.filter(like="BB_lower").iloc[:, 0]
            close = df["returns"].index  # placeholder

            # Use reconstructed price if available
            if "Close" in df.columns:
                close = df["Close"]

            return pd.cut(
                close,
                bins=[-float("inf"), lower, upper, float("inf")],
                labels=[-1, 0, 1]  # below lower, inside, above upper
            ).astype(int)

        # -------------------------
        # RSI overbought/oversold
        # -------------------------
        if ttype == "rsi_signal":
            rsi_col = df.filter(like="RSI_").iloc[:, 0]
            over = cfg["rsi_levels"]["overbought"]
            under = cfg["rsi_levels"]["oversold"]

            return pd.cut(
                rsi_col,
                bins=[-float("inf"), under, over, float("inf")],
                labels=[-1, 0, 1]  # oversold, neutral, overbought
            ).astype(int)

        raise ValueError(f"Unknown target type: {ttype}")

    # -----------------------------
    # Build dataset for one ticker
    # -----------------------------
    def build_single(self, file: Path) -> pd.DataFrame:
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        target = self._compute_target(df)
        df["target"] = target

        df = df.dropna()

        return df

    # -----------------------------
    # Build dataset for all tickers
    # -----------------------------
    def build_all(self):
        files = list(self.features_dir.glob("*.csv"))
        if not files:
            logger.error("No feature files found.")
            return

        combined = []

        for f in files:
            try:
                df = self.build_single(f)
                df["ticker"] = f.stem
                combined.append(df)
                logger.info(f"Built dataset for {f.stem}")
            except Exception as e:
                logger.error(f"Failed to build dataset for {f.name}: {e}")

        full = pd.concat(combined).sort_index()
        out = self.dataset_dir / "dataset.csv"
        full.to_csv(out)

        logger.info(f"Saved dataset: {out}")
        return full