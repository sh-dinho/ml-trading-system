# tests/test_features.py
import pandas as pd
from pathlib import Path

from src.features.engineer import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger("test_features")


def assert_directory(path: Path):
    if not path.exists() or not path.is_dir():
        raise AssertionError(f"Directory missing: {path}")
    logger.info(f"[OK] Directory exists: {path}")


def assert_non_empty_csv(path: Path):
    if not path.exists():
        raise AssertionError(f"Expected file missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise AssertionError(f"CSV is empty: {path}")
    logger.info(f"[OK] Valid feature CSV: {path} ({len(df)} rows)")


def assert_expected_columns(df: pd.DataFrame, expected_prefixes):
    missing = [p for p in expected_prefixes if not any(col.startswith(p) for col in df.columns)]
    if missing:
        raise AssertionError(f"Missing expected feature groups: {missing}")
    logger.info(f"[OK] Feature groups present: {expected_prefixes}")


def test_feature_pipeline():
    logger.info("=== Testing Feature Engineering Pipeline ===")

    pipeline = FeatureEngineer(
        data_config="config/data.yaml",
        feature_config="config/features.yaml"
    )
    pipeline.process_all()

    features_dir = pipeline.features_dir
    assert_directory(features_dir)

    feature_files = list(features_dir.glob("*.csv"))
    if not feature_files:
        raise AssertionError("FeatureEngineer produced no feature files.")
    logger.info(f"[OK] Feature files found: {len(feature_files)}")

    # Expected prefixes based on your full features.yaml
    expected_prefixes = [
        "SMA_", "EMA_", "MACD_", "RSI_", "ROC_", "STOCH_",
        "BB_upper_", "BB_lower_", "volatility_", "ATR_",
        "returns", "log_returns", "OBV", "VROC_", "CMF_"
    ]

    for f in feature_files:
        df = pd.read_csv(f, index_col=0)
        assert_non_empty_csv(f)
        assert_expected_columns(df, expected_prefixes)

    logger.info("=== Feature Engineering Test Complete ===")


if __name__ == "__main__":
    test_feature_pipeline()