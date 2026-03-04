import pandas as pd
from pathlib import Path

from src.data.dataset import DatasetBuilder
from src.utils.logger import get_logger

logger = get_logger("test_dataset")


def assert_directory(path: Path):
    if not path.exists() or not path.is_dir():
        raise AssertionError(f"Directory missing: {path}")
    logger.info(f"[OK] Directory exists: {path}")


def assert_non_empty_csv(path: Path):
    if not path.exists():
        raise AssertionError(f"Expected dataset file missing: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        raise AssertionError(f"Dataset CSV is empty: {path}")
    logger.info(f"[OK] Dataset contains {len(df)} rows")
    return df


def assert_target_present(df: pd.DataFrame):
    if "target" not in df.columns:
        raise AssertionError("Dataset missing required 'target' column.")
    logger.info("[OK] Target column present")


def assert_no_nan_columns(df: pd.DataFrame):
    nan_cols = [col for col in df.columns if df[col].isna().all()]
    if nan_cols:
        raise AssertionError(f"Columns contain only NaN values: {nan_cols}")
    logger.info("[OK] No NaN-only columns")


def assert_sorted_index(df: pd.DataFrame):
    if not df.index.is_monotonic_increasing:
        raise AssertionError("Dataset index is not sorted chronologically.")
    logger.info("[OK] Index sorted")


def test_dataset_builder():
    logger.info("=== Testing Dataset Builder ===")

    builder = DatasetBuilder(
        data_config="config/data.yaml",
        feature_config="config/features.yaml",
        target_config="config/targets.yaml"
    )

    dataset = builder.build_all()

    dataset_dir = builder.dataset_dir
    assert_directory(dataset_dir)

    dataset_path = dataset_dir / "dataset.csv"
    df = assert_non_empty_csv(dataset_path)

    assert_target_present(df)
    assert_no_nan_columns(df)
    assert_sorted_index(df)

    logger.info("=== Dataset Builder Test Complete ===")


if __name__ == "__main__":
    test_dataset_builder()