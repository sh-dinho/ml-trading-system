import pandas as pd
from pathlib import Path

from src.data.fetcher import DataFetcher
from src.data.preprocess import DataPreprocessor
from src.utils.logger import get_logger

logger = get_logger("test_ingestion")


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
    logger.info(f"[OK] Valid CSV: {path} ({len(df)} rows)")


def test_ingestion_pipeline():
    logger.info("=== Testing Data Fetcher ===")
    fetcher = DataFetcher("config/data.yaml")
    fetcher.fetch_data()

    raw_dir = fetcher.output_dir
    assert_directory(raw_dir)

    raw_files = list(raw_dir.glob("*.csv"))
    if not raw_files:
        raise AssertionError("Fetcher produced no CSV files.")
    logger.info(f"[OK] Raw files found: {len(raw_files)}")

    for f in raw_files:
        assert_non_empty_csv(f)

    logger.info("\n=== Testing Data Preprocessor ===")
    preprocessor = DataPreprocessor("config/data.yaml")
    preprocessor.process_all()

    processed_dir = preprocessor.processed_dir
    assert_directory(processed_dir)

    processed_files = list(processed_dir.glob("*.csv"))
    if not processed_files:
        raise AssertionError("Preprocessor produced no CSV files.")
    logger.info(f"[OK] Processed files found: {len(processed_files)}")

    for f in processed_files:
        assert_non_empty_csv(f)

    logger.info("\n=== Ingestion Pipeline Test Complete ===")


if __name__ == "__main__":
    test_ingestion_pipeline()