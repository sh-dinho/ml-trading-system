import os
import sys
import logging
from pathlib import Path
from time import time

from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.backtest_pipeline import BacktestPipeline
from src.data.preprocess import DataPreprocessor
from src.features.builder import FeatureBuilder
from src.models.registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Step 1: Data Preprocessing (Raw data → Processed data)
    logger.info("Starting data preprocessing...")
    start_time = time()
    preprocessor = DataPreprocessor(config_file="config/data.yaml")
    preprocessor.process_all()  # Process all CSVs in raw directory
    logger.info(f"Data preprocessing completed in {time() - start_time:.2f} seconds.")

    # Step 2: Feature Engineering (Raw data → Features)
    logger.info("Starting feature engineering...")
    start_time = time()
    feature_builder = FeatureBuilder(config_path="config/features.yaml")
    feature_builder.process_all()  # Process all processed files to extract features
    logger.info(f"Feature engineering completed in {time() - start_time:.2f} seconds.")

    # Step 3: Model Training (Features → Models)
    logger.info("Starting model training...")
    start_time = time()
    train_pipeline = TrainPipeline(dataset_dir="data/datasets", model_dir="models")
    train_pipeline.run_all()  # Train model for all tickers
    logger.info(f"Model training completed in {time() - start_time:.2f} seconds.")

    # Step 4: Model Backtesting (Models → Backtest Results)
    logger.info("Starting model backtesting...")
    start_time = time()
    backtest_pipeline = BacktestPipeline(dataset_dir="data/datasets", model_dir="models", backtest_dir="data/backtests")
    backtest_pipeline.run_all()  # Backtest models for all tickers
    logger.info(f"Backtesting completed in {time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        sys.exit(1)