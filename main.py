import os
import sys
from src.data.fetcher import DataFetcher
from src.data.preprocess import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.data.dataset import DatasetBuilder
from src.pipelines.end_to_end import EndToEndPipeline
from src.utils.logger import get_logger

logger = get_logger("v1.1_automation")

def run_automated_pipeline():
    logger.info("=== Starting Quant Pipeline v1.1 ===")
    
    try:
        # 1. Ingestion
        logger.info("[1/5] Fetching latest market data...")
        fetcher = DataFetcher(config_file="config/data.yaml")
        # Assuming DataFetcher has a method to process all tickers defined in config
        fetcher.fetch_all() 
        
        # 2. Preprocessing
        logger.info("[2/5] Cleaning and standardizing data...")
        preprocessor = DataPreprocessor()
        preprocessor.process_all()
        
        # 3. Feature Engineering
        logger.info("[3/5] Computing technical indicators and alpha features...")
        engineer = FeatureEngineer()
        engineer.process_all()
        
        # 4. Dataset Building
        logger.info("[4/5] Aligning targets and cleaning NaNs...")
        builder = DatasetBuilder()
        builder.build_all()
        
        # 5. End-to-End Training & Backtesting
        logger.info("[5/5] Executing Walk-Forward Optimization & Backtesting...")
        pipeline = EndToEndPipeline(global_model=True)
        pipeline.run_portfolio()
        
        logger.info("=== Pipeline Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_automated_pipeline()