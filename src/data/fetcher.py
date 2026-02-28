import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.paths import data_path, ensure_dir
from src.utils.logger import get_logger

logger = get_logger("data_fetcher")


class DataFetcher:
    def __init__(self, config_file: str, max_workers: int = 5, force_refresh: bool = False):
        """
        Initialize DataFetcher.

        Args:
            config_file (str): YAML config path
            max_workers (int): Number of parallel downloads
            force_refresh (bool): If True, re-download even if file exists
        """
        self.config = self._load_config(config_file)
        self.output_dir = ensure_dir(data_path(*Path(self.config.get("output_dir", "raw")).parts))
        self.max_workers = max_workers
        self.force_refresh = force_refresh

        # Always use today as end date
        self.start_date = self.config.get("start_date", "2014-01-01")
        self.end_date = datetime.today().strftime("%Y-%m-%d")

    @staticmethod
    def _load_config(config_file: str) -> dict:
        try:
            with open(config_file, "r") as f:
                cfg = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_file}")
            return cfg
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML: {e}")
            raise

    def fetch_data(self, tickers: List[str] = None):
        tickers = tickers or self.config.get("tickers", [])
        if not tickers:
            logger.warning("No tickers specified in config.")
            return

        logger.info(f"Fetching {len(tickers)} tickers from {self.start_date} to {self.end_date}")

        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._download_ticker, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Unhandled error fetching {ticker}: {e}")

    def _download_ticker(self, ticker: str):
        file_path = self.output_dir / f"{ticker}.csv"

        # Skip if cached
        if file_path.exists() and not self.force_refresh:
            logger.info(f"Skipping {ticker}, file exists: {file_path}")
            return

        try:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return

            df = self._clean_data(df)
            df.to_csv(file_path)
            logger.info(f"Data for {ticker} saved to {file_path}")
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaning_cfg = self.config.get("cleaning", {})
        if cleaning_cfg.get("remove_duplicates", True):
            df = df[~df.index.duplicated(keep="first")]

        if cleaning_cfg.get("forward_fill", False):
            df = df.ffill()

        if cleaning_cfg.get("dropna", False):
            df = df.dropna()

        if cleaning_cfg.get("sort_index", True):
            df = df.sort_index()

        return df