import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.paths import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("data_fetcher")


class DataFetcher:
    def __init__(
        self,
        config_file: str,
        max_workers: int = 5,
        force_refresh: bool = False,
        retries: int = 2
    ):
        self.config = self._load_config(config_file)
        self.max_workers = max_workers
        self.force_refresh = force_refresh
        self.retries = retries

        # Directories from improved config
        output_cfg = self.config.get("output", {})
        raw_dir = output_cfg.get("raw_dir", "data/raw")
        self.output_dir = ensure_dir(Path(raw_dir))

        # Date range
        date_cfg = self.config.get("date_range", {})
        self.start_date = date_cfg.get("start", "2014-01-01")
        self.end_date = datetime.today().strftime("%Y-%m-%d")

        # Source parameters
        source_cfg = self.config.get("source", {})
        self.provider = source_cfg.get("provider", "yfinance")
        self.interval = source_cfg.get("interval", "1d")
        self.auto_adjust = source_cfg.get("auto_adjust", True)

        # Cleaning rules
        self.cleaning_cfg = self.config.get("cleaning", {})

    # -----------------------------
    # Config loading
    # -----------------------------
    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
            logger.info(f"Loaded config: {path}")
            return cfg or {}
        except Exception as e:
            logger.error(f"Failed to load config {path}: {e}")
            raise

    # -----------------------------
    # Ticker extraction
    # -----------------------------
    def _get_all_tickers(self) -> List[str]:
        tickers_cfg = self.config.get("tickers", {})
        if isinstance(tickers_cfg, list):
            return tickers_cfg

        # Support grouped tickers (equity, crypto, forex)
        tickers = []
        for group, items in tickers_cfg.items():
            if isinstance(items, list):
                tickers.extend(items)
        return tickers

    # -----------------------------
    # Main fetch entrypoint
    # -----------------------------
    def fetch_data(self, tickers: List[str] = None):
        tickers = tickers or self._get_all_tickers()
        if not tickers:
            logger.warning("No tickers found in config.")
            return

        logger.info(f"Fetching {len(tickers)} tickers from {self.start_date} to {self.end_date}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._download_with_retry, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Unhandled error fetching {ticker}: {e}")

    # -----------------------------
    # Retry wrapper
    # -----------------------------
    def _download_with_retry(self, ticker: str):
        for attempt in range(1, self.retries + 1):
            try:
                return self._download_ticker(ticker)
            except Exception as e:
                logger.warning(f"{ticker}: attempt {attempt}/{self.retries} failed: {e}")
                if attempt == self.retries:
                    logger.error(f"{ticker}: all retries failed")

    # -----------------------------
    # Download logic
    # -----------------------------
    def _download_ticker(self, ticker: str):
        file_path = self.output_dir / f"{ticker}.csv"

        if file_path.exists() and not self.force_refresh:
            logger.info(f"Skipping {ticker}, cached file exists.")
            return

        df = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=self.auto_adjust,
            progress=False,
        )

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return

        df = self._clean_data(df)

        tmp_path = file_path.with_suffix(".tmp")
        df.to_csv(tmp_path)
        tmp_path.rename(file_path)

        logger.info(f"Saved {ticker}: {file_path}")

    # -----------------------------
    # Cleaning logic
    # -----------------------------
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cleaning_cfg.get("remove_duplicates", True):
            df = df[~df.index.duplicated(keep="first")]

        if self.cleaning_cfg.get("forward_fill", False):
            df = df.ffill()

        if self.cleaning_cfg.get("backward_fill", False):
            df = df.bfill()

        if self.cleaning_cfg.get("dropna", False):
            df = df.dropna()

        if self.cleaning_cfg.get("sort_index", True):
            df = df.sort_index()

        if self.cleaning_cfg.get("enforce_numeric", False):
            df = df.apply(pd.to_numeric, errors="coerce")

        return df