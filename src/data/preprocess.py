import pandas as pd
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.paths import data_path, ensure_dir
from src.utils.logger import get_logger

logger = get_logger("data_preprocessor")


class DataPreprocessor:
    def __init__(self, config_file: str = "config/data.yaml", max_workers: int = 4):
        # Load YAML config
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        # Set directories using paths.py
        self.raw_dir = ensure_dir(data_path("raw"))
        self.processed_dir = ensure_dir(data_path("processed"))
        self.max_workers = max_workers

        # Cleaning rules
        self.cleaning_rules = self.config.get("cleaning", {})

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning rules from config.
        """
        # Ensure datetime index
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()

        if self.cleaning_rules.get("remove_duplicates", True):
            df = df[~df.index.duplicated(keep="first")]

        if self.cleaning_rules.get("forward_fill", False):
            df = df.ffill()

        if self.cleaning_rules.get("dropna", False):
            df = df.dropna()

        return df

    def process_file(self, file: Path):
        """
        Process a single CSV file.
        """
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=[0], infer_datetime_format=True)
            df_clean = self.clean(df)
            output_path = self.processed_dir / file.name
            df_clean.to_csv(output_path)
            logger.info(f"Processed: {file.name} → {output_path}")
        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")

    def process_all(self):
        """
        Process all CSV files in raw_dir in parallel.
        """
        files = list(self.raw_dir.glob("*.csv"))
        if not files:
            logger.warning(f"No CSV files found in {self.raw_dir}")
            return

        logger.info(f"Processing {len(files)} files with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_file, f): f for f in files}
            for future in as_completed(futures):
                # Exceptions are handled inside process_file
                future.result()