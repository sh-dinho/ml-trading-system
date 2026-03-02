import pandas as pd
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.paths import ensure_dir, data_path
from src.utils.logger import get_logger

logger = get_logger("data_preprocessor")


class DataPreprocessor:
    def __init__(self, config_file: str = "config/data.yaml", max_workers: int = 4):
        self.config = self._load_config(config_file)
        output_cfg = self.config.get("output", {})

        # Directories from improved config
        self.raw_dir = ensure_dir(Path(output_cfg.get("raw_dir", "data/raw")))
        self.processed_dir = ensure_dir(Path(output_cfg.get("processed_dir", "data/processed")))

        self.max_workers = max_workers
        self.cleaning_cfg = self.config.get("cleaning", {})

        # Output format (csv, parquet)
        self.output_format = output_cfg.get("format", "csv").lower()

    # -----------------------------
    # Config loading
    # -----------------------------
    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg or {}
        except Exception as e:
            logger.error(f"Failed to load config {path}: {e}")
            raise

    # -----------------------------
    # Cleaning logic
    # -----------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()

        if self.cleaning_cfg.get("remove_duplicates", True):
            df = df[~df.index.duplicated(keep="first")]

        if self.cleaning_cfg.get("forward_fill", False):
            df = df.ffill()

        if self.cleaning_cfg.get("backward_fill", False):
            df = df.bfill()

        if self.cleaning_cfg.get("enforce_numeric", False):
            df = df.apply(pd.to_numeric, errors="coerce")

        if self.cleaning_cfg.get("dropna", False):
            df = df.dropna()

        # Optional outlier clipping
        clip_cfg = self.cleaning_cfg.get("clip_outliers", {})
        if clip_cfg.get("enabled", False):
            method = clip_cfg.get("method", "zscore")
            threshold = clip_cfg.get("threshold", 3.0)
            df = self._clip_outliers(df, method, threshold)

        return df

    def _clip_outliers(self, df: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
        if method == "zscore":
            z = (df - df.mean()) / df.std()
            return df.mask(z.abs() > threshold)
        elif method == "quantile":
            lower = df.quantile(threshold)
            upper = df.quantile(1 - threshold)
            return df.clip(lower, upper, axis=1)
        return df

        # -----------------------------
    # File processing
    # -----------------------------
    def process_file(self, file: Path):
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=[0], infer_datetime_format=True)
        except Exception as e:
            logger.error(f"Failed to read {file.name}: {e}")
            return

        try:
            df_clean = self.clean(df)
            output_path = self.processed_dir / file.name

            if self.output_format == "parquet":
                output_path = output_path.with_suffix(".parquet")
                df_clean.to_parquet(output_path)
            else:
                df_clean.to_csv(output_path)

            logger.info(f"Processed: {file.name} -> {output_path}")
        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")

    # -----------------------------
    # Batch processing
    # -----------------------------
    def process_all(self):
        files = list(self.raw_dir.glob("*.csv"))
        if not files:
            logger.warning(f"No CSV files found in {self.raw_dir}")
            return

        logger.info(f"Processing {len(files)} files with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_file, f): f for f in files}
            for future in as_completed(futures):
                future.result()