import pandas as pd
from pathlib import Path
import yaml

class DataPreprocessor:
    def __init__(self, config_path="config/data.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        rules = self.config.get("cleaning", {})

        # Ensure datetime index with explicit format
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")

        # Sort by date
        df = df.sort_index()

        if rules.get("remove_duplicates", True):
            df = df[~df.index.duplicated(keep="first")]

        if rules.get("dropna", False):
            df = df.dropna()

        if rules.get("forward_fill", False):
            df = df.ffill()

        return df

    def process_all(self):
        for file in self.raw_dir.glob("*.csv"):
            df = pd.read_csv(
                file,
                index_col=0,
                parse_dates=[0]
            )

            df_clean = self.clean(df)
            df_clean.to_csv(self.processed_dir / file.name)
            print(f"Processed: {file.name} → {self.processed_dir / file.name}")