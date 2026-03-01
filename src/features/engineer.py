import pandas as pd
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.paths import ensure_dir
from src.utils.logger import get_logger
from .indicators import sma, ema, rsi, bollinger_bands, returns, log_returns, rolling_volatility

logger = get_logger("feature_engineer")

class FeatureEngineer:
    def __init__(self, data_config="config/data.yaml", feature_config="config/features.yaml", max_workers=4):
        with open(feature_config) as f:
            self.feature_cfg = yaml.safe_load(f)
        with open(data_config) as f:
            self.data_cfg = yaml.safe_load(f)
        self.processed_dir = ensure_dir(Path(self.data_cfg.get("output_dir", "data/processed")))
        self.features_dir = ensure_dir(Path("data/features"))
        self.max_workers = max_workers

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(pd.to_numeric, errors="coerce")
        features = pd.DataFrame(index=df.index)
        for w in self.feature_cfg.get("sma", []): features[f"SMA_{w}"] = sma(df, w)
        for w in self.feature_cfg.get("ema", []): features[f"EMA_{w}"] = ema(df, w)
        r_cfg = self.feature_cfg.get("rsi", {})
        if r_cfg.get("enabled", False): features[f"RSI_{r_cfg.get('window',14)}"] = rsi(df,r_cfg.get('window',14))
        b_cfg = self.feature_cfg.get("bollinger",{})
        if b_cfg.get("enabled",False): features = features.join(bollinger_bands(df,b_cfg.get('window',20),b_cfg.get('num_std',2)))
        if self.feature_cfg.get("returns",{}).get("enabled",False): features["returns"]=returns(df)
        if self.feature_cfg.get("log_returns",{}).get("enabled",False): features["log_returns"]=log_returns(df)
        if self.feature_cfg.get("rolling_volatility",{}).get("enabled",False): features[f"volatility_{self.feature_cfg['rolling_volatility'].get('window',20)}"]=rolling_volatility(df,self.feature_cfg['rolling_volatility'].get('window',20))
        return features

    def process_all(self):
        files = list(self.processed_dir.glob("*.csv"))
        if not files:
            logger.warning("No CSVs to process.")
            return
        def process_file(file):
            df = pd.read_csv(file,index_col=0,parse_dates=True)
            features = self.build_features(df)
            out = self.features_dir/file.name
            features.to_csv(out)
            logger.info(f"Saved features: {out}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(process_file,f):f for f in files}
            for future in as_completed(futures): future.result()