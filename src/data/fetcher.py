import yfinance as yf
import pandas as pd
from pathlib import Path
import yaml

class DataFetcher:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self):
        tickers = self.config.get('tickers', [])
        start = self.config.get('start_date')
        end = self.config.get('end_date')

        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end)
                if df.empty:
                    print(f"Warning: No data returned for {ticker}")
                    continue

                self.save_data(ticker, df)
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

    def save_data(self, ticker: str, data: pd.DataFrame):
        file_path = self.output_dir / f"{ticker}.csv"
        data.to_csv(file_path)
        print(f"Data for {ticker} saved to {file_path}")