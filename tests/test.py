from src.data.fetcher import DataFetcher
from src.data.preprocess import DataPreprocessor

fetcher = DataFetcher("config/data.yaml")
fetcher.fetch_data()

pre = DataPreprocessor("config/data.yaml")
pre.process_all()