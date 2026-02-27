from src.data.fetcher import DataFetcher
from src.data.preprocess import DataPreprocessor

fetcher = DataFetcher()
raw_data = fetcher.fetch()

pre = DataPreprocessor()
pre.process_all()