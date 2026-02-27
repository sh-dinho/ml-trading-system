from src.data.fetcher import DataFetcher
from src.data.preprocess import DataPreprocessor
from src.features.builder import FeatureBuilder

class FeatureEngineeringPipeline:
    def __init__(self,
                 data_config="config/data.yaml",
                 feature_config="config/features.yaml"):
        self.fetcher = DataFetcher(data_config)
        self.preprocessor = DataPreprocessor(data_config)
        self.builder = FeatureBuilder(feature_config)

    def run(self):
        print("=== Fetching raw data ===")
        self.fetcher.fetch_data()

        print("\n=== Preprocessing data ===")
        self.preprocessor.process_all()

        print("\n=== Building features ===")
        self.builder.process_all()

        print("\n=== Feature engineering complete ===")