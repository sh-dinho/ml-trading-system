from src.data.fetcher import DataFetcher
from src.data.preprocess import DataPreprocessor

def test_ingestion_pipeline():
    print("=== Testing Data Fetcher ===")
    fetcher = DataFetcher("config/data.yaml")
    fetcher.fetch_data()

    print("\n=== Testing Data Preprocessor ===")
    preprocessor = DataPreprocessor("config/data.yaml")
    preprocessor.process_all()

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_ingestion_pipeline()