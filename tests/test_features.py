# tests/test_features.py
from src.features.engineer import FeatureEngineer

def test_feature_pipeline():
    pipeline = FeatureEngineer(
        data_config="config/data.yaml",
        feature_config="config/features.yaml"
    )
    pipeline.process_all()  # call process_all instead of run()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_feature_pipeline()