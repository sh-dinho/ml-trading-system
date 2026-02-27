from src.features.engineer import FeatureEngineeringPipeline

def test_feature_pipeline():
    pipeline = FeatureEngineeringPipeline(
        data_config="config/data.yaml",
        feature_config="config/features.yaml"
    )
    pipeline.run()

if __name__ == "__main__":
    test_feature_pipeline()