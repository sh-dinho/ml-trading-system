from src.pipeline.train_dataset import TrainingDatasetBuilder
from pathlib import Path

def test_training_dataset_builder():
    builder = TrainingDatasetBuilder("config/training.yaml")
    builder.build_all()

    output_dir = Path("data/datasets")
    assert output_dir.exists(), "Dataset directory was not created."

    files = list(output_dir.glob("*.csv"))
    assert len(files) > 0, "No training datasets were generated."

    print("Training dataset test passed. Files generated:")
    for f in files:
        print(" -", f)

if __name__ == "__main__":
    test_training_dataset_builder()