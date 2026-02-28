from pathlib import Path
import pandas as pd

from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry

class TrainPipeline:
    def __init__(self, dataset_dir="data/datasets", model_dir="models"):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

    def run_for_ticker(self, ticker):
        dataset_path = self.dataset_dir / f"{ticker}.csv"
        df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

        X = df.drop("target", axis=1)
        y = df["target"]

        trainer = ModelTrainer()
        model, metrics = trainer.train(X, y)

        registry = ModelRegistry(self.model_dir)
        saved_path = registry.save_model(model, f"{ticker}.pkl")

        print(f"\nModel trained for {ticker}")
        print("Saved to:", saved_path)
        print("Metrics:", metrics)

    def run_all(self):
        for file in self.dataset_dir.glob("*.csv"):
            ticker = file.stem
            self.run_for_ticker(ticker)


if __name__ == "__main__":
    TrainPipeline().run_all()