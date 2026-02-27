from src.pipeline.train_dataset import TrainingDatasetBuilder

class TrainPipeline:
    def run(self):
        builder = TrainingDatasetBuilder("config/training.yaml")
        builder.build_all()
        print("Training datasets created.")