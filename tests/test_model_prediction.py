from pathlib import Path
import pandas as pd

from src.models.predictor import ModelPredictor

def test_model_prediction():
    # Load the latest saved model
    predictor = ModelPredictor(model_dir="models")

    # Load a dataset row to test prediction
    dataset_path = Path("data/datasets/AAPL.csv")
    assert dataset_path.exists(), "Dataset file not found. Run training first."

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

    # Use the last row for prediction
    X = df.drop("target", axis=1).iloc[-1]

    # Run prediction
    pred = predictor.predict(X)
    print("Prediction:", pred)

    # If classification, try probability
    try:
        proba = predictor.predict_proba(X)
        print("Probabilities:", proba)
    except Exception:
        print("Model does not support predict_proba.")

    print("Model prediction test completed successfully.")

if __name__ == "__main__":
    test_model_prediction()