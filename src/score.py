import json
import os
from pathlib import Path

import joblib
import pandas as pd


def init():
    global model

    model_dir = Path(os.environ["AZUREML_MODEL_DIR"])

    model_files = list(model_dir.rglob("model.pkl"))

    if not model_files:
        raise FileNotFoundError("model.pkl was not found.")

    model = joblib.load(model_files[0])

    print(f"Model loaded from {model_files[0]}")


def run(raw_data):
    try:
        payload = json.loads(raw_data)

        df = pd.DataFrame(payload["data"])

        predictions = model.predict(df)

        return {
            "predictions": predictions.tolist()
        }

    except Exception as exc:
        return {
            "error": str(exc)
        }