import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd


def test_training_pipeline(tmp_path):

    rows = []

    airlines = ["Vistara", "Air_India", "Indigo"]
    cities = ["Delhi", "Mumbai", "Bangalore"]

    for i in range(120):
        rows.append(
            {
                "airline": airlines[i % 3],
                "source_city": cities[i % 3],
                "departure_time": "Morning" if i % 2 == 0 else "Evening",
                "stops": "zero" if i % 2 == 0 else "one",
                "arrival_time": "Afternoon" if i % 2 == 0 else "Night",
                "destination_city": cities[(i + 1) % 3],
                "class": "Economy" if i % 4 else "Business",
                "duration": 2.0 + (i % 10),
                "days_left": 1 + (i % 30),
                "price": 5000 + (i * 100),
            }
        )

    df = pd.DataFrame(rows)

    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)

    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "src" / "train.py"

    result = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--data",
            str(data_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    model_path = tmp_path / "outputs" / "model" / "model.pkl"
    metrics_path = tmp_path / "outputs" / "metrics.json"

    assert model_path.exists()
    assert metrics_path.exists()

    model = joblib.load(model_path)

    sample = df.drop(columns=["price"]).head(5)

    predictions = model.predict(sample)

    assert len(predictions) == 5