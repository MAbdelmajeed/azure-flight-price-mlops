import argparse
import json
import math
import os

import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the input CSV file",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading data...")
    df = pd.read_csv(args.data)

    print(f"Dataset shape: {df.shape}")

    target = "price"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    # Flight number is excluded from the first baseline model.
    # It is a high-cardinality identifier and can be tested separately later.
    features = [
        "airline",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
        "duration",
        "days_left",
    ]

    missing_columns = [
        column for column in features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[features + [target]].copy()

    df = df.drop_duplicates()

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    df = df.dropna(subset=[target])

    X = df[features]
    y = df[target]

    categorical_features = [
        "airline",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
    ]

    numerical_features = [
        "duration",
        "days_left",
    ]

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
        ]
    )

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.1,
        max_depth=8,
        random_state=args.random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    started_run = False

    if mlflow.active_run() is None:
        mlflow.start_run()
        started_run = True

    try:
        mlflow.log_param(
            "model_type",
            "HistGradientBoostingRegressor",
        )
        mlflow.log_param(
            "test_size",
            args.test_size,
        )
        mlflow.log_param(
            "random_state",
            args.random_state,
        )
        mlflow.log_param(
            "training_rows",
            len(X_train),
        )
        mlflow.log_param(
            "testing_rows",
            len(X_test),
        )

        print("Training model...")
        pipeline.fit(X_train, y_train)

        print("Evaluating model...")
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        mse = mean_squared_error(
            y_test,
            predictions,
        )

        rmse = math.sqrt(mse)

        r2 = r2_score(
            y_test,
            predictions,
        )

        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²:   {r2:.4f}")

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        os.makedirs(
            "outputs/model",
            exist_ok=True,
        )

        model_path = "outputs/model/model.pkl"

        joblib.dump(
            pipeline,
            model_path,
        )

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        with open(
            "outputs/metrics.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=4,
            )

        print(
            f"Model saved to {model_path}"
        )

    finally:
        if started_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()