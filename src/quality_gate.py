import argparse
import json
import math
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics.json",
    )

    parser.add_argument(
        "--min-r2",
        type=float,
        required=True,
        help="Minimum acceptable R2 score",
    )

    parser.add_argument(
        "--max-mae",
        type=float,
        required=True,
        help="Maximum acceptable MAE",
    )

    parser.add_argument(
        "--max-rmse",
        type=float,
        required=True,
        help="Maximum acceptable RMSE",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.metrics, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    required_metrics = ["r2", "mae", "rmse"]

    missing = [
        metric
        for metric in required_metrics
        if metric not in metrics
    ]

    if missing:
        print(f"Quality gate failed: missing metrics {missing}")
        sys.exit(1)

    r2 = float(metrics["r2"])
    mae = float(metrics["mae"])
    rmse = float(metrics["rmse"])

    if any(math.isnan(value) for value in [r2, mae, rmse]):
        print("Quality gate failed: one or more metrics are NaN")
        sys.exit(1)

    print("Model quality gate")
    print("------------------")
    print(f"R2:   {r2:.4f}   required >= {args.min_r2}")
    print(f"MAE:  {mae:.2f}   required <= {args.max_mae}")
    print(f"RMSE: {rmse:.2f}   required <= {args.max_rmse}")

    failures = []

    if r2 < args.min_r2:
        failures.append(
            f"R2 {r2:.4f} is below {args.min_r2}"
        )

    if mae > args.max_mae:
        failures.append(
            f"MAE {mae:.2f} exceeds {args.max_mae}"
        )

    if rmse > args.max_rmse:
        failures.append(
            f"RMSE {rmse:.2f} exceeds {args.max_rmse}"
        )

    if failures:
        print("\nQUALITY GATE: FAILED")

        for failure in failures:
            print(f"- {failure}")

        sys.exit(1)

    print("\nQUALITY GATE: PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()