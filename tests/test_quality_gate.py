import json
import subprocess
import sys
from pathlib import Path


def test_quality_gate_passes(tmp_path):
    metrics = {
        "r2": 0.97,
        "mae": 2318.68,
        "rmse": 3923.93,
    }

    metrics_path = tmp_path / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file)

    project_root = Path(__file__).resolve().parents[1]
    gate_script = project_root / "src" / "quality_gate.py"

    result = subprocess.run(
        [
            sys.executable,
            str(gate_script),
            "--metrics",
            str(metrics_path),
            "--min-r2",
            "0.95",
            "--max-mae",
            "2500",
            "--max-rmse",
            "4500",
        ]
    )

    assert result.returncode == 0


def test_quality_gate_fails(tmp_path):
    metrics = {
        "r2": 0.80,
        "mae": 4000,
        "rmse": 6000,
    }

    metrics_path = tmp_path / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file)

    project_root = Path(__file__).resolve().parents[1]
    gate_script = project_root / "src" / "quality_gate.py"

    result = subprocess.run(
        [
            sys.executable,
            str(gate_script),
            "--metrics",
            str(metrics_path),
            "--min-r2",
            "0.95",
            "--max-mae",
            "2500",
            "--max-rmse",
            "4500",
        ]
    )

    assert result.returncode != 0