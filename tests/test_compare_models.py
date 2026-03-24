import subprocess
from pathlib import Path

import pandas as pd


def test_compare_models_runs_when_required_files_exist() -> None:
    required_inputs = [
        Path("artifacts/backfill_scored.parquet"),
        Path("artifacts/backfill_scored_candidate.parquet"),
    ]

    missing_inputs = [str(path) for path in required_inputs if not path.exists()]
    assert not missing_inputs, (
        "Missing required comparison inputs:\n" + "\n".join(missing_inputs)
    )

    result = subprocess.run(
        ["python", "src/monitoring/compare_models.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "compare_models.py failed.\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    expected_outputs = [
        Path("data/monitoring/model_comparison.csv"),
        Path("data/monitoring/model_recommendation.csv"),
    ]

    missing_outputs = [str(path) for path in expected_outputs if not path.exists()]
    assert not missing_outputs, (
        f"Missing expected comparison outputs: {missing_outputs}"
    )


def test_model_recommendation_file_has_expected_columns() -> None:
    rec_path = Path("data/monitoring/model_recommendation.csv")

    assert rec_path.exists(), (
        "Missing data/monitoring/model_recommendation.csv. "
        "Run compare_models.py first."
    )

    df = pd.read_csv(rec_path)

    expected_cols = {
        "delta_auprc",
        "delta_recall_at_5pct",
        "delta_net_benefit",
        "promote_candidate",
    }

    missing = sorted(expected_cols - set(df.columns))
    assert not missing, f"Missing recommendation columns: {missing}"

    assert len(df) == 1, "Recommendation file should have exactly one row"
