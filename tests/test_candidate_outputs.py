from pathlib import Path

import pandas as pd


def test_candidate_backfill_output_has_expected_columns() -> None:
    output_path = Path("artifacts/backfill_scored_candidate.parquet")

    assert output_path.exists(), (
        "Missing artifacts/backfill_scored_candidate.parquet. "
        "Run train_candidate.py first."
    )

    df = pd.read_parquet(output_path)

    expected_cols = {
        "dt",
        "amount",
        "is_fraud",
        "fraud_probability",
        "decision",
        "threshold",
    }

    missing = sorted(expected_cols - set(df.columns))
    assert not missing, f"Missing columns in candidate backfill output: {missing}"

    assert not df.empty, "Candidate backfill output is empty"
