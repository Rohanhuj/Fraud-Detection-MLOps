import os
import subprocess
from pathlib import Path


def test_train_candidate_runs_and_writes_outputs() -> None:
    data_dir = Path("data/sample/training_features")

    assert data_dir.exists(), (
        "Expected sample training features at data/sample/training_features. "
        "Download them from S3 before running this test."
    )

    env = os.environ.copy()
    env["TRAINING_FEATURES_DIR"] = str(data_dir)

    result = subprocess.run(
        ["python", "src/retraining/train_candidate.py"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, (
        "train_candidate.py failed.\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    expected_outputs = [
        Path("artifacts/candidate_model.pkl"),
        Path("artifacts/backfill_scored_candidate.parquet"),
    ]

    missing = [str(path) for path in expected_outputs if not path.exists()]
    assert not missing, f"Missing expected retraining outputs: {missing}"
