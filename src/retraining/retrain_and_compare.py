import subprocess
import sys
from pathlib import Path

import pandas as pd


COMPARE_OUTPUT = Path("data/monitoring/model_recommendation.csv")


def run_step(cmd: list[str], step_name: str) -> None:
    print(f"\n=== Running: {step_name} ===")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")


def main() -> None:
    # 1. Train candidate and generate candidate backfill file
    run_step(
        [sys.executable, "src/retraining/train_candidate.py"],
        "train_candidate",
    )

    # 2. Compare production vs candidate
    run_step(
        [sys.executable, "src/monitoring/compare_models.py"],
        "compare_models",
    )

    # 3. Read final recommendation
    if not COMPARE_OUTPUT.exists():
        raise FileNotFoundError(f"Missing recommendation file: {COMPARE_OUTPUT}")

    rec = pd.read_csv(COMPARE_OUTPUT)
    if rec.empty:
        raise ValueError("Recommendation file is empty.")

    latest = rec.iloc[-1]

    print("\n=== Final Recommendation ===")
    print(latest.to_string())

    recommendation_col = None
    for col in ["recommendation", "promote_candidate"]:
        if col in latest.index:
            recommendation_col = col
            break

    if recommendation_col is None:
        raise KeyError("Could not find recommendation column in recommendation file.")

    if str(latest[recommendation_col]).strip().lower() == "promote":
        print("\nAction: Candidate model is approved for promotion.")
    else:
        print("\nAction: Candidate model is rejected; keep current production model.")


if __name__ == "__main__":
    main()