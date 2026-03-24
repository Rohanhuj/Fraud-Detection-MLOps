from pathlib import Path
import pandas as pd
import subprocess
import sys


DRIFT_PATH = Path("data/monitoring/drift_report.csv")
BACKFILL_PATH = Path("data/monitoring/backfill_metrics.csv")
RECOMMENDATION_PATH = Path("data/monitoring/model_recommendation.csv")

# Starter thresholds — tune later
PSI_RETRAIN_THRESHOLD = 0.20
MAX_REVIEW_RATE = 0.30
MIN_NET_BENEFIT = 0.0


def should_trigger_from_drift() -> bool:
    if not DRIFT_PATH.exists():
        print("No drift report found. Skipping retrain trigger from drift.")
        return False

    df = pd.read_csv(DRIFT_PATH)
    if df.empty or "psi" not in df.columns:
        print("Drift report missing psi column or empty.")
        return False

    max_psi = float(df["psi"].max())
    print(f"Max PSI: {max_psi:.6f}")
    return max_psi >= PSI_RETRAIN_THRESHOLD


def should_trigger_from_backfill() -> bool:
    if not BACKFILL_PATH.exists():
        print("No backfill metrics found. Skipping retrain trigger from backfill.")
        return False

    df = pd.read_csv(BACKFILL_PATH)
    if df.empty:
        return False

    latest = df.iloc[-1]

    review_rate = float(latest["review_rate"]) if "review_rate" in latest else 0.0
    net_benefit = float(latest["net_benefit"]) if "net_benefit" in latest else 0.0

    print(f"Latest review_rate: {review_rate:.6f}")
    print(f"Latest net_benefit: {net_benefit:.2f}")

    return (review_rate > MAX_REVIEW_RATE) or (net_benefit < MIN_NET_BENEFIT)


def run_retraining() -> None:
    result = subprocess.run(
        ["python", "src/retraining/retrain_and_compare.py"],
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Retraining pipeline failed.")


def maybe_promote() -> None:
    if not RECOMMENDATION_PATH.exists():
        print("No recommendation file found after retraining.")
        return

    df = pd.read_csv(RECOMMENDATION_PATH)
    if df.empty or "promote_candidate" not in df.columns:
        print("Recommendation file missing promote_candidate column.")
        return

    decision = str(df.iloc[0]["promote_candidate"]).strip().lower()
    print(f"Promotion decision: {decision}")

    if decision == "true" or decision == "promote":
        print("Promoting candidate...")
        promote = subprocess.run(["bash", "scripts/promote_candidate.sh"], check=False)
        if promote.returncode != 0:
            raise RuntimeError("Promotion script failed.")
    else:
        print("Candidate rejected. Production remains unchanged.")


def main() -> None:
    trigger = should_trigger_from_drift() or should_trigger_from_backfill()

    if not trigger:
        print("No retraining trigger fired.")
        sys.exit(0)

    print("Retraining trigger fired.")
    run_retraining()
    maybe_promote()


if __name__ == "__main__":
    main()
