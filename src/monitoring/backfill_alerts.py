from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/monitoring/backfill_metrics.csv")

# Alert thresholds
MIN_AUPRC = 0.01
MIN_RECALL_AT_5PCT = 0.10
MAX_REVIEW_RATE = 0.30
MAX_MISSED_FRAUD_DOLLARS = 5000.0
MIN_NET_BENEFIT = 0.0


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing backfill metrics file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    if df.empty:
        raise ValueError("Backfill metrics file is empty.")

    latest = df.iloc[-1]
    alerts = []

    if latest["auprc"] < MIN_AUPRC:
        alerts.append(
            f"Low PR-AUC detected ({latest['auprc']:.6f} < {MIN_AUPRC:.6f})"
        )

    if latest["recall_at_5pct"] < MIN_RECALL_AT_5PCT:
        alerts.append(
            f"Low Recall@5% detected ({latest['recall_at_5pct']:.4f} < {MIN_RECALL_AT_5PCT:.4f})"
        )

    if latest["review_rate"] > MAX_REVIEW_RATE:
        alerts.append(
            f"High review rate detected ({latest['review_rate']:.2%} > {MAX_REVIEW_RATE:.2%})"
        )

    if latest["missed_fraud_dollars"] > MAX_MISSED_FRAUD_DOLLARS:
        alerts.append(
            f"High missed fraud dollars detected (${latest['missed_fraud_dollars']:.2f} > ${MAX_MISSED_FRAUD_DOLLARS:.2f})"
        )

    if latest["net_benefit"] < MIN_NET_BENEFIT:
        alerts.append(
            f"Negative net benefit detected (${latest['net_benefit']:.2f} < ${MIN_NET_BENEFIT:.2f})"
        )

    if alerts:
        print("=== Backfill Alerts ===")
        for alert in alerts:
            print(f"- {alert}")
    else:
        print("No backfill alerts triggered.")


if __name__ == "__main__":
    main()