import json
from pathlib import Path

import pandas as pd

LOG_FILE = Path("data/monitoring/inference_logs.jsonl")
OUTPUT_FILE = Path("data/monitoring/daily_metrics.csv")

def main():
    if not LOG_FILE.exists():
        print("No inference logs found.")
        return

    records = []
    with open(LOG_FILE, "r", encoding = 'utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    daily = (
        df.groupby("date")
        .agg(
            scored_transactions = ("request_id", "count"),
            mean_score = ("fraud_probability", "mean"),
            median_score = ("fraud_probability", "median"),
            p95_score = ("fraud_probability", lambda x: x.quantile(0.95)),
            review_rate = ("decision", lambda x: (x == "review").mean()),
        )
        .reset_index()
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_FILE, index=False)
    print(daily)


if __name__ == "__main__":
    main()