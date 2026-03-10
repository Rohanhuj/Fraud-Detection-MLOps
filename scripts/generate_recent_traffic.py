import json
from pathlib import Path

import pandas as pd
import requests


INPUT_PATH = Path("artifacts/reference_features.parquet")
API_URL = "http://localhost:8000/score"

# Use the full API schema columns, not just the drift subset
FEATURE_COLS = [
    "amount",
    "merchant_category",
    "country",
    "channel",
    "hour_of_day",
    "day_of_week",
    "tx_count_1d",
    "tx_count_7d",
    "tx_count_30d",
    "avg_amount_30d",
    "std_amount_30d",
    "amount_zscore_user_30d",
    "new_merchant_flag",
    "new_category_flag",
    "country_switch_flag",
    "new_device_flag",
    "merchant_txn_30d",
    "merchant_fraud_rate_30d",
]


def main():
    # Better option: use a dedicated raw sample file if you save one.
    raw_sample_path = Path("artifacts/serving_sample.parquet")
    if raw_sample_path.exists():
        df = pd.read_parquet(raw_sample_path)
    else:
        raise FileNotFoundError(
            "Missing artifacts/serving_sample.parquet. "
            "Save a raw serving sample from train_baseline.py first."
        )

    df = df[FEATURE_COLS].dropna().sample(n=min(300, len(df)), random_state=7)

    success = 0
    failures = 0

    for _, row in df.iterrows():
        payload = row.to_dict()

        # convert numpy/pandas scalars to native Python types
        clean_payload = {}
        for k, v in payload.items():
            if hasattr(v, "item"):
                clean_payload[k] = v.item()
            else:
                clean_payload[k] = v

        if "day_of_week" in clean_payload:
            clean_payload["day_of_week"] = int(clean_payload["day_of_week"]) % 7

        resp = requests.post(API_URL, json=clean_payload, timeout=10)

        if resp.ok:
            success += 1
        else:
            failures += 1
            print("Request failed:", resp.status_code, resp.text)

    print(f"Finished. Success={success}, Failures={failures}")


if __name__ == "__main__":
    main()