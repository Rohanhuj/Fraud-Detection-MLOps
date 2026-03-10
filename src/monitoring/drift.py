import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_PATH = Path("artifacts/reference_features.parquet")
LOG_PATH = Path("data/monitoring/inference_logs.jsonl")
OUTPUT_PATH = Path("data/monitoring/drift_report.csv")

NUMERIC_FEATURES = [
    "amount",
    "tx_count_7d",
    "avg_amount_30d",
    "amount_zscore_user_30d",
    "merchant_fraud_rate_30d",
]

CATEGORICAL_FEATURES = [
    "country",
    "merchant_category",
]

EPSILON = 1e-6


def load_recent_from_logs(log_path: Path) -> pd.DataFrame:
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    cur = pd.to_numeric(cur, errors="coerce").dropna()

    if ref.empty or cur.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(ref.quantile(quantiles).values)

    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_bins = pd.cut(ref, bins=breakpoints, include_lowest=True)
    cur_bins = pd.cut(cur, bins=breakpoints, include_lowest=True)

    ref_dist = ref_bins.value_counts(normalize=True, sort=False)
    cur_dist = cur_bins.value_counts(normalize=True, sort=False)

    ref_dist = ref_dist + EPSILON
    cur_dist = cur_dist.reindex(ref_dist.index, fill_value=0.0) + EPSILON

    return float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref = ref.astype(str).fillna("MISSING")
    cur = cur.astype(str).fillna("MISSING")

    ref_dist = ref.value_counts(normalize=True)
    cur_dist = cur.value_counts(normalize=True)

    all_categories = sorted(set(ref_dist.index).union(set(cur_dist.index)))

    ref_dist = ref_dist.reindex(all_categories, fill_value=0.0) + EPSILON
    cur_dist = cur_dist.reindex(all_categories, fill_value=0.0) + EPSILON

    return float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())


def classify_psi(psi_value: float) -> str:
    if pd.isna(psi_value):
        return "unknown"
    if psi_value < 0.1:
        return "low"
    if psi_value < 0.2:
        return "moderate"
    return "significant"


def main():
    reference_df = pd.read_parquet(REFERENCE_PATH)
    recent_df = load_recent_from_logs(LOG_PATH)

    rows = []

    for feature in NUMERIC_FEATURES:
        psi_value = psi_numeric(reference_df[feature], recent_df[feature])
        rows.append({
            "feature": feature,
            "feature_type": "numeric",
            "psi": round(psi_value, 6) if not pd.isna(psi_value) else np.nan,
            "drift_level": classify_psi(psi_value),
        })

    for feature in CATEGORICAL_FEATURES:
        psi_value = psi_categorical(reference_df[feature], recent_df[feature])
        rows.append({
            "feature": feature,
            "feature_type": "categorical",
            "psi": round(psi_value, 6) if not pd.isna(psi_value) else np.nan,
            "drift_level": classify_psi(psi_value),
        })

    report = pd.DataFrame(rows).sort_values(by="psi", ascending=False, na_position="last")
    report.to_csv(OUTPUT_PATH, index=False)

    print(report.to_string(index=False))
    print(f"\nSaved drift report to {OUTPUT_PATH}")

    print(f"Reference rows: {len(reference_df)}")
    print(f"Recent rows: {len(recent_df)}")

if __name__ == "__main__":
    main()