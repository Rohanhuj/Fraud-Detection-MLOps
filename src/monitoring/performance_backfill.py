from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


INPUT_PATH = Path("artifacts/backfill_scored.parquet")
OUTPUT_PATH = Path("data/monitoring/backfill_metrics.csv")

REVIEW_COST = 5.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].mean())


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    idx = np.argsort(-y_score)[:k]
    total_pos = max(1, int(y_true.sum()))
    return float(y_true[idx].sum() / total_pos)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    y_true = df["is_fraud"].astype(int).to_numpy()
    y_score = df["fraud_probability"].astype(float).to_numpy()
    y_pred = df["decision"].astype(int).to_numpy()

    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_score)

    p_at_1pct = precision_at_k(y_true, y_score, 0.01)
    r_at_1pct = recall_at_k(y_true, y_score, 0.01)
    p_at_5pct = precision_at_k(y_true, y_score, 0.05)
    r_at_5pct = recall_at_k(y_true, y_score, 0.05)

    review_rate = float(y_pred.mean())
    review_volume = int(y_pred.sum())

    fraud_dollars_captured = float(df.loc[(df["is_fraud"] == 1) & (df["decision"] == 1), "amount"].sum())
    missed_fraud_dollars = float(df.loc[(df["is_fraud"] == 1) & (df["decision"] == 0), "amount"].sum())
    review_cost = float(review_volume * REVIEW_COST)

    net_benefit = fraud_dollars_captured - missed_fraud_dollars - review_cost

    metrics = pd.DataFrame([{
        "rows_evaluated": len(df),
        "fraud_rate": float(df["is_fraud"].mean()),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "precision_at_1pct": float(p_at_1pct),
        "recall_at_1pct": float(r_at_1pct),
        "precision_at_5pct": float(p_at_5pct),
        "recall_at_5pct": float(r_at_5pct),
        "review_rate": review_rate,
        "review_volume": review_volume,
        "fraud_dollars_captured": fraud_dollars_captured,
        "missed_fraud_dollars": missed_fraud_dollars,
        "review_cost": review_cost,
        "net_benefit": net_benefit,
    }])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Backfill Metrics ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved backfill metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()