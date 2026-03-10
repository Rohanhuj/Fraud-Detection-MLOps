from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


PROD_PATH = Path("artifacts/backfill_scored.parquet")
CANDIDATE_PATH = Path("artifacts/backfill_scored_candidate.parquet")
OUTPUT_PATH = Path("data/monitoring/model_comparison.csv")

REVIEW_COST = 5.0

# Promotion rules
MIN_AUPRC_LIFT = 0.001
MIN_RECALL_AT_5PCT_LIFT = 0.01
MIN_NET_BENEFIT_LIFT = 0.0


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

def summarize(df: pd.DataFrame) -> dict:
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

    fraud_dollars_captured = float(
        df.loc[(df["is_fraud"] == 1) & (df["decision"] == 1), "amount"].sum()
    )
    missed_fraud_dollars = float(
        df.loc[(df["is_fraud"] == 1) & (df["decision"] == 0), "amount"].sum()
    )
    review_cost = float(review_volume * REVIEW_COST)
    net_benefit = fraud_dollars_captured - missed_fraud_dollars - review_cost

    return {
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
    }

def main():
    if not PROD_PATH.exists():
        raise FileNotFoundError(f"Missing production scored file: {PROD_PATH}")
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(f"Missing candidate scored file: {CANDIDATE_PATH}")

    prod_df = pd.read_parquet(PROD_PATH)
    cand_df = pd.read_parquet(CANDIDATE_PATH)

    prod = summarize(prod_df)
    cand = summarize(cand_df)

    comparison = pd.DataFrame([
        {"model" : "production", **prod},
        {"model" : "candidate", **cand},
    ])

    delta_auprc = cand["auprc"] - prod["auprc"]
    delta_recall_at_5pct = cand["recall_at_5pct"] - prod["recall_at_5pct"]
    delta_net_benefit = cand["net_benefit"] - prod["net_benefit"]

    promote = (
        (delta_auprc >= MIN_AUPRC_LIFT) and
        (delta_recall_at_5pct >= MIN_RECALL_AT_5PCT_LIFT) and
        (delta_net_benefit >= MIN_NET_BENEFIT_LIFT)
    )

    recommendation = pd.DataFrame([{
        "delta_auprc" : float(delta_auprc),
        "delta_recall_at_5pct" : float(delta_recall_at_5pct),
        "delta_net_benefit" : float(delta_net_benefit),
        "promote_candidate": promote if promote else "reject",
    }])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)
    print("=== Model Comparison ===")
    print(comparison.to_string(index=False))
    print("=== Recommendation ===")
    print(recommendation.to_string(index=False))

    rec_path = OUTPUT_PATH.parent / "model_recommendation.csv"
    recommendation.to_csv(rec_path, index=False)

    print(f"\nSaved comparison to {OUTPUT_PATH}")
    print(f"Saved recommendation to {rec_path}")


if __name__ == "__main__":
    main()