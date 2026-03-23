import os
import glob
import json
import re
import pickle
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn


@dataclass(frozen=True)
class Config:
    data_dir: str = os.getenv("TRAINING_FEATURES_DIR", "data/features/training_features")
    train_end_dt: str = "2025-12-20"
    test_start_dt: str = "2025-12-21"
    test_end_dt: str = "2025-12-30"
    label_col: str = "is_fraud"
    dt_col: str = "dt"
    run_name: str = "xgb_baseline"


FEATURE_COLS: List[str] = [
    "amount",
    "log_amount",
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

CATEGORICAL_COLS: List[str] = ["merchant_category", "country", "channel"]

DT_RE = re.compile(r"dt=(\d{4}-\d{2}-\d{2})")


def load_parquet_dataset(root_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No files found in {root_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue

        # If Athena partition column wasn't written into Parquet, recover from path
        if "dt" not in df.columns:
            m = DT_RE.search(f)
            if m:
                df["dt"] = m.group(1)
            else:
                raise ValueError(f"Could not infer dt from path: {f}")

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No readable parquet files under {root_dir}.")

    return pd.concat(dfs, ignore_index=True)


def time_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[cfg.dt_col] <= cfg.train_end_dt].copy()
    test = df[
        (df[cfg.dt_col] >= cfg.test_start_dt) & (df[cfg.dt_col] <= cfg.test_end_dt)
    ].copy()
    if train.empty or test.empty:
        raise ValueError("Train or test split is empty. Check date filters and dt_col values.")
    return train, test


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(n * k_frac))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].mean())


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    idx = np.argsort(-y_score)[:k]
    total_pos = max(1, int(y_true.sum()))
    return float(y_true[idx].sum() / total_pos)


def main() -> None:
    cfg = Config()

    df = load_parquet_dataset(cfg.data_dir)
    df = df.sample(frac=0.2, random_state=7)

    df["avg_amount_30d"] = df["avg_amount_30d"].fillna(0.0)
    df["std_amount_30d"] = df["std_amount_30d"].fillna(0.0)
    df["amount_zscore_user_30d"] = df["amount_zscore_user_30d"].fillna(0.0)

    train_df, test_df = time_split(df, cfg)

    y_train = train_df[cfg.label_col].astype(int).to_numpy()
    y_test = test_df[cfg.label_col].astype(int).to_numpy()

    # Keep raw feature frames BEFORE one-hot encoding
    X_train_raw = train_df[FEATURE_COLS].copy()
    X_test_raw = test_df[FEATURE_COLS].copy()

    serving_sample_cols = [
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

    X_test_raw[serving_sample_cols].to_parquet(
        "artifacts/serving_sample.parquet",
        index=False
    )

    print("Saved serving sample to artifacts/serving_sample.parquet")

    # Save reference dataset for drift monitoring
    reference_cols = [
        "amount",
        "country",
        "merchant_category",
        "tx_count_7d",
        "avg_amount_30d",
        "amount_zscore_user_30d",
        "merchant_fraud_rate_30d",
    ]

    Path("artifacts").mkdir(exist_ok=True)
    X_train_raw[reference_cols].to_parquet(
        "artifacts/reference_features.parquet",
        index=False,
    )
    print("Saved reference features to artifacts/reference_features.parquet")

    # One-hot encode from raw frames
    X_train = pd.get_dummies(
        X_train_raw,
        columns=CATEGORICAL_COLS,
        dummy_na=True,
    )
    X_test = pd.get_dummies(
        X_test_raw,
        columns=CATEGORICAL_COLS,
        dummy_na=True,
    )

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    n_pos = max(1, int(y_train.sum()))
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=7,
        scale_pos_weight=scale_pos_weight,
    )

    mlflow.set_experiment("fraud_detection_mvp")
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(
            {
                "train_end_dt": cfg.train_end_dt,
                "test_start_dt": cfg.test_start_dt,
                "test_end_dt": cfg.test_end_dt,
                "model": "XGBClassifier",
                "scale_pos_weight": scale_pos_weight,
                "feature_count": X_train.shape[1],
            }
        )

        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]

        # -----------------------------
        # Diagnostics: score separation
        # -----------------------------
        fraud_scores = y_score[y_test == 1]
        legit_scores = y_score[y_test == 0]

        def q(arr, p):
            return float(np.quantile(arr, p)) if len(arr) else float("nan")

        print("\n=== Score separation diagnostics ===")
        print(
            f"Test set size: {len(y_test):,} | frauds: {int(y_test.sum()):,} | fraud rate: {y_test.mean():.6f}"
        )

        print("\nScore distribution (legit):")
        print(
            f"  mean={legit_scores.mean():.6f}  q50={q(legit_scores,0.50):.6f}  "
            f"q90={q(legit_scores,0.90):.6f}  q99={q(legit_scores,0.99):.6f}  "
            f"q999={q(legit_scores,0.999):.6f}"
        )

        print("\nScore distribution (fraud):")
        print(
            f"  mean={fraud_scores.mean():.6f}  q50={q(fraud_scores,0.50):.6f}  "
            f"q90={q(fraud_scores,0.90):.6f}  q99={q(fraud_scores,0.99):.6f}  "
            f"q999={q(fraud_scores,0.999):.6f}"
        )

        # -----------------------------
        # Diagnostics: precision/recall at multiple K
        # -----------------------------
        for k in [0.005, 0.01, 0.02, 0.05, 0.10]:
            p = precision_at_k(y_test, y_score, k)
            r = recall_at_k(y_test, y_score, k)
            print(f"Precision@{int(k * 100)}%: {p:.4f} | Recall@{int(k * 100)}%: {r:.4f}")

        # -----------------------------
        # Diagnostics: best F1 threshold (rough)
        # -----------------------------
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        f1 = (2 * precision[:-1] * recall[:-1]) / np.clip(
            precision[:-1] + recall[:-1], 1e-12, None
        )
        best_idx = int(np.nanargmax(f1)) if len(f1) else 0
        best_thr = float(thresholds[best_idx]) if len(thresholds) else 0.5

        print("\n=== Threshold diagnostic ===")
        print(
            f"Best F1 threshold (approx): {best_thr:.6f} | "
            f"precision={precision[best_idx]:.4f} | recall={recall[best_idx]:.4f} | "
            f"f1={f1[best_idx]:.4f}"
        )

        # -----------------------------
        # Diagnostics: feature importance (top 20)
        # -----------------------------
        try:
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            print("\n=== Top 20 feature importances ===")
            for i in top_idx:
                print(f"{X_train.columns[i]}: {importances[i]:.6f}")
        except Exception as e:
            print(f"Could not print feature importances: {e}")

        auroc = roc_auc_score(y_test, y_score) if len(np.unique(y_test)) > 1 else float("nan")
        auprc = average_precision_score(y_test, y_score)

        p_at_1pct = precision_at_k(y_test, y_score, 0.01)
        r_at_1pct = recall_at_k(y_test, y_score, 0.01)

        p_at_5pct = precision_at_k(y_test, y_score, 0.05)
        r_at_5pct = recall_at_k(y_test, y_score, 0.05)


        mlflow.log_metrics(
            {
                "test_auroc": auroc,
                "test_auprc": auprc,
                "precision_at_1pct": p_at_1pct,
                "recall_at_1pct": r_at_1pct,
                "precision_at_5pct": p_at_5pct,
                "recall_at_5pct": r_at_5pct,
                "test_fraud_rate": float(y_test.mean()),
            }
        )

        os.makedirs("artifacts", exist_ok=True)

        feature_spec_path = os.path.join("artifacts", "feature_columns.json")
        with open(feature_spec_path, "w") as f:
            json.dump({"columns": list(X_train.columns)}, f, indent=2)

        mlflow.log_artifact(feature_spec_path)
        mlflow.sklearn.log_model(model, artifact_path="model")

        best_threshold = 0.2

        ARTIFACT_DIR = Path("artifacts")
        ARTIFACT_DIR.mkdir(exist_ok=True)

        # Save scored backfill dataset for delayed-label evaluation
        backfill_df = test_df[["dt", "amount", cfg.label_col]].copy()
        backfill_df = backfill_df.rename(columns={cfg.label_col: "is_fraud"})
        backfill_df["fraud_probability"] = y_score.astype(float)
        backfill_df["decision"] = (backfill_df["fraud_probability"] >= best_threshold).astype(int)
        backfill_df["threshold"] = float(best_threshold)

        backfill_path = ARTIFACT_DIR / "backfill_scored.parquet"
        backfill_df.to_parquet(backfill_path, index=False)
        print(f"Saved backfill scoring dataset to {backfill_path}")

        with open(ARTIFACT_DIR / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(ARTIFACT_DIR / "feature_columns.json", "w") as f:
            json.dump(list(X_train.columns), f, indent=2)

        with open(ARTIFACT_DIR / "threshold.json", "w") as f:
            json.dump({"threshold": best_threshold}, f, indent=2)

        print("done")
        print(f"Test fraud rate: {y_test.mean():.6f}")
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Precision@1%: {p_at_1pct:.4f} | Recall@1%: {r_at_1pct:.4f}")
        print(f"Precision@5%: {p_at_5pct:.4f} | Recall@5%: {r_at_5pct:.4f}")


if __name__ == "__main__":
    main()