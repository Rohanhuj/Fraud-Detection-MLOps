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
    run_name: str = "xgb_candidate_v1"


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

    X_train_raw = train_df[FEATURE_COLS].copy()
    X_test_raw = test_df[FEATURE_COLS].copy()

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

    # Slightly different candidate configuration
    model = XGBClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=11,
        scale_pos_weight=scale_pos_weight,
    )

    mlflow.set_experiment("fraud_detection_mvp")
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(
            {
                "train_end_dt": cfg.train_end_dt,
                "test_start_dt": cfg.test_start_dt,
                "test_end_dt": cfg.test_end_dt,
                "model": "XGBClassifier_candidate",
                "scale_pos_weight": scale_pos_weight,
                "feature_count": X_train.shape[1],
                "n_estimators": 700,
                "max_depth": 4,
                "learning_rate": 0.03,
                "reg_lambda": 2.0,
                "min_child_weight": 2.0,
            }
        )

        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, y_score) if len(np.unique(y_test)) > 1 else float("nan")
        auprc = average_precision_score(y_test, y_score)

        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        f1 = (2 * precision[:-1] * recall[:-1]) / np.clip(
            precision[:-1] + recall[:-1], 1e-12, None
        )
        best_idx = int(np.nanargmax(f1)) if len(f1) else 0
        best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5

        mlflow.log_metrics(
            {
                "test_auroc": float(auroc),
                "test_auprc": float(auprc),
                "best_threshold": float(best_threshold),
                "test_fraud_rate": float(y_test.mean()),
            }
        )

        ARTIFACT_DIR = Path("artifacts")
        ARTIFACT_DIR.mkdir(exist_ok=True)

        with open(ARTIFACT_DIR / "candidate_model.pkl", "wb") as f:
            pickle.dump(model, f)

        backfill_df = test_df[["dt", "amount", cfg.label_col]].copy()
        backfill_df = backfill_df.rename(columns={cfg.label_col: "is_fraud"})
        backfill_df["fraud_probability"] = y_score.astype(float)
        backfill_df["decision"] = (backfill_df["fraud_probability"] >= best_threshold).astype(int)
        backfill_df["threshold"] = float(best_threshold)

        candidate_backfill_path = ARTIFACT_DIR / "backfill_scored_candidate.parquet"
        backfill_df.to_parquet(candidate_backfill_path, index=False)

        print(f"Saved candidate backfill scoring dataset to {candidate_backfill_path}")
        print(f"Candidate AUROC: {auroc:.4f}")
        print(f"Candidate AUPRC: {auprc:.4f}")
        print(f"Candidate best threshold: {best_threshold:.6f}")


if __name__ == "__main__":
    main()