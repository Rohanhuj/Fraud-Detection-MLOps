import os
import glob
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn


@dataclass(frozen=True)
class Config:
    data_dir: str = "data/features/training_features"
    train_end_dt: str = "2026-01-05"
    test_start_dt: str = "2026-01-06"
    test_end_dt: str = "2026-01-07"
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

def load_parquet_dataset(root_dir:str) -> pd.DataFrame:
    files = glob.glob(os.path.join(root_dir,"**","*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No files found in {root_dir}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            continue
    if not dfs:
        raise FileNotFoundError(f"No valid parquet files found in {root_dir}")
    return pd.concat(dfs, ignore_index=True)

def time_split(df: pd.DataFrame, cfg:Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[cfg.dt_col] <= cfg.train_end_dt].copy()
    test = df[(df[cfg.dt_col] >= cfg.test_start_dt) & (df[cfg.dt_col] <= cfg.test_end_dt)].copy()
    if train.empty or test.empty:
        raise ValueError(f"Train or test split is empty. Check date filters and dt_col values.")
    return train, test

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac : float) -> float:
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

    df['avg_amount_30d'] = df['avg_amount_30d'].fillna(0.0)
    df['std_amount_30d'] = df['std_amount_30d'].fillna(0.0)
    df['amount_zscore_user_30d'] = df['amount_zscore_user_30d'].fillna(0.0)

    train_df, test_df = time_split(df, cfg)

    y_train = train_df[cfg.label_col].astype(int).to_numpy()
    y_test = test_df[cfg.label_col].astype(int).to_numpy()

    X_train = pd.get_dummies(train_df[FEATURE_COLS], columns = CATEGORICAL_COLS, dummy_na=True)
    X_test = pd.get_dummies(test_df[FEATURE_COLS], columns = CATEGORICAL_COLS, dummy_na=True)

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
        mlflow.log_params({
            "train_end_dt": cfg.train_end_dt,
            "test_start_dt": cfg.test_start_dt,
            "test_end_dt": cfg.test_end_dt,
            "model": "XGBClassifier",
            "scale_pos_weight": scale_pos_weight,
            "feature_count": X_train.shape[1],
        })
    
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, y_score) if len(np.unique(y_test)) > 1 else float('nan')
        aupr = average_precision_score(y_test, y_score)

        p_at_1pct = precision_at_k(y_test, y_score, 0.01)
        r_at_1pct = recall_at_k(y_test, y_score, 0.01)

        p_at_5pct = precision_at_k(y_test, y_score, 0.05)
        r_at_5pct = recall_at_k(y_test, y_score, 0.05)

        mlflow.log_metrics({
            "test_auroc": auroc,
            "test_auprc": auprc,
            "precision_at_1pct": p_at_1pct,
            "recall_at_1pct": r_at_1pct,
            "precision_at_5pct": p_at_5pct,
            "recall_at_5pct": r_at_5pct,
            "test_fraud_rate": float(y_test.mean()),
        })

        os.makedirs("artifacts", exist_ok=True)
        feature_spec_path = os.path.join("artifacts", "feature_columns.json")
        with open(feature_spec_path, "w") as f:
            json.dump({"columns" : list(X_train.columns)}, f, indent = 2)
        
        mlflow.log_artifact(feature_spec_path)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("done")
        print(f"Test fraud rate: {y_test.mean():.6f}")
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Precision@1%: {p_at_1pct:.4f} | Recall@1%: {r_at_1pct:.4f}")
        print(f"Precision@5%: {p_at_5pct:.4f} | Recall@5%: {r_at_5pct:.4f}")

if __name__ == "__main__":
    main()