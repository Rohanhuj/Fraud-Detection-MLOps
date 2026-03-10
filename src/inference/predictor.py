import json
import pickle
from pathlib import Path

import pandas as pd

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "feature_columns.json"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.json"


class FraudPredictor:
    def __init__(self):
        self.model = self._load_model()
        self.feature_columns = self._load_feature_columns()
        self.threshold = self._load_threshold()

    def _load_model(self):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    def _load_feature_columns(self):
        with open(FEATURE_COLUMNS_PATH, "r") as f:
            return json.load(f)

    def _load_threshold(self):
        with open(THRESHOLD_PATH, "r") as f:
            payload = json.load(f)
        return float(payload["threshold"])

    def _prepare_dataframe(self, payload: dict) -> pd.DataFrame:
        df = pd.DataFrame([payload])
        df = pd.get_dummies(df, columns=["merchant_category", "country", "channel"], dtype = int)
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        return df

    def predict(self, payload: dict) -> dict:
        X = self._prepare_dataframe(payload)
        prob = float(self.model.predict_proba(X)[:, 1][0])
        decision = "review" if prob >= float(self.threshold) else "approve"

        return {
            "fraud_probability": float(round(prob, 6)),
            "decision": str(decision),
            "threshold": float(self.threshold),
        }