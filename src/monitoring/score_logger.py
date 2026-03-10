import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


LOG_DIR = Path("data/monitoring")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "inference_logs.jsonl"


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def log_prediction(payload: dict, result: dict, model_version: str = "baseline_v1") -> str:
    request_id = str(uuid4())
    print("log_prediction called")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "fraud_probability": result["fraud_probability"],
        "decision": result["decision"],
        "threshold": result["threshold"],
        **payload,
    }

    record = to_jsonable(record)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return request_id