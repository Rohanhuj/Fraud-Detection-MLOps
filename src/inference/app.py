from fastapi import FastAPI
from src.inference.schema import TransactionFeatures, ScoreResponse
from src.inference.predictor import FraudPredictor
from src.monitoring.score_logger import log_prediction

app = FastAPI(title="Fraud Scoring Service", version="1.0.0")
predictor = FraudPredictor()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score_transaction(features: TransactionFeatures):
    payload = features.model_dump()
    result = predictor.predict(payload)
    request_id = log_prediction(payload, result)

    return {
        **result,
        "request_id": request_id,
    }