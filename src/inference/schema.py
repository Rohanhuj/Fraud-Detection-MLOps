from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    amount: float
    merchant_category: str
    country: str
    channel: str
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    tx_count_1d: int
    tx_count_7d: int
    tx_count_30d: int
    avg_amount_30d: float
    std_amount_30d: float
    amount_zscore_user_30d: float
    new_merchant_flag: int
    new_category_flag: int
    country_switch_flag: int
    new_device_flag: int
    merchant_txn_30d: int
    merchant_fraud_rate_30d: float


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: str
    threshold: float
    request_id: str