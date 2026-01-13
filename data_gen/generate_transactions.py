import argparse
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Tuple

import numpy as np
import pandas as pd

COUNTRIES = ["US", "CA", "GB", "FR", "DE", "MX", "BR", "IN", "JP", "AU"]
CHANNELS = ["online", "in_store"]

@dataclass(frozen=True)
class GenConfig:
    start_date: str          # YYYY-MM-DD
    days: int
    tx_per_day: int
    n_users: int
    n_merchants: int
    n_devices: int
    base_fraud_rate: float   # e.g., 0.006 = 0.6%
    seed: int

def to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()

def sample_amount(rng: np.random.Generator, n: int) -> np.ndarray:
    # heavy-tailed transaction amounts (approximate card spend distribution)
    amt = rng.lognormal(mean=3.2, sigma=0.9, size=n)
    return np.clip(amt, 1.0, 5000.0)

def compute_fraud_prob(
    channel: np.ndarray,
    amount: np.ndarray,
    country: np.ndarray,
    user_home: np.ndarray,
    base_rate: float,
) -> np.ndarray:
    # Convert base rate to log-odds
    eps = 1e-9
    base_logit = np.log(base_rate + eps) - np.log(1 - base_rate + eps)

    # Risk factors (simple but interpretable)
    online_boost = np.where(channel == "online", 0.8, 0.0)
    cross_border = np.where(country != user_home, 0.7, 0.0)
    high_amount = np.where(amount > np.percentile(amount, 90), 0.9, 0.0)

    logits = base_logit + online_boost + cross_border + high_amount
    return 1.0 / (1.0 + np.exp(-logits))

def generate_day(cfg: GenConfig, day_idx: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    day_start = datetime.fromisoformat(cfg.start_date).replace(tzinfo=timezone.utc) + timedelta(days=day_idx)
    dt_str = day_start.date().isoformat()

    n = cfg.tx_per_day
    user_id = rng.integers(1, cfg.n_users + 1, size=n)
    merchant_id = rng.integers(1, cfg.n_merchants + 1, size=n)
    device_id = rng.integers(1, cfg.n_devices + 1, size=n)

    # Deterministic home country mapping per user (stable without a lookup table)
    user_home = np.array([COUNTRIES[(int(u) * 2654435761) % len(COUNTRIES)] for u in user_id])

    channel = rng.choice(CHANNELS, size=n, p=[0.65, 0.35])
    amount = sample_amount(rng, n)

    # Country: usually home, sometimes foreign
    is_foreign = rng.random(size=n) < 0.12
    country = np.where(is_foreign, rng.choice(COUNTRIES, size=n), user_home)

    # Timestamp within the day
    seconds = rng.integers(0, 24 * 3600, size=n)
    ts = [to_iso(day_start + timedelta(seconds=int(s))) for s in seconds]

    fraud_prob = compute_fraud_prob(channel, amount, country, user_home, cfg.base_fraud_rate)
    is_fraud_hidden = rng.random(size=n) < fraud_prob

    tx_df = pd.DataFrame({
        "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
        "user_id": user_id.astype(int),
        "merchant_id": merchant_id.astype(int),
        "device_id": device_id.astype(int),
        "amount": amount.astype(float),
        "country": country.astype(str),
        "timestamp": ts,
        "channel": channel.astype(str),
        "dt": dt_str,  # partition key
    })

    fraud_tx = tx_df[is_fraud_hidden].copy()

    if len(fraud_tx) == 0:
        cb_df = pd.DataFrame({"transaction_id": [], "chargeback_date": [], "dt": []})
    else:
        # Chargeback delay 3–45 days after transaction
        delays = rng.integers(3, 46, size=len(fraud_tx))
        cb_dates = [
            to_iso(datetime.fromisoformat(t).replace(tzinfo=timezone.utc) + timedelta(days=int(d)))
            for t, d in zip(fraud_tx["timestamp"].tolist(), delays)
        ]
        cb_df = pd.DataFrame({
            "transaction_id": fraud_tx["transaction_id"].tolist(),
            "chargeback_date": cb_dates,
            "dt": dt_str,  # partition by transaction date for now (keeps ingestion simple)
        })

    return tx_df, cb_df

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--tx-per-day", type=int, default=20000)
    p.add_argument("--n-users", type=int, default=5000)
    p.add_argument("--n-merchants", type=int, default=800)
    p.add_argument("--n-devices", type=int, default=7000)
    p.add_argument("--base-fraud-rate", type=float, default=0.006)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--outdir", default="output")
    args = p.parse_args()

    cfg = GenConfig(
        start_date=args.start_date,
        days=args.days,
        tx_per_day=args.tx_per_day,
        n_users=args.n_users,
        n_merchants=args.n_merchants,
        n_devices=args.n_devices,
        base_fraud_rate=args.base_fraud_rate,
        seed=args.seed,
    )

    rng = np.random.default_rng(cfg.seed)
    os.makedirs(args.outdir, exist_ok=True)

    tx_parts = []
    cb_parts = []

    for i in range(cfg.days):
        tx, cb = generate_day(cfg, i, rng)
        tx_parts.append(tx)
        cb_parts.append(cb)

    tx_all = pd.concat(tx_parts, ignore_index=True)
    cb_all = pd.concat(cb_parts, ignore_index=True)

    tx_path = os.path.join(args.outdir, "transactions.parquet")
    cb_path = os.path.join(args.outdir, "chargebacks.parquet")
    tx_all.to_parquet(tx_path, index=False)
    cb_all.to_parquet(cb_path, index=False)

    # Sanity prints
    cb_rate = (len(cb_all) / max(len(tx_all), 1)) * 100.0
    if len(cb_all) > 0:
        tx_ts = pd.to_datetime(tx_all["timestamp"], utc=True)
        cb_ts = pd.to_datetime(cb_all["chargeback_date"], utc=True)
        # map chargebacks back to their tx timestamps for delay distribution
        tx_map = tx_all.set_index("transaction_id")["timestamp"]
        tx_for_cb = pd.to_datetime(cb_all["transaction_id"].map(tx_map), utc=True)
        delays = (cb_ts - tx_for_cb).dt.days
        delay_stats = f"delay_days(min/median/max)={delays.min()}/{int(delays.median())}/{delays.max()}"
    else:
        delay_stats = "delay_days(min/median/max)=n/a"

    print(f"Wrote {len(tx_all):,} transactions -> {tx_path}")
    print(f"Wrote {len(cb_all):,} chargebacks   -> {cb_path}")
    print(f"Observed chargeback rate: {cb_rate:.3f}% | {delay_stats}")

if __name__ == "__main__":
    main()