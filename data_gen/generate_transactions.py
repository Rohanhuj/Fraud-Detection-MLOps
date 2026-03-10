#!/usr/bin/env python3


import argparse
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# ---------- Domain setup ----------

CATEGORIES: List[str] = [
    "grocery",
    "gas",
    "restaurants",
    "travel",
    "electronics",
    "digital_goods",
    "retail",
    "other",
]

# Category-specific amount distributions (lognormal params).
# amount ~ LogNormal(meanlog, sigma)  (USD-ish)
AMOUNT_LN_PARAMS: Dict[str, Tuple[float, float]] = {
    "grocery": (3.2, 0.45),       # ~ $25
    "gas": (3.4, 0.35),           # ~ $30
    "restaurants": (3.1, 0.55),   # ~ $22
    "travel": (4.3, 0.75),        # heavier tail
    "electronics": (4.1, 0.85),   # heavier tail
    "digital_goods": (3.6, 0.90), # variable
    "retail": (3.6, 0.60),        # ~ $36
    "other": (3.5, 0.70),
}


@dataclass
class Config:
    start_date: str
    days: int
    tx_per_day: int
    base_fraud_rate: float
    seed: int
    outdir: str

    # Entity space sizes (tune later)
    n_users: int = 5000
    n_merchants: int = 6000
    n_devices: int = 9000

    # Chargeback delay window
    min_delay_days: int = 3
    max_delay_days: int = 45


# ---------- Helpers ----------

def iso_ts(dt: datetime) -> str:
    # ISO8601 with Z (UTC) so Athena from_iso8601_timestamp works
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def merchant_category_from_id(merchant_id: int) -> str:
    return CATEGORIES[merchant_id % len(CATEGORIES)]


def pick_channel(rng: np.random.Generator, cat: str) -> str:
    # Semantics: gas/grocery mostly in-store, digital_goods online only, travel mostly online
    if cat in ("gas", "grocery"):
        return "in_store" if rng.random() < 0.95 else "online"
    if cat == "digital_goods":
        return "online"
    if cat == "travel":
        return "online" if rng.random() < 0.90 else "in_store"
    return "online" if rng.random() < 0.65 else "in_store"


def sample_amount(rng: np.random.Generator, cat: str) -> float:
    meanlog, sigma = AMOUNT_LN_PARAMS.get(cat, AMOUNT_LN_PARAMS["other"])
    amt = float(rng.lognormal(mean=meanlog, sigma=sigma))
    # clip to keep sanity in small sims
    return float(np.clip(amt, 1.0, 5000.0))


def compute_base_fraud_prob(
    base_rate: float,
    channel: str,
    is_cross_border: int,
    amount: float,
) -> float:
    """
    Start with base_rate and apply interpretable multipliers.
    Feature-aligned boosts (new_device/new_merchant/mismatch/burst) are applied later.
    """
    p = base_rate

    if channel == "online":
        p *= 1.8
    if is_cross_border:
        p *= 1.7
    if amount >= 300:
        p *= 2.0
    if amount >= 1000:
        p *= 1.5

    return p


def sample_chargeback_delay_days(rng: np.random.Generator, min_d: int, max_d: int) -> int:
    """
    Sample delay with mid/late mass; scaled Beta distribution.
    """
    x = float(rng.beta(2.4, 2.1))
    d = min_d + int(round(x * (max_d - min_d)))
    return int(np.clip(d, min_d, max_d))


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--days", type=int, required=True)
    ap.add_argument("--tx-per-day", type=int, required=True)
    ap.add_argument("--base-fraud-rate", type=float, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    cfg = Config(
        start_date=args.start_date,
        days=args.days,
        tx_per_day=args.tx_per_day,
        base_fraud_rate=args.base_fraud_rate,
        seed=args.seed,
        outdir=args.outdir,
    )

    rng = np.random.default_rng(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)

    # Output roots (partitioned by dt)
    tx_root = os.path.join(cfg.outdir, "transactions")
    cb_root = os.path.join(cfg.outdir, "chargebacks")
    os.makedirs(tx_root, exist_ok=True)
    os.makedirs(cb_root, exist_ok=True)

    # Stable per-user "home country"
    countries = np.array(["US", "CA", "BR", "IN", "AU", "GB", "DE", "FR", "JP", "MX"])
    user_home_country: Dict[int, str] = {
        u: str(rng.choice(countries, p=[0.55, 0.08, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03]))
        for u in range(1, cfg.n_users + 1)
    }

    # State across days to create learnable patterns
    seen_user_device = set()    # (user_id, device_id)
    seen_user_merchant = set()  # (user_id, merchant_id)

    # Burst: per-user deque of tx epoch seconds in last 1 hour
    recent_user_ts = defaultdict(deque)  # user_id -> deque[int]

    start_dt = datetime.strptime(cfg.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    total_tx = 0
    total_cb = 0
    all_delays: List[int] = []

    for day_idx in range(cfg.days):
        day_dt = start_dt + timedelta(days=day_idx)
        dt_str = day_dt.strftime("%Y-%m-%d")

        n = cfg.tx_per_day

        # Sample entities
        user_id = rng.integers(1, cfg.n_users + 1, size=n, dtype=np.int32)
        merchant_id = rng.integers(1, cfg.n_merchants + 1, size=n, dtype=np.int32)
        device_id = rng.integers(1, cfg.n_devices + 1, size=n, dtype=np.int32)

        # Random seconds in day, sorted to simulate streaming order
        seconds_in_day = rng.integers(0, 24 * 3600, size=n, dtype=np.int32)
        order = np.argsort(seconds_in_day)

        user_id = user_id[order]
        merchant_id = merchant_id[order]
        device_id = device_id[order]
        seconds_in_day = seconds_in_day[order]

        # RAM-safe: only keep ONE DAY in memory
        day_tx_rows = []
        day_cb_rows = []

        for i in range(n):
            u = int(user_id[i])
            m = int(merchant_id[i])
            d = int(device_id[i])

            tx_time = day_dt + timedelta(seconds=int(seconds_in_day[i]))
            tx_epoch = int(tx_time.timestamp())

            cat = merchant_category_from_id(m)
            ch = pick_channel(rng, cat)
            amt = sample_amount(rng, cat)

            home = user_home_country[u]

            # Country: mostly home; travel category more likely to be foreign
            if cat == "travel" and rng.random() < 0.35:
                ctry = str(rng.choice(countries))
            elif rng.random() < 0.08:
                ctry = str(rng.choice(countries))
            else:
                ctry = home

            is_cross_border = int(ctry != home)

            # Newness signals (align with features you compute later)
            ud = (u, d)
            um = (u, m)
            new_device = 1 if ud not in seen_user_device else 0
            new_merchant = 1 if um not in seen_user_merchant else 0
            if new_device:
                seen_user_device.add(ud)
            if new_merchant:
                seen_user_merchant.add(um)

            # Burst count in last hour (prior tx only)
            dq = recent_user_ts[u]
            while dq and dq[0] < tx_epoch - 3600:
                dq.popleft()
            burst_1h = len(dq)  # number of prior tx in last hour
            dq.append(tx_epoch)

            # Semantic anomaly / mismatch
            mismatch = int(
                (cat == "gas" and ch == "online")
                or (cat == "digital_goods" and ch == "in_store")
                or (cat == "grocery" and ch == "online" and rng.random() < 0.5)
            )

            # Base fraud prob + feature-aligned boosts
            p = compute_base_fraud_prob(cfg.base_fraud_rate, ch, is_cross_border, amt)

            # Strong boosts that your engineered features should capture
            p *= (1.0 + 2.0 * new_device)     # account takeover via new device
            p *= (1.0 + 1.3 * new_merchant)   # unfamiliar merchant
            p *= (1.0 + 2.5 * mismatch)       # weird semantics
            if burst_1h >= 3:
                p *= 2.0
            if burst_1h >= 6:
                p *= 2.0

            p = float(np.clip(p, 0.0, 0.35))
            is_fraud = int(rng.random() < p)

            tx_id = str(uuid.uuid4())

            day_tx_rows.append({
                "transaction_id": tx_id,
                "user_id": u,
                "merchant_id": m,
                "device_id": d,
                "merchant_category": cat,
                "amount": amt,
                "country": ctry,
                "timestamp": iso_ts(tx_time),
                "channel": ch,
                "dt": dt_str,
            })

            if is_fraud:
                delay_days = sample_chargeback_delay_days(rng, cfg.min_delay_days, cfg.max_delay_days)
                cb_time = tx_time + timedelta(days=delay_days)
                day_cb_rows.append({
                    "transaction_id": tx_id,
                    "chargeback_date": iso_ts(cb_time),
                    # partition by original transaction day (keeps joins easy)
                    "dt": dt_str,
                    "delay_days": delay_days,
                })

        # Write one day to Parquet immediately (RAM-safe)
        tx_df = pd.DataFrame(day_tx_rows)
        cb_df = pd.DataFrame(day_cb_rows)

        tx_out_dir = os.path.join(tx_root, f"dt={dt_str}")
        cb_out_dir = os.path.join(cb_root, f"dt={dt_str}")
        os.makedirs(tx_out_dir, exist_ok=True)
        os.makedirs(cb_out_dir, exist_ok=True)

        tx_df.to_parquet(os.path.join(tx_out_dir, "part-00000.parquet"), index=False)
        cb_df.to_parquet(os.path.join(cb_out_dir, "part-00000.parquet"), index=False)

        total_tx += len(tx_df)
        total_cb += len(cb_df)
        if len(cb_df):
            all_delays.extend(cb_df["delay_days"].tolist())

        # Lightweight progress print
        rate_day = (len(cb_df) / len(tx_df)) if len(tx_df) else 0.0
        print(f"[{dt_str}] wrote tx={len(tx_df):,} cb={len(cb_df):,} rate={rate_day:.3%}")

    # Final summary
    rate = (total_cb / total_tx) if total_tx else 0.0
    if total_cb:
        delays = np.array(all_delays, dtype=np.int32)
        dmin, dmed, dmax = int(delays.min()), int(np.median(delays)), int(delays.max())
    else:
        dmin = dmed = dmax = 0

    print("\n=== DONE ===")
    print(f"Wrote {total_tx:,} transactions -> {tx_root}/dt=.../part-00000.parquet")
    print(f"Wrote {total_cb:,} chargebacks   -> {cb_root}/dt=.../part-00000.parquet")
    print(f"Observed chargeback rate: {rate:.3%} | delay_days(min/median/max)={dmin}/{dmed}/{dmax}")


if __name__ == "__main__":
    main()