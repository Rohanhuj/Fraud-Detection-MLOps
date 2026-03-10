from pathlib import Path
import pandas as pd

DRIFT_PATH = Path("data/monitoring/drift_report.csv")
DAILY_PATH = Path("data/monitoring/daily_metrics.csv")


def main():
    alerts = []

    if DRIFT_PATH.exists():
        drift = pd.read_csv(DRIFT_PATH)
        bad_drift = drift[drift["psi"] >= 0.2]
        for _, row in bad_drift.iterrows():
            alerts.append(f"Significant Drift detected in feature '{row['feature']}' (PSI={row['psi']:.3f})")
    
    if DAILY_PATH.exists():
        daily = pd.read_csv(DAILY_PATH)
        if not daily.empty:
            latest = daily.iloc[-1]
            if latest["review_rate"] > 0.30:
                alerts.append(f"High review rate detected on {latest['date']} ({latest['review_rate']:.2%})")
            if latest["p95_score"] > 0.8:
                alerts.append(f"High 95th percentile score detected on {latest['date']} ({latest['p95_score']:.3f})")

    if alerts:
        print("ALERTS:")
        for alert in alerts:
            print(f"- {alert}")
    else:
        print("No alerts detected.")

if __name__ == "__main__":
    main()    