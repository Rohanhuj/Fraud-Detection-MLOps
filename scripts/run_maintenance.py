#!/usr/bin/env bash
set -euo pipefail

echo "===== Starting maintenance run ====="

export TRAINING_FEATURES_DIR="${TRAINING_FEATURES_DIR:-data/sample/training_features}"

mkdir -p data/monitoring
mkdir -p artifacts

echo "===== Running score monitoring ====="
python src/monitoring/score_monitor.py || true

echo "===== Running drift detection ====="
python src/monitoring/drift.py

echo "===== Running threshold alerts ====="
python src/monitoring/alerts.py || true

echo "===== Running performance backfill ====="
python src/monitoring/performance_backfill.py || true

echo "===== Running backfill alerts ====="
python src/monitoring/backfill_alerts.py || true

echo "===== Deciding whether retraining is needed ====="
python scripts/should_retrain.py

echo "===== Maintenance run complete ====="
