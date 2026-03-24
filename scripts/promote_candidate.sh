#!/usr/bin/env bash
set -e

BUCKET="fraud-ml-rohan-12345"
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

echo "Archiving current production artifacts..."
aws s3 cp s3://$BUCKET/production_artifacts/current/ \
         s3://$BUCKET/production_artifacts/archive/$TIMESTAMP/ \
         --recursive

echo "Promoting candidate artifacts to current production..."
aws s3 cp artifacts/candidate_model.pkl \
         s3://$BUCKET/production_artifacts/current/model.pkl

aws s3 cp artifacts/feature_columns.json \
         s3://$BUCKET/production_artifacts/current/feature_columns.json

aws s3 cp artifacts/threshold.json \
         s3://$BUCKET/production_artifacts/current/threshold.json

aws s3 cp artifacts/backfill_scored_candidate.parquet \
         s3://$BUCKET/production_artifacts/current/backfill_scored.parquet

echo "Promotion complete."
