"""Scheduled jobs for live operation (Phase L1c/L1d).

Run by EventBridge → Fargate in production (see deploy/DEPLOY.md), or by hand:

    python -m services.jobs.retrain     # weekly: retrain behind a promotion gate

Both push the updated hot DuckDB to S3 when a bucket is configured, and are
safe no-ops on the parts that need credentials they don't have.
"""
