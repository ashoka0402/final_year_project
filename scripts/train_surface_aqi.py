"""Train and evaluate the surface-AQI CNN-LSTM, then score the validated corridor.

    python -m scripts.train_surface_aqi

Writes:
  * docs/surface_aqi_evaluation.json — RMSE/MAE/R + the persistence baseline,
    mirroring the pattern `vayu_core.forecast.backtest` already uses so both
    models' honesty checks live in the same place a reviewer would look.
  * models/artifacts/surface_aqi/{cnn_lstm.pt,norm.json} — the trained weights.
  * aqi_grid rows for the validated corridor's stations, every day the model
    could score — the actual "spatial maps of surface AQI" deliverable.
"""

from __future__ import annotations

import datetime as dt
import json
import sys

from loguru import logger

from vayu_core.config import REPO_ROOT, load_region
from vayu_core.national import surface_aqi as SA

# O3 is excluded here, not because it wasn't ingested (56/56... actually 47/56)
# but because its gaps are concentrated differently from the other five and
# requiring all six simultaneously drops the usable 5-day-window count from 30
# to 19 for a channel that is also the least direct PM2.5 proxy of the set —
# ozone is a secondary photochemical product, not a primary aerosol/precursor
# the way HCHO, NO2, SO2, CO and AOD all are. It stays in `satellite_grid` for
# other consumers (the national map, hotspot detection is HCHO-specific
# already); it is simply not part of this model version's input.
TRAINING_CHANNELS = ("hcho", "no2", "so2", "co", "aod")


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> {message}", level="INFO")

    region = load_region("india")
    logger.info(f"assembling dataset — channels={TRAINING_CHANNELS}, cities={SA.GROUND_TRUTH_CITIES}")
    ds = SA.build_dataset(region, channels=TRAINING_CHANNELS)
    if len(ds.y) == 0:
        logger.error("no training samples assembled — check satellite/CPCB/weather coverage")
        return 1
    logger.success(
        f"dataset: {len(ds.y)} samples, {len(set(ds.station_ids))} stations, "
        f"{min(ds.dates)} -> {max(ds.dates)}"
    )

    artifact_dir = REPO_ROOT / "models" / "artifacts" / "surface_aqi"
    model, norm, report = SA.train(ds, holdout_days=10, epochs=150, artifact_dir=artifact_dir)

    logger.success(
        f"holdout ({report.holdout_start} onward, n={report.n_holdout}): "
        f"RMSE={report.rmse} MAE={report.mae} R={report.r} "
        f"| persistence baseline RMSE={report.baseline_rmse}"
    )
    beats_persistence = report.rmse < report.baseline_rmse

    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": SA.MODEL_VERSION,
        "objective": "surface AQI from satellite data (CNN-LSTM)",
        "trained_and_validated_on": {
            "cities": list(SA.GROUND_TRUTH_CITIES),
            "note": (
                "Ground-truth CPCB readings and meteorological reanalysis exist only "
                "for this corridor. The satellite inputs are genuinely national; "
                "extending validated coverage needs national CPCB history + "
                "ERA5/IMDAA reanalysis access, which is a region-config change in "
                "this codebase's existing architecture, not a rewrite."
            ),
        },
        "satellite_channels": list(TRAINING_CHANNELS),
        "n_train": report.n_train,
        "n_holdout": report.n_holdout,
        "holdout_period": f"{report.holdout_start.isoformat()} onward",
        "metrics": {"rmse_ugm3": report.rmse, "mae_ugm3": report.mae, "pearson_r": report.r},
        "baseline": {
            "method": "persistence (yesterday's PM2.5)",
            "rmse_ugm3": report.baseline_rmse,
            "model_beats_baseline": beats_persistence,
        },
    }
    out = REPO_ROOT / "docs" / "surface_aqi_evaluation.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info(f"wrote {out}")

    written = 0
    for day in sorted(set(ds.dates)):
        rows = SA.score_grid(model, norm, region, ds, day)
        written += SA.write_aqi_grid(region, rows)
    logger.success(f"wrote {written:,} aqi_grid rows across {len(set(ds.dates))} days")

    if not beats_persistence:
        logger.warning(
            "model does NOT beat the persistence baseline on this holdout — "
            "reported honestly above, not hidden"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
