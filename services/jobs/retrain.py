"""Weekly retrain behind a promotion gate (L1d).

The honesty claim — "our forecast beats persistence" — can only stay true in
production if every retrain is *earned*. So this never blindly replaces the live
model:

    1. Back up the current artifacts.
    2. Retrain in place on the latest data.
    3. Backtest on a held-out trailing window.
    4. Promote only if the new model beats persistence at t+24h.
       Otherwise roll back to the backup.

A retrain that can't beat "tomorrow looks like today" is worse than no retrain,
and this gate makes that call automatically. Run: `python -m services.jobs.retrain`.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from loguru import logger

from vayu_core.forecast.model import ARTIFACT_DIR


def _backup_dir() -> Path:
    return ARTIFACT_DIR.parent / "artifacts_prev"


def _rmtree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def _beats_persistence(result: dict) -> bool:
    """True if VAYU's t+24h RMSE is below persistence's on the holdout.

    `backtest.run()` returns `metrics` as a list of dicts (asdict of Metrics),
    each with `model`, `horizon_h`, `rmse`. It trains its evaluation models on
    pre-cutoff data only (to its own dir), so this asks the honest question:
    does our *method* still beat "tomorrow looks like today" on the latest data?
    """
    metrics = result.get("metrics", [])
    vayu = {m["horizon_h"]: m for m in metrics if m.get("model") == "VAYU"}
    pers = {m["horizon_h"]: m for m in metrics if m.get("model") == "Persistence"}
    if 24 not in vayu or 24 not in pers:
        logger.warning("backtest missing t+24h metrics — treating as NOT promotable")
        return False
    margin = pers[24]["rmse"] - vayu[24]["rmse"]
    logger.info(
        f"t+24h RMSE: VAYU={vayu[24]['rmse']:.1f} vs persistence={pers[24]['rmse']:.1f} (margin {margin:+.1f})"
    )
    return vayu[24]["rmse"] < pers[24]["rmse"]


def retrain() -> int:
    from services.pipeline.score import train_and_score
    from vayu_core.forecast import backtest

    # 1. Back up current artifacts (if any).
    backup = _backup_dir()
    _rmtree(backup)
    if ARTIFACT_DIR.exists():
        shutil.copytree(ARTIFACT_DIR, backup)
        logger.info(f"backed up current model → {backup.name}/")

    # 2. Retrain in place + rescore.
    rc = train_and_score(force_train=True)
    if rc != 0:
        logger.error("retrain failed during train/score — rolling back")
        _restore(backup)
        return 1

    # 3. Backtest the fresh model.
    try:
        result = backtest.run()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"backtest errored ({exc}) — rolling back")
        _restore(backup)
        return 1

    # 4. Promotion gate.
    if _beats_persistence(result):
        logger.success("PROMOTED — new model beats persistence at t+24h")
        _rmtree(backup)
        from services.api.scoring import reset_forecaster

        reset_forecaster()
        return 0

    logger.warning("REJECTED — new model does not beat persistence; rolling back")
    _restore(backup)
    return 0


def _restore(backup: Path) -> None:
    if not backup.exists():
        logger.warning("no backup to restore (first-ever train?) — leaving new artifacts in place")
        return
    _rmtree(ARTIFACT_DIR)
    shutil.copytree(backup, ARTIFACT_DIR)
    _rmtree(backup)
    logger.info("rolled back to previous model")


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> {message}", level="INFO")
    return retrain()


if __name__ == "__main__":
    raise SystemExit(main())
