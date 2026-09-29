"""LightGBM quantile forecaster (TRD 5.1).

One model per (horizon, quantile): horizons 24/48/72h x quantiles p10/p50/p90,
so nine models. The quantiles are the point: a commissioner deciding whether to
halt a construction site needs the *band*, not a single number pretending to
certainty — p90 crossing 300 is an operational fact even when p50 does not.

Models are pooled across cities with a `city_code` feature (TRD 5.1). This is
what makes a new city viable on day one: Lucknow contributes ~6 stations, far
too few to learn a seasonal cycle alone, but it inherits the pooled model and
its own features still steer the prediction.

RESIDUAL TARGET — the single most important design decision here.
The models predict the *change* from the current reading, `y - pm25(t)`, and the
prediction is `pm25(t) + delta`. They do not predict the level directly.

Why: gradient-boosted trees are piecewise-constant and cannot extrapolate beyond
the range they were trained on. Measured on this data the holdout period (Delhi
stubble season, mean 205 ug/m3) runs 3.6x hotter than the training period
(mean 57). A level-target model asked about 400 ug/m3 returns the highest leaf it
ever learned (~150-200) and systematically under-predicts precisely the severe
episodes VAYU exists to catch — it lost to persistence by 53% on RMSE.

Predicting the residual fixes this structurally:
  * the delta distribution is far closer to stationary across regimes, so the
    tree is never asked to extrapolate;
  * persistence becomes the model's natural zero — predicting delta=0 *is*
    persistence, so the model starts from that baseline and learns corrections
    to it, rather than having to rediscover it;
  * the level anchor carries the regime, so a 500 ug/m3 morning is representable
    even though no training row reached it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from vayu_core.config import REPO_ROOT

from .features import FEATURE_COLUMNS

ARTIFACT_DIR = REPO_ROOT / "models" / "artifacts"
MODEL_VERSION = "lgbm-1.0"

HORIZONS = (24, 48, 72)
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}

# Fixed hyperparameters (TRD 5.1: "don't tune long"). Tuning these would buy
# little and cost the time the demo needs.
PARAMS = {
    "objective": "quantile",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "n_estimators": 600,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "verbose": -1,
    "n_jobs": -1,
}

MODEL_FEATURES = [*FEATURE_COLUMNS, "city_code"]


@dataclass
class TrainReport:
    horizon_h: int
    quantile: str
    n_train: int
    n_valid: int
    best_iteration: int
    valid_l1: float


def _artifact(horizon: int, q: str, base: Path | None = None) -> Path:
    return (base or ARTIFACT_DIR) / f"forecast_h{horizon}_{q}.txt"


def anchor_of(df: pd.DataFrame) -> pd.Series:
    """The level the residual is measured from: PM2.5 at issue time.

    Falls back to the 1h lag when the issue hour itself is missing — station
    feeds drop hours, and losing the anchor would drop the row entirely.
    """
    return df["pm25"].fillna(df["pm25_lag1"])


def add_city_code(df: pd.DataFrame, city_id: str) -> pd.DataFrame:
    """Stable integer code per city so pooled models can tell them apart."""
    from vayu_core.config import list_cities

    codes = {c.id: i for i, c in enumerate(list_cities())}
    out = df.copy()
    out["city_code"] = codes.get(city_id, -1)
    return out


def train(
    frames: dict[str, pd.DataFrame],
    weather: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = HORIZONS,
    valid_days: int = 60,
    artifact_dir: Path | None = None,
    train_until: pd.Timestamp | None = None,
) -> list[TrainReport]:
    """Train the pooled quantile models.

    `frames` maps city_id -> base feature frame; `weather` maps city_id -> its
    weather frame, which is joined per horizon to attach the forecast-time
    (`fx_*`) fields — the weather at the hour being predicted, which is what
    actually drives the answer.

    Validation is the last `valid_days` of history — a *time* split, never a
    random one: shuffling hours would let the model see tomorrow while
    predicting today and produce a beautiful, worthless score.

    `train_until` hard-truncates the data. The backtest uses it to build models
    that provably never saw the held-out window, and writes them to its own
    `artifact_dir` so a backtest can never quietly replace the production
    models with weaker, truncated-data ones.
    """
    import lightgbm as lgb

    from .features import model_frame

    base = artifact_dir or ARTIFACT_DIR
    base.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no features to train on — run `make seed` first")

    reports: list[TrainReport] = []
    for horizon in horizons:
        # fx_* depends on the horizon, so the frame is rebuilt per horizon and
        # only then pooled across cities.
        per_city = [
            add_city_code(model_frame(df, weather.get(cid, pd.DataFrame()), horizon), cid)
            for cid, df in frames.items()
            if not df.empty
        ]
        pooled = pd.concat(per_city, ignore_index=True)
        d = pooled.dropna(subset=["y", "pm25_lag1"])
        # Residual target: predict the change from the anchor, not the level.
        d = d.assign(_anchor=anchor_of(d))
        d = d.dropna(subset=["_anchor"])
        d = d.assign(_y_delta=d["y"] - d["_anchor"])
        if train_until is not None:
            # Drop rows whose *target* lands in the held-out window, not just
            # rows whose features do — otherwise a row at cutoff-12h carries a
            # label from inside the holdout and the evaluation is contaminated.
            d = d[d["ts"] + pd.Timedelta(hours=horizon) <= train_until]
        if d.empty:
            logger.warning(f"h{horizon}: no rows with a target — skipping")
            continue

        cutoff = d["ts"].max() - pd.Timedelta(days=valid_days)
        tr, va = d[d["ts"] <= cutoff], d[d["ts"] > cutoff]
        if len(va) < 500 or tr.empty:
            # Not enough history to hold out honestly; train on everything and
            # say so rather than reporting a validation score off 12 rows.
            logger.warning(f"h{horizon}: thin validation ({len(va)} rows) — training on all data")
            tr, va = d, d.tail(min(len(d), 1000))

        for qname, alpha in QUANTILES.items():
            model = lgb.LGBMRegressor(**PARAMS, alpha=alpha)
            model.fit(
                tr[MODEL_FEATURES],
                tr["_y_delta"],
                eval_set=[(va[MODEL_FEATURES], va["_y_delta"])],
                eval_metric="quantile",
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            model.booster_.save_model(str(_artifact(horizon, qname, base)))
            # Report MAE on the LEVEL (anchor + delta), not the residual — the
            # residual MAE would look flatteringly small and mean nothing.
            pred = va["_anchor"].to_numpy() + model.predict(va[MODEL_FEATURES])
            l1 = float(np.mean(np.abs(pred - va["y"])))
            reports.append(
                TrainReport(horizon, qname, len(tr), len(va), model.best_iteration_ or PARAMS["n_estimators"], l1)
            )
            logger.info(f"h{horizon} {qname}: train={len(tr):,} valid={len(va):,} MAE={l1:.1f} µg/m³")

    (base / "meta.json").write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "horizons": list(horizons),
                "quantiles": list(QUANTILES),
                "features": MODEL_FEATURES,
            },
            indent=2,
        )
    )
    return reports


class Forecaster:
    """Loads the trained boosters and predicts p10/p50/p90."""

    def __init__(self, artifact_dir: Path | None = None) -> None:
        import lightgbm as lgb

        self.models: dict[tuple[int, str], "lgb.Booster"] = {}
        for h in HORIZONS:
            for q in QUANTILES:
                p = _artifact(h, q, artifact_dir)
                if p.exists():
                    self.models[(h, q)] = lgb.Booster(model_file=str(p))
        self.version = MODEL_VERSION

    @property
    def available(self) -> bool:
        return bool(self.models)

    def horizons(self) -> list[int]:
        return sorted({h for h, _ in self.models})

    def predict(self, features: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
        """Return p10/p50/p90 *levels* per input row for one horizon.

        The boosters emit a residual; the anchor (PM2.5 now) is added back here,
        so callers always deal in concentrations.
        """
        if not self.available:
            raise RuntimeError("no trained models — run `make backtest` or the seeder's train step")
        X = features[MODEL_FEATURES]
        anchor = anchor_of(features).to_numpy()
        out = pd.DataFrame(index=features.index)
        for q in QUANTILES:
            m = self.models.get((horizon_h, q))
            out[q] = anchor + m.predict(X) if m else np.nan

        # Quantile models are fit independently, so nothing stops p10 > p50 on a
        # given row (quantile crossing). Sorting each row restores a coherent
        # band; presenting a band whose lower edge exceeds its median would be
        # visibly wrong on the ward chart.
        vals = np.sort(out[["p10", "p50", "p90"]].to_numpy(), axis=1)
        out["p10"], out["p50"], out["p90"] = vals[:, 0], vals[:, 1], vals[:, 2]
        # Negative concentration is unphysical.
        return out.clip(lower=0.0)

    def explain(self, features: pd.DataFrame, horizon_h: int, top_n: int = 6) -> list[dict]:
        """Top-N SHAP contributions for the median model (PRD A4).

        Uses LightGBM's exact tree SHAP via pred_contrib, so this is the real
        decomposition of this prediction, not a global importance ranking
        borrowed from elsewhere.
        """
        from .features import FEATURE_LABELS

        m = self.models.get((horizon_h, "p50"))
        if m is None or features.empty:
            return []

        contrib = m.predict(features[MODEL_FEATURES].head(1), pred_contrib=True)[0]
        pairs = [(f, float(c)) for f, c in zip(MODEL_FEATURES, contrib[:-1])]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        # The boosters explain the *residual*, so every contribution is already
        # in "µg/m³ added to / removed from the current reading" — which is
        # exactly what the panel wants to say. The baseline is the anchor plus
        # the model's own intercept, i.e. where the forecast starts before any
        # feature moves it.
        anchor = float(anchor_of(features.head(1)).iloc[0])
        base = anchor + float(contrib[-1])

        return [
            {
                "feature": f,
                "label": FEATURE_LABELS.get(f, f.replace("_", " ").capitalize()),
                "contribution": round(c, 1),
                "direction": "increases" if c > 0 else "decreases",
                "value": (None if pd.isna(v := features.iloc[0].get(f)) else round(float(v), 1)),
            }
            for f, c in pairs[:top_n]
        ] + [
            {
                "feature": "_base",
                "label": "Current level (starting point)",
                "contribution": round(base, 1),
                "direction": "base",
                "value": round(anchor, 1),
            }
        ]
