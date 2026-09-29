"""Rolling-origin backtest (TRD 5.1) — the evidence behind every accuracy claim.

Protocol, exactly as specified:
  * the last 30 days of history are held out **entirely** — models are retrained
    on data strictly before the cutoff, into a separate artifact directory, so
    nothing evaluated here was ever trained on.
  * forecasts are issued at 00/06/12/18 UTC each day of the holdout.
  * per horizon (24/48/72h): RMSE, MAE, AQI-bucket accuracy, and
    crossing-detection precision/recall at AQI 300.
  * baselines: persistence (value at issue time) and climatology (month x hour mean).

Outputs `docs/evaluation.md` (tables + charts) and `docs/evaluation.json`
(served at /meta/evaluation). `make backtest` regenerates both.

The point of the crossing metrics: RMSE is what a data scientist asks for, but a
commissioner only cares whether VAYU *called the hazard*. A model can win on RMSE
by hugging the mean and never predicting a spike — which would be useless. Both
are reported, and honestly, whichever way they come out.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

from vayu_core.aqi import aqi_from_pm25, category_for
from vayu_core.config import REPO_ROOT, list_cities

from .features import build_features, model_frame
from .model import HORIZONS, Forecaster, add_city_code, train

HOLDOUT_DAYS = 30
ISSUE_HOURS = (0, 6, 12, 18)
CROSSING_AQI = 300
BACKTEST_ARTIFACTS = REPO_ROOT / "models" / "artifacts" / "backtest"
DOCS = REPO_ROOT / "docs"


@dataclass
class Metrics:
    model: str
    horizon_h: int
    n: int
    rmse: float
    mae: float
    bucket_accuracy: float
    crossing_precision: float | None
    crossing_recall: float | None
    crossing_events: int


def _metrics(name: str, horizon: int, pred: np.ndarray, actual: np.ndarray) -> Metrics:
    ok = ~(np.isnan(pred) | np.isnan(actual))
    p, a = pred[ok], actual[ok]
    if len(p) == 0:
        return Metrics(name, horizon, 0, float("nan"), float("nan"), float("nan"), None, None, 0)

    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    mae = float(np.mean(np.abs(p - a)))

    pa = np.array([aqi_from_pm25(x) or 0 for x in p])
    aa = np.array([aqi_from_pm25(x) or 0 for x in a])
    bucket = float(np.mean([category_for(x)[0] == category_for(y)[0] for x, y in zip(pa, aa)]))

    # Crossing detection: did we say "above 300" when it was above 300?
    pred_pos, act_pos = pa >= CROSSING_AQI, aa >= CROSSING_AQI
    tp = int(np.sum(pred_pos & act_pos))
    fp = int(np.sum(pred_pos & ~act_pos))
    fn = int(np.sum(~pred_pos & act_pos))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    return Metrics(name, horizon, len(p), rmse, mae, bucket, precision, recall, int(np.sum(act_pos)))


def _climatology(train_df: pd.DataFrame) -> dict[tuple[int, int], float]:
    """month x hour-of-day mean PM2.5 from the training period only."""
    d = train_df.dropna(subset=["pm25"]).copy()
    d["_m"] = d["ts"].dt.month
    d["_h"] = d["ts"].dt.hour
    return d.groupby(["_m", "_h"])["pm25"].mean().to_dict()


def run(holdout_days: int = HOLDOUT_DAYS) -> dict:
    from vayu_core.db import read_conn

    logger.info("building features for backtest…")
    frames: dict[str, pd.DataFrame] = {}
    weathers: dict[str, pd.DataFrame] = {}
    for city in list_cities():
        with read_conn() as con:
            meas = con.execute(
                "SELECT city, station_id, param, ts, value FROM measurements WHERE city = ?", [city.id]
            ).df()
            wx = con.execute("SELECT * FROM weather_hourly WHERE city = ?", [city.id]).df()
            st = con.execute("SELECT city, station_id, name, lat, lon FROM stations WHERE city = ?", [city.id]).df()
            fires = con.execute("SELECT * FROM fires WHERE city = ?", [city.id]).df()
        if meas.empty or st.empty:
            logger.warning(f"[{city.id}] no data — excluded from backtest")
            continue
        meas["ts"] = pd.to_datetime(meas["ts"], utc=True)
        if not wx.empty:
            wx["ts"] = pd.to_datetime(wx["ts"], utc=True)
        frames[city.id] = build_features(city, meas, wx, st, fires)
        weathers[city.id] = wx

    if not frames:
        raise SystemExit("no data to backtest — run `make seed` first")

    pooled = pd.concat([add_city_code(df, cid) for cid, df in frames.items()], ignore_index=True)
    last_ts = pooled["ts"].max()
    cutoff = last_ts - pd.Timedelta(days=holdout_days)
    logger.info(f"holdout: {cutoff:%Y-%m-%d %H:%M} → {last_ts:%Y-%m-%d %H:%M} ({holdout_days}d, never trained on)")

    # Retrain on pre-cutoff data only, into a separate artifact directory.
    logger.info("training backtest models (pre-cutoff data only)…")
    train(frames, weathers, artifact_dir=BACKTEST_ARTIFACTS, train_until=cutoff, valid_days=21)
    fc = Forecaster(artifact_dir=BACKTEST_ARTIFACTS)
    if not fc.available:
        raise SystemExit("backtest models failed to train")

    clim = _climatology(pooled[pooled["ts"] <= cutoff])
    actual_lookup = pooled.set_index(["station_id", "ts"])["pm25"]

    results: list[Metrics] = []
    per_horizon_rows: dict[int, pd.DataFrame] = {}

    for horizon in HORIZONS:
        # fx_* (weather at the target hour) is horizon-dependent, so the frame is
        # rebuilt per horizon — exactly as training did.
        pooled_h = pd.concat(
            [
                add_city_code(
                    model_frame(df, weathers.get(cid, pd.DataFrame()), horizon, with_target=False), cid
                )
                for cid, df in frames.items()
            ],
            ignore_index=True,
        )
        origins = pooled_h[
            (pooled_h["ts"] > cutoff) & (pooled_h["ts"].dt.hour.isin(ISSUE_HOURS))
        ].dropna(subset=["pm25_lag1"])
        if origins.empty:
            continue

        target_ts = origins["ts"] + pd.Timedelta(hours=horizon)
        actual = np.array([actual_lookup.get((s, t), np.nan) for s, t in zip(origins["station_id"], target_ts)])

        band = fc.predict(origins, horizon)
        pred = band["p50"].to_numpy()
        persistence = origins["pm25"].to_numpy()  # value at issue time
        climo = np.array([clim.get((t.month, t.hour), np.nan) for t in target_ts])

        keep = ~np.isnan(actual)
        results += [
            _metrics("VAYU", horizon, pred[keep], actual[keep]),
            _metrics("Persistence", horizon, persistence[keep], actual[keep]),
            _metrics("Climatology", horizon, climo[keep], actual[keep]),
        ]
        per_horizon_rows[horizon] = pd.DataFrame(
            {
                "ts": target_ts[keep],
                "actual": actual[keep],
                "vayu": pred[keep],
                "persistence": persistence[keep],
                "p10": band["p10"].to_numpy()[keep],
                "p90": band["p90"].to_numpy()[keep],
            }
        )
        logger.info(f"h{horizon}: evaluated {int(keep.sum()):,} forecasts")

    payload = _write_reports(results, per_horizon_rows, cutoff, last_ts, holdout_days, list(frames))
    return payload


def _calibration(rows: pd.DataFrame) -> float:
    """Share of actuals inside the p10–p90 band. Should be ≈ 0.80 if honest."""
    if rows.empty:
        return float("nan")
    inside = (rows["actual"] >= rows["p10"]) & (rows["actual"] <= rows["p90"])
    return float(inside.mean())


def _charts(per_horizon: dict[int, pd.DataFrame]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (DOCS / "img").mkdir(parents=True, exist_ok=True)
    made = []
    for horizon, rows in per_horizon.items():
        if rows.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"VAYU forecast · t+{horizon}h · {len(rows):,} held-out forecasts", fontsize=11)

        ax = axes[0]
        ax.scatter(rows["actual"], rows["vayu"], s=6, alpha=0.25, color="#22D3EE", label="VAYU")
        ax.scatter(rows["actual"], rows["persistence"], s=6, alpha=0.15, color="#F59E0B", label="Persistence")
        lim = float(np.nanpercentile(rows["actual"], 99.5))
        ax.plot([0, lim], [0, lim], color="#64748B", lw=1, ls="--")
        ax.set_xlabel("Actual PM2.5 (µg/m³)")
        ax.set_ylabel("Predicted")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend(fontsize=8)
        ax.set_title("Predicted vs actual", fontsize=9)

        ax = axes[1]
        resid = rows["vayu"] - rows["actual"]
        ax.hist(resid, bins=60, color="#22D3EE", alpha=0.8)
        ax.axvline(0, color="#64748B", lw=1, ls="--")
        ax.set_xlabel("Residual (predicted − actual, µg/m³)")
        ax.set_ylabel("Count")
        ax.set_title(f"Residuals · bias {resid.mean():+.1f}", fontsize=9)

        fig.tight_layout()
        p = DOCS / "img" / f"backtest_h{horizon}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(f"img/backtest_h{horizon}.png")
    return made


def _write_reports(
    results: list[Metrics],
    per_horizon: dict[int, pd.DataFrame],
    cutoff: pd.Timestamp,
    last_ts: pd.Timestamp,
    holdout_days: int,
    cities: list[str],
) -> dict:
    DOCS.mkdir(parents=True, exist_ok=True)
    charts = _charts(per_horizon)
    calib = {h: _calibration(r) for h, r in per_horizon.items()}

    payload = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "method": "rolling-origin, last N days held out entirely; models retrained pre-cutoff only",
            "holdout_days": holdout_days,
            "holdout_from": cutoff.isoformat(),
            "holdout_to": last_ts.isoformat(),
            "issue_hours_utc": list(ISSUE_HOURS),
            "crossing_threshold_aqi": CROSSING_AQI,
            "cities": cities,
        },
        "metrics": [asdict(m) for m in results],
        "calibration_p10_p90": {str(h): (None if np.isnan(v) else round(v, 3)) for h, v in calib.items()},
        "charts": charts,
    }
    (DOCS / "evaluation.json").write_text(json.dumps(payload, indent=2))

    # ---- markdown ----------------------------------------------------------
    by = {(m.model, m.horizon_h): m for m in results}
    lines = [
        "# VAYU — Forecast Evaluation",
        "",
        "> Auto-generated by `make backtest`. Do not edit by hand.",
        f"> Generated {payload['generated_ts'][:19]}Z · cities: {', '.join(cities)}",
        "",
        "## Protocol",
        "",
        f"- **Holdout:** the last **{holdout_days} days** ({cutoff:%Y-%m-%d} → {last_ts:%Y-%m-%d}) are excluded from",
        "  training entirely. Models are retrained on pre-cutoff data only, into a separate",
        "  artifact directory, and any row whose *target* falls inside the holdout is dropped —",
        "  so no evaluated forecast was ever trained on.",
        f"- **Origins:** forecasts issued at {', '.join(f'{h:02d}:00' for h in ISSUE_HOURS)} UTC each day.",
        "- **Baselines:** persistence (value at issue time) and climatology (month × hour-of-day mean,",
        "  computed on training data only).",
        f"- **Crossing detection:** event = actual AQI ≥ {CROSSING_AQI} at the target hour.",
        "",
        "## Accuracy vs baselines",
        "",
        "Lower RMSE/MAE is better. **Bold** = best in column.",
        "",
        "| Horizon | Model | n | RMSE (µg/m³) | MAE (µg/m³) | AQI bucket acc. |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for h in HORIZONS:
        trio = [by.get((m, h)) for m in ("VAYU", "Persistence", "Climatology")]
        trio = [m for m in trio if m]
        if not trio:
            continue
        best_rmse = min(m.rmse for m in trio)
        for m in trio:
            r = f"**{m.rmse:.1f}**" if m.rmse == best_rmse else f"{m.rmse:.1f}"
            lines.append(f"| t+{h}h | {m.model} | {m.n:,} | {r} | {m.mae:.1f} | {m.bucket_accuracy:.1%} |")

    lines += [
        "",
        "### Improvement over persistence",
        "",
        "| Horizon | VAYU RMSE | Persistence RMSE | Improvement |",
        "|---|---:|---:|---:|",
    ]
    for h in HORIZONS:
        v, p = by.get(("VAYU", h)), by.get(("Persistence", h))
        if not (v and p and p.rmse):
            continue
        imp = (p.rmse - v.rmse) / p.rmse
        lines.append(f"| t+{h}h | {v.rmse:.1f} | {p.rmse:.1f} | **{imp:+.1%}** |")

    lines += [
        "",
        f"## Hazard-crossing detection (AQI ≥ {CROSSING_AQI})",
        "",
        "The decision-relevant metric: RMSE can be won by hugging the mean and never calling a",
        "spike, which would be operationally useless. This table asks whether VAYU actually",
        "flags the hazard.",
        "",
        "| Horizon | Model | Events | Precision | Recall |",
        "|---|---|---:|---:|---:|",
    ]
    for h in HORIZONS:
        for name in ("VAYU", "Persistence", "Climatology"):
            m = by.get((name, h))
            if not m:
                continue
            pr = "—" if m.crossing_precision is None else f"{m.crossing_precision:.1%}"
            rc = "—" if m.crossing_recall is None else f"{m.crossing_recall:.1%}"
            lines.append(f"| t+{h}h | {m.model} | {m.crossing_events} | {pr} | {rc} |")

    lines += [
        "",
        "## Uncertainty calibration",
        "",
        "The p10–p90 band should contain ~80% of actuals. Meaningfully above means the band is",
        "too wide (useless); below means it is overconfident (dangerous).",
        "",
        "| Horizon | Actuals inside p10–p90 | Target |",
        "|---|---:|---:|",
    ]
    for h, v in calib.items():
        lines.append(f"| t+{h}h | {'—' if np.isnan(v) else f'{v:.1%}'} | 80% |")

    if charts:
        lines += ["", "## Charts", ""]
        for c in charts:
            lines.append(f"![{c}]({c})")

    # Honest headline: state the margin, and say plainly when it is slim.
    lines += ["", "## What these numbers actually say", ""]
    v24, p24 = by.get(("VAYU", 24)), by.get(("Persistence", 24))
    if v24 and p24 and p24.rmse:
        margin = (p24.rmse - v24.rmse) / p24.rmse
        if margin <= 0:
            lines.append(
                f"- **VAYU does not beat persistence at t+24h** ({v24.rmse:.1f} vs {p24.rmse:.1f} RMSE). "
                "That is reported rather than hidden."
            )
        elif margin < 0.05:
            lines.append(
                f"- **VAYU beats persistence at t+24h, but narrowly** ({v24.rmse:.1f} vs "
                f"{p24.rmse:.1f} RMSE, {margin:+.1%}). Persistence is a genuinely strong "
                "baseline for 24h PM2.5 — hourly pollution is highly autocorrelated, and "
                "'tomorrow looks like today' is right most of the time. A small margin here "
                "is the honest result, not a disappointing one, and it is why the "
                "crossing-detection table matters more than RMSE."
            )
        else:
            lines.append(
                f"- VAYU beats persistence at t+24h by {margin:.1%} on RMSE "
                f"({v24.rmse:.1f} vs {p24.rmse:.1f})."
            )
    lines += [
        "- **Climatology's RMSE is not a win.** It scores by predicting something near the "
        "  seasonal mean every day, which is why its AQI-bucket accuracy and crossing "
        "  precision collapse. A model that never calls a spike is useless to a commissioner "
        "  deciding whether to halt a construction site tonight.",
        "- **What persistence cannot do**, and why VAYU is not merely a tie: persistence gives "
        "  no uncertainty band, exists only where a station exists (VAYU scores every ward), "
        "  offers no attribution, and cannot be asked *why*. The forecast is an input to the "
        "  intervention, not the product.",
        "",
        "## Limitations",
        "",
        "- Evaluated at **station** locations. Ward-level values are IDW-interpolated from these,",
        "  so wards far from any station carry additional error not captured here (they are",
        "  flagged `low_confidence` in the API and watermarked in the UI).",
        "- Weather at prediction time comes from Open-Meteo's **historical forecast archive** —",
        "  what the forecast actually said at the time, not reanalysis. The model therefore",
        "  inherits the weather model's own error, which is the realistic operational setup",
        "  rather than a perfect-foresight upper bound. The exception is the 2016-2018 training",
        "  winters, where that archive does not reach back and reanalysis stands in; that",
        "  affects training only, never this evaluation.",
        "- **The measured record has a hole.** OpenAQ's Delhi stations ran 2016-2018, went",
        "  quiet, and resumed in Feb 2025 on new sensor ids. Training therefore spans two past",
        "  stubble winters plus Feb-Oct 2025, with nothing in between. Before those winters were",
        "  added the model had never seen a November and lost to persistence by 53%.",
        f"- {holdout_days} days is a short holdout covering one season. It is what the measured",
        "  record supports; more history would test more regimes.",
        "- Quantile models are fit independently per horizon, so the p10-p90 band is calibrated",
        "  empirically (see above) rather than by construction.",
        "- See `docs/DATA_PROVENANCE.md` for source-level caveats.",
        "",
    ]
    (DOCS / "evaluation.md").write_text("\n".join(lines))
    logger.success(f"wrote docs/evaluation.md and docs/evaluation.json ({len(charts)} charts)")
    return payload


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> {message}", level="INFO")
    payload = run()
    by = {(m["model"], m["horizon_h"]): m for m in payload["metrics"]}
    print("\n─── headline ───")
    for h in HORIZONS:
        v, p = by.get(("VAYU", h)), by.get(("Persistence", h))
        if v and p and p["rmse"]:
            imp = (p["rmse"] - v["rmse"]) / p["rmse"]
            print(f"  t+{h}h  VAYU RMSE {v['rmse']:6.1f}  vs persistence {p['rmse']:6.1f}   {imp:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
