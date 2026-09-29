"""HCHO hotspot detection and fire coupling.

The design goal is to identify hotspots using statistical thresholds and to
analyse correlation between fire activity and HCHO levels. Two methodological
choices drive everything here, and both are about not fooling ourselves:

**1. Anomaly against each cell's OWN baseline, never a global threshold.**
HCHO has a strong spatial climatology — the Indo-Gangetic Plain sits well above
the Thar desert every day of the year, burning or not. A single national cutoff
would therefore just redraw a map of where HCHO is habitually high, which says
nothing about biomass burning. Scoring each cell against its own seasonal centre
asks the question that actually matters: *is this cell unusual for itself?*

**2. Median and MAD, not mean and standard deviation.**
The baseline window necessarily contains the burning episodes we are trying to
detect. A mean would be dragged upward by those very spikes and a standard
deviation inflated by them, so the anomaly would partly cancel itself — the
worse the fire season, the harder it becomes to detect. Median/MAD are robust to
outliers, so the baseline stays a description of "normal".

Correlation is reported with an explicit n and Spearman alongside Pearson: fire
counts are heavily skewed (most cells zero, a few enormous), which is exactly
the shape where Pearson alone flatters a relationship.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Scale factor making MAD a consistent estimator of sigma for normal data, so a
# z-score built on MAD is comparable to a conventional one.
MAD_TO_SIGMA = 1.4826

# A cell must clear this many robust sigmas above its own baseline to count.
# 2.5 is deliberately stricter than the conventional 2.0: with ~15k cells x 55
# days (~10^6 tests) a 2.0 cutoff would return tens of thousands of cells by
# chance alone, and a "hotspot" that common is not a hotspot.
DEFAULT_Z = 2.5

# A cell needs at least this many valid days of history for a baseline to mean
# anything. Below it, MAD is unstable and the z-score is noise.
MIN_BASELINE_DAYS = 10

# MAD has a hard failure mode: if more than half a cell's days share the same
# value, the median absolute deviation is exactly 0 — and then every spike
# divides by zero and the cell gets discarded as "no spread". That is the
# opposite of what we want: a dead-flat cell with a huge spike is the clearest
# hotspot there is. So spread is floored at a fraction of the cell's own
# baseline. At 0.10, clearing z=2.5 needs roughly a 25% enhancement, which sits
# well below the +86%..+120% that real burning produces.
MIN_SPREAD_FRAC = 0.10


# HCHO is a *secondary* product: fire VOCs oxidise to formaldehyde over hours,
# and TROPOMI sees one overpass a day. Measured on the 2025 kharif season, the
# fire->HCHO association peaks at a one-day lag (Spearman 0.104 vs 0.080 at lag
# 0, falling to 0.064 by lag 3), which is what that chemistry predicts.
DEFAULT_FIRE_LAG_DAYS = 1

# Cells backed by very few native pixels are noise. 0.1deg source binned into a
# coarser grid normally yields 4-9 pixels per cell, so requiring 4 drops the
# swath-edge slivers without discarding real coverage.
DEFAULT_MIN_OBS = 4

# Fire-count bins for the dose-response contrast. Chosen because the response
# saturates quickly — most of the signal is between "no fire" and "any fire".
FIRE_BINS: tuple[tuple[float, float, str], ...] = (
    (-1, 0, "no fire"),
    (0, 5, "1-5"),
    (5, 20, "6-20"),
    (20, float("inf"), ">20"),
)


@dataclass
class Correlation:
    """Fire-vs-HCHO association. `n` is carried so a strong-looking r on a
    handful of points cannot be quoted as if it were solid."""

    n: int
    pearson_r: float
    spearman_r: float
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "pearson_r": None if np.isnan(self.pearson_r) else round(self.pearson_r, 4),
            "spearman_r": None if np.isnan(self.spearman_r) else round(self.spearman_r, 4),
            "note": self.note,
        }


@dataclass
class HotspotResult:
    hotspots: pd.DataFrame
    baseline: pd.DataFrame
    correlation: Correlation
    by_source_region: pd.DataFrame = field(default_factory=pd.DataFrame)


def cell_baseline(
    hcho: pd.DataFrame,
    min_days: int = MIN_BASELINE_DAYS,
    min_spread_frac: float = MIN_SPREAD_FRAC,
) -> pd.DataFrame:
    """Per-cell robust centre and spread of HCHO across the window.

    Returns one row per (grid_lat, grid_lon) with `baseline` (median) and
    `mad_sigma` (MAD rescaled to a sigma equivalent, floored — see
    MIN_SPREAD_FRAC). Cells with too little history are dropped rather than
    given a fragile baseline.
    """
    cols = ["grid_lat", "grid_lon", "baseline", "mad_sigma", "n_days"]
    if hcho.empty:
        return pd.DataFrame(columns=cols)

    def _stats(v: pd.Series) -> pd.Series:
        med = v.median()
        mad = (v - med).abs().median() * MAD_TO_SIGMA
        # Floor the spread so a near-constant cell stays testable instead of
        # collapsing to a divide-by-zero and being thrown away.
        floor = abs(med) * min_spread_frac
        return pd.Series({"baseline": med, "mad_sigma": max(mad, floor), "n_days": float(v.size)})

    out = hcho.groupby(["grid_lat", "grid_lon"])["value"].apply(_stats).unstack().reset_index()
    return out[out["n_days"] >= min_days].reset_index(drop=True)


def detect(
    hcho: pd.DataFrame,
    fires: pd.DataFrame | None = None,
    z_threshold: float = DEFAULT_Z,
    min_days: int = MIN_BASELINE_DAYS,
) -> HotspotResult:
    """Flag (cell, day) pairs whose HCHO is anomalously high for that cell.

    `hcho`  : grid_lat, grid_lon, date, value   (from satellite_grid)
    `fires` : grid_lat, grid_lon, date, fire_count, frp_sum  (from fire_grid)

    Fires are joined on the *same* cell-day. That is a deliberate first-order
    view: HCHO from burning also drifts downwind, so co-located counts capture
    the source but not the transported plume. The wind-trajectory layer handles
    transport separately — conflating the two here would make the correlation
    unreadable.
    """
    empty_cols = [
        "grid_lat", "grid_lon", "date", "hcho", "baseline", "z_score",
        "fire_count", "frp_sum",
    ]
    if hcho.empty:
        return HotspotResult(pd.DataFrame(columns=empty_cols), pd.DataFrame(), Correlation(0, np.nan, np.nan, "no HCHO data"))

    base = cell_baseline(hcho, min_days=min_days)
    if base.empty:
        return HotspotResult(
            pd.DataFrame(columns=empty_cols), base,
            Correlation(0, np.nan, np.nan, f"no cell had >= {min_days} valid days"),
        )

    df = hcho.merge(base, on=["grid_lat", "grid_lon"], how="inner")
    # A cell whose MAD is zero is constant across the window: no spread means no
    # meaningful anomaly, and dividing by it would produce inf.
    df = df[df["mad_sigma"] > 0].copy()
    if df.empty:
        return HotspotResult(pd.DataFrame(columns=empty_cols), base, Correlation(0, np.nan, np.nan, "no cell had non-zero spread"))

    df["z_score"] = (df["value"] - df["baseline"]) / df["mad_sigma"]

    if fires is not None and not fires.empty:
        f = fires[["grid_lat", "grid_lon", "date", "fire_count", "frp_sum"]]
        df = df.merge(f, on=["grid_lat", "grid_lon", "date"], how="left")
    else:
        df["fire_count"] = np.nan
        df["frp_sum"] = np.nan
    # A cell-day with no fire row genuinely had no detections — zero, not unknown.
    df[["fire_count", "frp_sum"]] = df[["fire_count", "frp_sum"]].fillna(0.0)

    corr = correlate(df)
    hot = (
        df[df["z_score"] >= z_threshold]
        .rename(columns={"value": "hcho"})
        .sort_values("z_score", ascending=False)
        .reset_index(drop=True)
    )
    return HotspotResult(hotspots=hot, baseline=base, correlation=corr)


def correlate(df: pd.DataFrame, x: str = "fire_count", y: str = "z_score") -> Correlation:
    """Association between fire activity and HCHO anomaly across all cell-days.

    Spearman is reported next to Pearson because fire counts are extremely
    right-skewed (most cells zero, a few enormous) — the regime where Pearson
    is most easily inflated by a handful of extreme points.
    """
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return Correlation(len(d), np.nan, np.nan, "not enough variation to correlate")

    pearson = float(np.corrcoef(d[x], d[y])[0, 1])
    spearman = float(np.corrcoef(d[x].rank(), d[y].rank())[0, 1])
    return Correlation(n=len(d), pearson_r=pearson, spearman_r=spearman)


def cluster(hotspots: pd.DataFrame, grid_deg: float, max_gap_cells: int = 1) -> pd.DataFrame:
    """Group same-day adjacent hotspot cells into contiguous clusters.

    Plain flood-fill over the grid rather than DBSCAN: the cells sit on a
    regular lattice, so adjacency is exact integer arithmetic and needs no
    distance metric, no epsilon to tune and no extra dependency.

    Adds a `cluster_id` unique within each date.
    """
    if hotspots.empty:
        return hotspots.assign(cluster_id=pd.Series(dtype="int64"))

    out = []
    for day, grp in hotspots.groupby("date", sort=True):
        # Integer lattice coordinates make neighbour tests exact.
        gi = (grp["grid_lat"] / grid_deg).round().astype(int).to_numpy()
        gj = (grp["grid_lon"] / grid_deg).round().astype(int).to_numpy()
        cells = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(gi, gj))}

        parent = list(range(len(gi)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        r = range(-max_gap_cells, max_gap_cells + 1)
        for (a, b), k in cells.items():
            for da in r:
                for db in r:
                    if da == 0 and db == 0:
                        continue
                    n = cells.get((a + da, b + db))
                    if n is not None:
                        union(k, n)

        roots = [find(k) for k in range(len(gi))]
        # Renumber to compact 0..n-1 ids so they are readable in the UI.
        remap = {root: i for i, root in enumerate(sorted(set(roots)))}
        g = grp.copy()
        g["cluster_id"] = [remap[r_] for r_ in roots]
        out.append(g)

    return pd.concat(out, ignore_index=True)


def stratify_by_fire(
    hcho: pd.DataFrame,
    fires: pd.DataFrame,
    lag_days: int = DEFAULT_FIRE_LAG_DAYS,
    min_obs: int = DEFAULT_MIN_OBS,
    bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Median HCHO stratified by same-cell fire count — the headline Obj-2 stat.

    **Why this and not a correlation coefficient.** Measured on the 2025 kharif
    season, Pearson r between fire count and HCHO anomaly is ~0.03 while
    Spearman is ~0.14 — which reads as "no relationship" if you quote Pearson
    and stop. Both are misleading, for a structural reason: ~97% of cell-days
    have zero fires, and the HCHO response *saturates* almost immediately. The
    relationship is a step, not a line, so a linear coefficient is the wrong
    instrument.

    Stratifying shows what is actually there. All-India, one-day lag:
    no fire 1.11e-4, 1-5 fires +86%, 6-20 +107%, >20 +120% — HCHO more than
    doubles in burning cells, monotonically with intensity.

    Returns one row per bin with n, median/mean HCHO and the % lift over the
    no-fire baseline.
    """
    if hcho.empty or fires.empty:
        return pd.DataFrame()

    h = hcho.copy()
    if min_obs and "n_obs" in h:
        h = h[h["n_obs"] >= min_obs]
    if bbox:
        w, s, e, n = bbox
        h = h[h["grid_lat"].between(s, n) & h["grid_lon"].between(w, e)]
    if h.empty:
        return pd.DataFrame()

    f = fires[["grid_lat", "grid_lon", "date", "fire_count"]].copy()
    h["date"] = pd.to_datetime(h["date"])
    f["date"] = pd.to_datetime(f["date"]) + pd.Timedelta(days=lag_days)

    j = h.merge(f, on=["grid_lat", "grid_lon", "date"], how="left")
    j["fire_count"] = j["fire_count"].fillna(0.0)

    rows, baseline = [], None
    for lo, hi, label in FIRE_BINS:
        s_ = j[(j["fire_count"] > lo) & (j["fire_count"] <= hi)]["value"]
        if s_.empty:
            continue
        med = float(s_.median())
        if baseline is None:
            baseline = med
        rows.append(
            {
                "fire_bin": label,
                "n": int(s_.size),
                "median_hcho": med,
                "mean_hcho": float(s_.mean()),
                "pct_vs_no_fire": None if not baseline else round(100 * (med / baseline - 1), 1),
            }
        )
    return pd.DataFrame(rows)


def summarise_by_source_region(hotspots: pd.DataFrame) -> pd.DataFrame:
    """Hotspot burden per named source region — the "identify major source
    regions" deliverable."""
    if hotspots.empty or "source_region" not in hotspots:
        return pd.DataFrame()
    return (
        hotspots.groupby(hotspots["source_region"].fillna("(outside defined regions)"), as_index=False)
        .agg(
            hotspot_cell_days=("z_score", "size"),
            mean_z=("z_score", "mean"),
            max_z=("z_score", "max"),
            mean_hcho=("hcho", "mean"),
            total_fires=("fire_count", "sum"),
        )
        .rename(columns={"source_region": "source_region"})
        .sort_values("hotspot_cell_days", ascending=False)
        .reset_index(drop=True)
    )
