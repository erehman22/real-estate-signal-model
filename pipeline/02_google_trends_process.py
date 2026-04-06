"""
Process and normalise raw Google Trends data.
Applies geo-aware processing, percentile rank (replacing Z-score),
and composite signal scoring.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("config/locations.yaml")
RAW_DIR     = Path("data/raw/google_trends")
OUTPUT_DIR  = Path("data/processed/google_trends")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_END = "2015-12-31"
GEO_FILTERS  = ["WW", "GB", "DE", "US"]

QUERY_TYPES  = [
    "awareness", "travel", "property", "real_estate",
    "buy", "nomad", "expat", "move_to",
]

# Weights for composite score — higher = more specific signal
QUERY_WEIGHTS = {
    "awareness":   0.10,
    "travel":      0.20,
    "property":    0.20,
    "real_estate": 0.15,
    "buy":         0.10,
    "nomad":       0.10,
    "expat":       0.10,
    "move_to":     0.05,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def load_raw(loc_id: str, geo: str, label: str) -> Optional[pd.DataFrame]:
    path = RAW_DIR / loc_id / geo / f"{label}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df.resample("MS").mean()
    return df


def compute_features(df: pd.DataFrame, inflection_year: int) -> pd.DataFrame:
    out = df.copy()
    out.columns = ["raw"]

    # Smooth
    out["rolling_12m"] = out["raw"].rolling(12, min_periods=6).mean()

    # Rate of change
    out["yoy_change"] = out["raw"].diff(12)
    out["yoy_pct"]    = out["raw"].pct_change(12) * 100

    # Percentile rank over trailing 36-month window — robust to outliers
    out["pct_rank"] = (
        out["raw"]
        .rolling(36, min_periods=12)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # Z-score vs baseline (kept for reference, capped at ±8)
    baseline = out.loc[:BASELINE_END, "raw"]
    if len(baseline) >= 12:
        mu  = baseline.mean()
        sig = baseline.std()
        raw_z = (out["raw"] - mu) / (sig if sig > 0 else 1)
        out["zscore"] = raw_z.clip(-8, 8)
    else:
        out["zscore"] = np.nan

    out["years_before_inflection"] = inflection_year - out.index.year

    return out


def compute_composite(
    all_features: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Weighted composite of percentile ranks across query types.
    Result is a single series from 0-1 representing overall signal strength.
    """
    weighted_sum  = None
    total_weight  = 0.0

    for label, df in all_features.items():
        w = weights.get(label, 0.0)
        if "pct_rank" not in df.columns:
            continue
        series = df["pct_rank"] * w
        if weighted_sum is None:
            weighted_sum = series
        else:
            weighted_sum = weighted_sum.add(series, fill_value=0)
        total_weight += w

    if weighted_sum is None or total_weight == 0:
        return pd.DataFrame()

    composite = (weighted_sum / total_weight).to_frame("composite_score")
    composite["composite_rolling"] = composite["composite_score"].rolling(6, min_periods=3).mean()
    return composite


def detect_structural_break(df: pd.DataFrame, col: str = "rolling_12m") -> Optional[int]:
    series = df[col].dropna()
    if len(series) < 36:
        return None

    best_year = None
    best_t    = 0

    for year in series.index.year.unique()[3:-3]:
        before = series[series.index.year.isin(range(year - 3, year))]
        after  = series[series.index.year.isin(range(year, year + 3))]
        if len(before) < 6 or len(after) < 6:
            continue
        t, p = stats.ttest_ind(after, before, alternative="greater")
        if t > best_t:
            best_t    = t
            best_year = year

    return best_year


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    locations    = load_locations(CONFIG_PATH)
    summary_rows = []

    for loc in locations:
        loc_id          = loc["id"]
        inflection_year = loc["price_inflection_year"]

        print(f"\n📍 Processing: {loc['name']}")

        for geo in GEO_FILTERS:
            print(f"  🌍 Geo: {geo}")
            geo_features = {}

            for label in QUERY_TYPES:
                raw = load_raw(loc_id, geo, label)
                if raw is None:
                    continue

                features = compute_features(raw, inflection_year)
                geo_features[label] = features

                out_path = OUTPUT_DIR / f"{loc_id}_{geo}_{label}.csv"
                features.to_csv(out_path)

            if not geo_features:
                continue

            # Composite signal
            composite = compute_composite(geo_features, QUERY_WEIGHTS)
            if not composite.empty:
                comp_path = OUTPUT_DIR / f"{loc_id}_{geo}_composite.csv"
                composite.to_csv(comp_path)

                break_year = detect_structural_break(
                    composite.rename(columns={"composite_rolling": "rolling_12m"}),
                    col="rolling_12m",
                )

                if break_year is not None:
                    lead_time = inflection_year - break_year
                    print(f"    Break detected: {break_year} → lead time {lead_time}y")
                    summary_rows.append({
                        "location_id":     loc_id,
                        "location_name":   loc["name"],
                        "geo":             geo,
                        "inflection_year": inflection_year,
                        "break_year":      break_year,
                        "lead_time_years": lead_time,
                    })

    # Cross-location summary
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(OUTPUT_DIR / "_signal_summary.csv", index=False)
        print(f"\n✅ Summary:\n")
        print(df.pivot_table(
            index="location_name",
            columns="geo",
            values="lead_time_years",
            aggfunc="first",
        ).to_string())


if __name__ == "__main__":
    main()