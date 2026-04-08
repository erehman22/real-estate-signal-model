"""
Process raw Reddit mention counts into signal features.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH  = Path("config/locations.yaml")
RAW_DIR      = Path("data/raw/reddit")
OUTPUT_DIR   = Path("data/processed/reddit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_END = "2015-12-31"

SUBREDDITS   = [
    "travel", "solotravel", "digitalnomad",
    "expats", "personalfinance", "realestateinvesting", "liveabroad",
]

# Weights for composite — nomad/expat subreddits matter more for our signal
SUBREDDIT_WEIGHTS = {
    "travel":               0.15,
    "solotravel":           0.15,
    "digitalnomad":         0.25,
    "expats":               0.20,
    "liveabroad":           0.10,
    "personalfinance":      0.05,
    "realestateinvesting":  0.10,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def load_raw(loc_id: str, subreddit: str) -> pd.DataFrame | None:
    path = RAW_DIR / loc_id / f"{subreddit}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df = df.resample("MS").sum()
    return df[["count"]]


def compute_features(df: pd.DataFrame, inflection_year: int) -> pd.DataFrame:
    out = df.rename(columns={"count": "raw"})

    out["rolling_3m"]  = out["raw"].rolling(3,  min_periods=1).mean()
    out["rolling_12m"] = out["raw"].rolling(12, min_periods=6).mean()

    out["yoy_change"]  = out["raw"].diff(12)
    out["yoy_pct"]     = out["raw"].pct_change(12) * 100

    # Percentile rank — robust for sparse subreddits
    out["pct_rank"] = (
        out["raw"]
        .rolling(36, min_periods=6)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # Z-score vs baseline, capped
    baseline = out.loc[:BASELINE_END, "raw"]
    if len(baseline) >= 6 and baseline.std() > 0:
        mu  = baseline.mean()
        sig = baseline.std()
        out["zscore"] = ((out["raw"] - mu) / sig).clip(-5, 8)
    else:
        out["zscore"] = np.nan

    out["years_before_inflection"] = inflection_year - out.index.year

    return out


def compute_composite(
    all_features: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    weighted_sum = None
    total_weight = 0.0

    for subreddit, df in all_features.items():
        w = weights.get(subreddit, 0.0)
        if "pct_rank" not in df.columns:
            continue
        series = df["pct_rank"].fillna(0) * w
        weighted_sum  = series if weighted_sum is None else weighted_sum.add(series, fill_value=0)
        total_weight += w

    if weighted_sum is None or total_weight == 0:
        return pd.DataFrame()

    comp = (weighted_sum / total_weight).to_frame("composite_score")
    comp["composite_rolling"] = comp["composite_score"].rolling(6, min_periods=3).mean()
    return comp


def detect_structural_break(series: pd.Series) -> int | None:
    series = series.dropna()
    if len(series) < 36:
        return None
    best_year, best_t = None, 0
    for year in series.index.year.unique()[3:-3]:
        before = series[series.index.year.isin(range(year - 3, year))]
        after  = series[series.index.year.isin(range(year, year + 3))]
        if len(before) < 6 or len(after) < 6:
            continue
        t, _ = stats.ttest_ind(after, before, alternative="greater")
        if t > best_t:
            best_t, best_year = t, year
    return best_year


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    locations    = load_locations(CONFIG_PATH)
    summary_rows = []

    for loc in locations:
        loc_id          = loc["id"]
        inflection_year = loc["price_inflection_year"]

        print(f"\n📍 Processing: {loc['name']}")
        loc_features = {}

        for subreddit in SUBREDDITS:
            raw = load_raw(loc_id, subreddit)
            if raw is None:
                continue

            features = compute_features(raw, inflection_year)
            loc_features[subreddit] = features

            out_path = OUTPUT_DIR / f"{loc_id}_{subreddit}.csv"
            features.to_csv(out_path)

        if not loc_features:
            print("  ⚠️  No data found")
            continue

        # Combined raw mention count across all subreddits
        combined_raw = pd.concat(
            [df["raw"].rename(sr) for sr, df in loc_features.items()], axis=1
        )
        combined_raw["total"] = combined_raw.sum(axis=1)
        combined_raw.to_csv(OUTPUT_DIR / f"{loc_id}_combined_raw.csv")

        # Composite signal
        composite = compute_composite(loc_features, SUBREDDIT_WEIGHTS)
        if not composite.empty:
            composite.to_csv(OUTPUT_DIR / f"{loc_id}_composite.csv")

            break_year = detect_structural_break(composite["composite_rolling"])
            if break_year:
                lead_time = inflection_year - break_year
                print(f"  Break detected: {break_year} → lead time {lead_time}y")
                summary_rows.append({
                    "location_id":     loc_id,
                    "location_name":   loc["name"],
                    "inflection_year": inflection_year,
                    "break_year":      break_year,
                    "lead_time_years": lead_time,
                })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(OUTPUT_DIR / "_signal_summary.csv", index=False)
        print(f"\n✅ Reddit signal summary:\n")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()