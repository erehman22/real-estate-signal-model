"""
Process raw Flickr quarterly photo counts into model-ready features.

Mirrors the approach of 02_google_trends_process.py:
  - Normalise to a pre-boom baseline (2005–2014)
  - Rolling averages to smooth noise
  - YoY growth rate
  - Z-score relative to baseline
  - Structural break detection (Chow test proxy)

Additionally computes Flickr-specific features:
  - unique_user_growth: rate of change in photographer diversity
    (more meaningful than raw count — 100 photos from 10 people >
     100 photos from 1 person for discovery signal)
  - upload_lag_trend: whether photos are being posted faster over time
    (declining lag = place is entering mainstream awareness)
  - discovery_ratio: unique_users / photo_count
    (high ratio = many first-time visitors; low = repeat visitors/locals)

Input:  data/raw/flickr/{location_id}/photos_quarterly.csv
Output: data/processed/flickr/{location_id}_flickr_features.csv
        data/processed/flickr/_signal_summary.csv
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH   = Path("config/locations.yaml")
RAW_DIR       = Path("data/raw/flickr")
PROCESSED_DIR = Path("data/processed/flickr")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Baseline window for normalisation — pre-discovery period
BASELINE_START = 2005
BASELINE_END   = 2014

# Rolling window in quarters
ROLLING_WINDOW = 4   # 1 year of quarters

# Z-score threshold to flag as "signal firing"
Z_THRESHOLD = 2.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def load_raw(loc_id: str) -> pd.DataFrame | None:
    path = RAW_DIR / loc_id / "photos_quarterly.csv"
    if not path.exists():
        print(f"  ⚠️  No raw data found for {loc_id} — run 03_flickr.py first")
        return None
    df = pd.read_csv(path)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df = df.sort_values("period_start").reset_index(drop=True)
    df = df.set_index("period_start")
    return df


def add_rolling_features(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """Add rolling mean and rolling std for a column."""
    df[f"{col}_roll{window}q"] = df[col].rolling(window, min_periods=2).mean()
    df[f"{col}_roll{window}q_std"] = df[col].rolling(window, min_periods=2).std()
    return df


def add_yoy_growth(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """YoY growth rate: compare same quarter last year."""
    df[f"{col}_yoy"] = df[col].pct_change(periods=4) * 100  # 4 quarters = 1 year
    return df


def add_zscore(df: pd.DataFrame, col: str, baseline_start: int, baseline_end: int) -> pd.DataFrame:
    """
    Z-score normalised against the baseline period.
    Values > Z_THRESHOLD suggest signal above baseline noise.
    """
    baseline_mask = (
        (df.index.year >= baseline_start) &
        (df.index.year <= baseline_end)
    )
    baseline_vals = df.loc[baseline_mask, col].dropna()

    if len(baseline_vals) < 4:
        print(f"  ⚠️  Insufficient baseline data for z-score on {col}")
        df[f"{col}_zscore"] = np.nan
        return df

    mu    = baseline_vals.mean()
    sigma = baseline_vals.std()

    if sigma == 0:
        df[f"{col}_zscore"] = 0.0
    else:
        df[f"{col}_zscore"] = (df[col] - mu) / sigma

    return df


def detect_structural_break(df: pd.DataFrame, col: str) -> int | None:
    """
    Chow test proxy: find the year where splitting the series
    minimises residual sum of squares. Returns candidate break year.

    Requires at least 8 quarters on each side of the candidate break.
    """
    series = df[col].dropna()
    if len(series) < 16:
        return None

    min_rss   = np.inf
    break_idx = None

    for i in range(8, len(series) - 8):
        before = series.iloc[:i].values
        after  = series.iloc[i:].values

        rss_before = np.sum((before - before.mean()) ** 2)
        rss_after  = np.sum((after  - after.mean())  ** 2)
        rss_total  = rss_before + rss_after

        if rss_total < min_rss:
            min_rss   = rss_total
            break_idx = i

    if break_idx is not None:
        break_date = series.index[break_idx]
        return break_date.year
    return None


def compute_lead_time(signal_year: int | None, inflection_year: int) -> float | None:
    """Lead time in years: positive = signal precedes price inflection."""
    if signal_year is None:
        return None
    return inflection_year - signal_year


# ── Processing ────────────────────────────────────────────────────────────────

def process_location(loc: dict) -> dict | None:
    """
    Process a single location. Returns summary dict for cross-location table.
    """
    loc_id         = loc["id"]
    loc_name       = loc["name"]
    inflection_year = loc["price_inflection_year"]

    print(f"\n📍  {loc_name}")

    df = load_raw(loc_id)
    if df is None:
        return None

    # ── Core photo count features ──────────────────────────────────────────

    df = add_rolling_features(df, "photo_count", ROLLING_WINDOW)
    df = add_yoy_growth(df, "photo_count")
    df = add_zscore(df, "photo_count", BASELINE_START, BASELINE_END)
    df = add_zscore(df, f"photo_count_roll{ROLLING_WINDOW}q", BASELINE_START, BASELINE_END)

    # ── Unique user features ───────────────────────────────────────────────

    df = add_rolling_features(df, "unique_users", ROLLING_WINDOW)
    df = add_yoy_growth(df, "unique_users")
    df = add_zscore(df, "unique_users", BASELINE_START, BASELINE_END)

    # ── Discovery ratio: unique_users / photo_count ────────────────────────
    # High = many new first-time visitors (early discovery)
    # Low  = repeat visitors or locals dominating

    df["discovery_ratio"] = np.where(
        df["photo_count"] > 0,
        df["unique_users"] / df["photo_count"],
        np.nan,
    )
    df = add_rolling_features(df, "discovery_ratio", ROLLING_WINDOW)

    # ── Upload lag trend ───────────────────────────────────────────────────
    # Declining lag = place entering mainstream awareness
    # (people post faster when somewhere is "hot")

    if "avg_upload_lag_days" in df.columns:
        df = add_rolling_features(df, "avg_upload_lag_days", ROLLING_WINDOW)

    # ── Structural break detection ─────────────────────────────────────────

    break_year_count = detect_structural_break(df, "photo_count")
    break_year_users = detect_structural_break(df, "unique_users")

    # Use rolling-smoothed count for break detection if raw is noisy
    break_year_smooth = detect_structural_break(
        df, f"photo_count_roll{ROLLING_WINDOW}q"
    )

    # Best break year = earliest of the three signals
    candidate_breaks = [
        y for y in [break_year_count, break_year_users, break_year_smooth]
        if y is not None
    ]
    best_break_year = min(candidate_breaks) if candidate_breaks else None

    # ── Z-score signal first-fire ──────────────────────────────────────────
    # First quarter where smoothed z-score exceeds threshold

    z_col     = f"photo_count_roll{ROLLING_WINDOW}q_zscore"
    above     = df[df[z_col] >= Z_THRESHOLD] if z_col in df.columns else pd.DataFrame()
    z_fire_year = above.index[0].year if not above.empty else None

    # ── Lead time calculations ─────────────────────────────────────────────

    lead_break = compute_lead_time(best_break_year, inflection_year)
    lead_z     = compute_lead_time(z_fire_year, inflection_year)

    # ── Save processed output ──────────────────────────────────────────────

    out_path = PROCESSED_DIR / f"{loc_id}_flickr_features.csv"
    df.to_csv(out_path)
    print(f"  ✓  Saved {len(df)} quarters of features → {out_path}")
    print(f"     Structural break: {best_break_year}  |  Z-threshold fire: {z_fire_year}")
    print(f"     Lead time (break): {lead_break} yrs  |  Lead time (z): {lead_z} yrs")

    return {
        "location_id":           loc_id,
        "location_name":         loc_name,
        "price_inflection_year": inflection_year,
        "break_year_count":      break_year_count,
        "break_year_users":      break_year_users,
        "break_year_smooth":     break_year_smooth,
        "best_break_year":       best_break_year,
        "z_fire_year":           z_fire_year,
        "lead_time_break_yrs":   lead_break,
        "lead_time_z_yrs":       lead_z,
        "total_photos_all_time": int(df["photo_count"].sum()),
        "peak_quarter":          df["photo_count"].idxmax().strftime("%Y-Q") +
                                 str((df["photo_count"].idxmax().month - 1) // 3 + 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    locations = load_locations(CONFIG_PATH)
    summary   = []

    print("📊  Processing Flickr data\n")

    for loc in locations:
        result = process_location(loc)
        if result:
            summary.append(result)

    if summary:
        summary_df   = pd.DataFrame(summary)
        summary_path = PROCESSED_DIR / "_signal_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n\n── Cross-location summary ──────────────────────────────────────")
        print(summary_df[[
            "location_name",
            "price_inflection_year",
            "best_break_year",
            "z_fire_year",
            "lead_time_break_yrs",
            "lead_time_z_yrs",
            "total_photos_all_time",
        ]].to_string(index=False))
        print(f"\n  ✓  Summary saved → {summary_path}")

    print("\n✅  Flickr processing complete.")


if __name__ == "__main__":
    main()