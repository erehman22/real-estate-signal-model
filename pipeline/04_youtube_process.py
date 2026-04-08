"""
Process raw YouTube quarterly data into model-ready features.

Mirrors the approach of 02_google_trends_process.py and 04_flickr_process.py:
  - Normalise to pre-boom baseline
  - Rolling averages
  - YoY growth rate
  - Z-score relative to baseline
  - Structural break detection

YouTube-specific features of note:

  weighted_reach (upload_count × median_channel_subs):
    This is the headline signal. A place being covered by mid-sized travel
    channels (50k–500k subs) is different from micro-vloggers or mega-channels.
    Mid-sized channels are the "early legitimisers" — they have reach but are
    still hunting for undiscovered content.

  view_velocity:
    Views accumulated on early video cohorts, measured now. A 2015 video about
    Tbilisi that has 800k views today reveals that content found a large audience
    retroactively — the place grew into its content. Compare this to a 2015 video
    about Kotor: if it has 50k views, that cohort never broke through.

  pct_large_channels:
    When mainstream travel channels (>100k subs) start covering a place, that's
    a T2 legitimisation signal. Tracking when this % first spikes is analogous
    to tracking the first prestige media mention.

Input:  data/raw/youtube/{location_id}/quarterly.csv
Output: data/processed/youtube/{location_id}_youtube_features.csv
        data/processed/youtube/_signal_summary.csv
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH   = Path("config/locations.yaml")
RAW_DIR       = Path("data/raw/youtube")
PROCESSED_DIR = Path("data/processed/youtube")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_START = 2006
BASELINE_END   = 2014
ROLLING_WINDOW = 4       # quarters
Z_THRESHOLD    = 2.0

# ── Helpers (same pattern as GT and Flickr processing scripts) ────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def load_raw(loc_id: str) -> pd.DataFrame | None:
    path = RAW_DIR / loc_id / "quarterly.csv"
    if not path.exists():
        print(f"  ⚠️  No raw data for {loc_id} — run 03_youtube.py first")
        return None
    df = pd.read_csv(path)
    df = df.sort_values(["published_year", "published_quarter"]).reset_index(drop=True)

    # Build a proper datetime index (first day of each quarter)
    df["period_start"] = pd.to_datetime(
        df["published_year"].astype(str) + "-" +
        ((df["published_quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01"
    )
    df = df.set_index("period_start")
    return df


def add_rolling(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[f"{col}_roll{window}q"] = df[col].rolling(window, min_periods=2).mean()
    return df


def add_yoy(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[f"{col}_yoy"] = df[col].pct_change(periods=4) * 100
    return df


def add_zscore(df: pd.DataFrame, col: str, start: int, end: int) -> pd.DataFrame:
    if col not in df.columns:
        return df
    mask     = (df.index.year >= start) & (df.index.year <= end)
    baseline = df.loc[mask, col].dropna()
    if len(baseline) < 4:
        df[f"{col}_zscore"] = np.nan
        return df
    mu    = baseline.mean()
    sigma = baseline.std()
    df[f"{col}_zscore"] = 0.0 if sigma == 0 else (df[col] - mu) / sigma
    return df


def detect_break(df: pd.DataFrame, col: str) -> int | None:
    if col not in df.columns:
        return None
    series = df[col].dropna()
    if len(series) < 16:
        return None
    min_rss = np.inf
    break_idx = None
    for i in range(8, len(series) - 8):
        before = series.iloc[:i].values
        after  = series.iloc[i:].values
        rss    = np.sum((before - before.mean()) ** 2) + np.sum((after - after.mean()) ** 2)
        if rss < min_rss:
            min_rss   = rss
            break_idx = i
    if break_idx is not None:
        return series.index[break_idx].year
    return None


def first_z_fire(df: pd.DataFrame, col: str, threshold: float) -> int | None:
    z_col = f"{col}_zscore"
    if z_col not in df.columns:
        return None
    above = df[df[z_col] >= threshold]
    return above.index[0].year if not above.empty else None


def lead_time(signal_year: int | None, inflection_year: int) -> float | None:
    if signal_year is None:
        return None
    return inflection_year - signal_year


# ── Processing ────────────────────────────────────────────────────────────────

def process_location(loc: dict) -> dict | None:
    loc_id          = loc["id"]
    loc_name        = loc["name"]
    inflection_year = loc["price_inflection_year"]

    print(f"\n📍  {loc_name}")

    df = load_raw(loc_id)
    if df is None:
        return None

    # ── upload_count features ──────────────────────────────────────────────
    df = add_rolling(df, "upload_count", ROLLING_WINDOW)
    df = add_yoy(df, "upload_count")
    df = add_zscore(df, "upload_count", BASELINE_START, BASELINE_END)
    df = add_zscore(df, f"upload_count_roll{ROLLING_WINDOW}q", BASELINE_START, BASELINE_END)

    # ── weighted_reach features ────────────────────────────────────────────
    # This is the primary signal: volume × typical audience size
    df = add_rolling(df, "weighted_reach", ROLLING_WINDOW)
    df = add_yoy(df, "weighted_reach")
    df = add_zscore(df, "weighted_reach", BASELINE_START, BASELINE_END)
    df = add_zscore(df, f"weighted_reach_roll{ROLLING_WINDOW}q", BASELINE_START, BASELINE_END)

    # ── total_views features ───────────────────────────────────────────────
    # Cumulative views on each quarter's cohort — reveals view velocity
    df = add_rolling(df, "total_views", ROLLING_WINDOW)
    df = add_zscore(df, "total_views", BASELINE_START, BASELINE_END)

    # ── pct_large_channels ────────────────────────────────────────────────
    # When mainstream channels arrive — T2 legitimisation proxy
    df = add_rolling(df, "pct_large_channels", ROLLING_WINDOW)

    # ── Structural break detection ─────────────────────────────────────────
    break_upload  = detect_break(df, "upload_count")
    break_reach   = detect_break(df, "weighted_reach")
    break_smooth  = detect_break(df, f"upload_count_roll{ROLLING_WINDOW}q")

    candidate_breaks = [y for y in [break_upload, break_reach, break_smooth] if y is not None]
    best_break = min(candidate_breaks) if candidate_breaks else None

    # ── Z-score signal first-fire ──────────────────────────────────────────
    # Use weighted_reach as primary — it's more meaningful than raw upload count
    z_fire_reach  = first_z_fire(df, f"weighted_reach_roll{ROLLING_WINDOW}q", Z_THRESHOLD)
    z_fire_upload = first_z_fire(df, f"upload_count_roll{ROLLING_WINDOW}q", Z_THRESHOLD)

    # Take earliest of the two
    z_candidates = [y for y in [z_fire_reach, z_fire_upload] if y is not None]
    best_z_fire  = min(z_candidates) if z_candidates else None

    # ── Large channel first appearance ────────────────────────────────────
    # First quarter where pct_large_channels > 30% (mainstream has arrived)
    large_ch_col = "pct_large_channels_roll4q"
    if large_ch_col in df.columns:
        mainstream_rows = df[df[large_ch_col] > 30]
        mainstream_year = mainstream_rows.index[0].year if not mainstream_rows.empty else None
    else:
        mainstream_year = None

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / f"{loc_id}_youtube_features.csv"
    df.to_csv(out_path)

    lead_break = lead_time(best_break, inflection_year)
    lead_z     = lead_time(best_z_fire, inflection_year)
    lead_main  = lead_time(mainstream_year, inflection_year)

    print(f"  ✓  Saved {len(df)} quarters → {out_path}")
    print(f"     Structural break: {best_break}  |  Z-fire: {best_z_fire}  |  Mainstream: {mainstream_year}")
    print(f"     Lead times — break: {lead_break} yrs, z: {lead_z} yrs, mainstream: {lead_main} yrs")

    return {
        "location_id":              loc_id,
        "location_name":            loc_name,
        "price_inflection_year":    inflection_year,
        "best_break_year":          best_break,
        "z_fire_year":              best_z_fire,
        "mainstream_arrival_year":  mainstream_year,
        "lead_time_break_yrs":      lead_break,
        "lead_time_z_yrs":          lead_z,
        "lead_time_mainstream_yrs": lead_main,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    locations = load_locations(CONFIG_PATH)
    summary   = []

    print("📊  Processing YouTube data\n")

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
            "mainstream_arrival_year",
            "lead_time_break_yrs",
            "lead_time_z_yrs",
            "lead_time_mainstream_yrs",
        ]].to_string(index=False))
        print(f"\n  ✓  Summary saved → {summary_path}")

    print("\n✅  YouTube processing complete.")


if __name__ == "__main__":
    main()
