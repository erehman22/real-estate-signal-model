"""
Google Trends data collection for all case study locations.
Pulls multiple query types and geo filters per location.
"""

import time
import yaml
import pandas as pd
from pathlib import Path
from pytrends.request import TrendReq
from tqdm import tqdm
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("config/locations.yaml")
OUTPUT_DIR  = Path("data/raw/google_trends")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME   = "2008-01-01 2024-12-31"
GEO_FILTERS = ["", "GB", "DE", "US"]   # worldwide + key source countries
SLEEP_SHORT = 6
SLEEP_LONG  = 35
BATCH_EVERY = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def build_queries(location: dict) -> dict[str, str]:
    base = location["google_trends_query"]
    return {
        "awareness":    base,
        "travel":       f"{base} travel",
        "property":     f"{base} property",
        "real_estate":  f"{base} real estate",
        "buy":          f"{base} buy",
        "nomad":        f"{base} digital nomad",
        "expat":        f"{base} expat",
        "move_to":      f"move to {base}",
    }


def pull_single_query(
    pytrends: TrendReq,
    query: str,
    timeframe: str,
    geo: str,
) -> Optional[pd.DataFrame]:
    try:
        pytrends.build_payload(
            [query],
            cat=0,
            timeframe=timeframe,
            geo=geo,
            gprop="",
        )
        df = pytrends.interest_over_time()
        if df.empty:
            return None
        df = df.drop(columns=["isPartial"], errors="ignore")
        df.columns = ["interest"]
        df.index.name = "date"
        return df
    except Exception as e:
        print(f"  ❌  Error for '{query}' geo='{geo}': {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    locations = load_locations(CONFIG_PATH)
    pytrends  = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=3, backoff_factor=1.5)

    request_count = 0

    for loc in tqdm(locations, desc="Locations"):
        loc_id  = loc["id"]
        queries = build_queries(loc)

        print(f"\n📍 {loc['name']} ({loc_id})")

        for geo in GEO_FILTERS:
            geo_label = geo if geo else "WW"
            geo_dir   = OUTPUT_DIR / loc_id / geo_label
            geo_dir.mkdir(parents=True, exist_ok=True)

            for label, query in queries.items():
                out_path = geo_dir / f"{label}.csv"

                if out_path.exists():
                    print(f"  ✓  [{geo_label}] {label} already exists, skipping")
                    continue

                print(f"  ↓  [{geo_label}] {query}")
                df = pull_single_query(pytrends, query, TIMEFRAME, geo)

                if df is not None:
                    df.to_csv(out_path)
                    print(f"  ✓  Saved {len(df)} rows → {out_path}")
                else:
                    print(f"  ⚠️  No data returned")

                request_count += 1
                time.sleep(SLEEP_SHORT)

                if request_count % BATCH_EVERY == 0:
                    print(f"\n  ⏸  Pausing {SLEEP_LONG}s...\n")
                    time.sleep(SLEEP_LONG)

    print("\n✅ Collection complete.")


if __name__ == "__main__":
    main()