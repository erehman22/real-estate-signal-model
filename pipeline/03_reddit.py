"""
Reddit mention frequency collection for all case study locations.

Pulls a window of 8 years pre-inflection + 2 years post per location.
Uses Pullpush.io (Pushshift mirror) — no auth required.

Output: data/raw/reddit/{loc_id}/{subreddit}.csv
"""

import time
import yaml
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH   = Path("config/locations.yaml")
OUTPUT_DIR    = Path("data/raw/reddit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL      = "https://api.pullpush.io/reddit/search/submission"
YEARS_PRE     = 8      # years before price inflection to collect
YEARS_POST    = 2      # years after price inflection to collect
REDDIT_FLOOR  = 2008   # Reddit data before this is too sparse to be useful
SLEEP         = 2
MAX_RETRIES   = 3

SUBREDDITS = [
    "travel",
    "solotravel",
    "digitalnomad",
    "expats",
    "personalfinance",
    "realestateinvesting",
    "liveabroad",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def get_window(inflection_year: int) -> tuple[datetime, datetime]:
    """Return start/end datetimes for this location's collection window."""
    start_year = max(REDDIT_FLOOR, inflection_year - YEARS_PRE)
    end_year   = inflection_year + YEARS_POST
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(end_year, 12, 31, tzinfo=timezone.utc)
    return start, end


def to_unix(dt: datetime) -> int:
    return int(dt.timestamp())


def fetch_monthly_counts(
    keywords: list[str],
    subreddit: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    records = []
    current = start

    while current < end:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)

        query = " OR ".join(f'"{kw}"' for kw in keywords)

        params = {
            "q":           query,
            "subreddit":   subreddit,
            "after":       to_unix(current),
            "before":      to_unix(next_month),
            "size":        0,
            "track_total": True,
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data  = resp.json()
                count = data.get("metadata", {}).get("total_results", 0)
                records.append({
                    "date":      current.strftime("%Y-%m-01"),
                    "subreddit": subreddit,
                    "count":     count,
                })
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)
                else:
                    print(f"    ❌ Failed {current.strftime('%Y-%m')}: {e}")
                    records.append({
                        "date":      current.strftime("%Y-%m-01"),
                        "subreddit": subreddit,
                        "count":     None,
                    })

        current = next_month
        time.sleep(SLEEP)

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    locations = load_locations(CONFIG_PATH)

    for loc in tqdm(locations, desc="Locations"):
        loc_id          = loc["id"]
        keywords        = loc["reddit_keywords"]
        inflection_year = loc["price_inflection_year"]

        start, end = get_window(inflection_year)
        window_str = f"{start.year}–{end.year}"

        loc_dir = OUTPUT_DIR / loc_id
        loc_dir.mkdir(exist_ok=True)

        print(f"\n📍 {loc['name']} | window: {window_str} | keywords: {keywords}")

        for subreddit in SUBREDDITS:
            out_path = loc_dir / f"{subreddit}.csv"

            if out_path.exists():
                print(f"  ✓  r/{subreddit} already collected, skipping")
                continue

            print(f"  ↓  r/{subreddit}")
            df = fetch_monthly_counts(keywords, subreddit, start, end)

            if not df.empty:
                df.to_csv(out_path, index=False)
                total = df["count"].sum()
                print(f"  ✓  {len(df)} months, {total} total mentions → {out_path}")
            else:
                print(f"  ⚠️  No data returned")

    print("\n✅ Reddit collection complete.")


if __name__ == "__main__":
    main()