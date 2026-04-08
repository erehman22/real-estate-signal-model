"""
Historical route count estimation via Wayback Machine.

For each airport, fetches one archived timetable page per year
and counts the number of destination cities mentioned.
This gives us a year-by-year proxy for route growth.

Runs slowly by design — be respectful of Wayback Machine's servers.
"""

import re
import time
import yaml
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("config/locations.yaml")
RAW_DIR     = Path("data/raw/airline_routes")
OUTPUT_DIR  = Path("data/processed/airline_routes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_WAYBACK = 3   # seconds between Wayback requests
START_YEAR    = 2008
END_YEAR      = 2024

# Timetable page templates per airline
# These are the URLs we'll look for in Wayback
AIRLINE_TIMETABLE_URLS = {
    "ryanair": "https://www.ryanair.com/gb/en/cheap-flights-from-{code_lower}.html",
    "wizz":    "https://wizzair.com/en-gb/flights/timetable/{code}",
    "easyjet": "https://www.easyjet.com/en/cheap-flights/{code_lower}",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def get_wayback_snapshot_for_year(url: str, year: int) -> str | None:
    """Get the best Wayback Machine snapshot URL for a given year."""
    cdx_url = "http://web.archive.org/cdx/search/cdx"
    params  = {
        "url":      url,
        "output":   "json",
        "fl":       "timestamp,statuscode",
        "filter":   "statuscode:200",
        "from":     f"{year}0601",   # mid-year snapshot preferred
        "to":       f"{year}1231",
        "limit":    1,
    }
    try:
        resp = requests.get(cdx_url, params=params, timeout=10)
        data = resp.json()
        if len(data) <= 1:
            return None
        ts = data[1][0]
        return f"https://web.archive.org/web/{ts}/{url}"
    except Exception:
        return None


def count_destinations_in_page(snapshot_url: str) -> int | None:
    """
    Fetch a Wayback snapshot and count destination cities.
    Uses heuristics — looks for city names in flight listing elements.
    Returns count or None on failure.
    """
    try:
        resp = requests.get(snapshot_url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Heuristic 1: count links containing "flights-to" or "to-"
        flight_links = soup.find_all("a", href=re.compile(r"flights?[-/]to[-/]", re.I))
        if len(flight_links) > 5:
            return len(set(
                re.sub(r".*flights?[-/]to[-/]", "", a["href"]).split("/")[0]
                for a in flight_links
            ))

        # Heuristic 2: count items in route/destination lists
        list_items = soup.select("ul.routes li, ul.destinations li, .route-item, .destination-item")
        if list_items:
            return len(list_items)

        # Heuristic 3: count city-like text patterns
        city_pattern = re.compile(r'\b[A-Z][a-z]{2,}\b')
        text_block   = soup.get_text()
        cities       = set(city_pattern.findall(text_block))
        # Filter to plausible city names (rough heuristic)
        cities = {c for c in cities if 3 < len(c) < 20}
        return len(cities) if len(cities) > 10 else None

    except Exception:
        return None


def build_route_history(
    airport_code: str,
    airline_key: str,
    url_template: str,
) -> pd.DataFrame:
    """
    Build year-by-year route count estimates for one airport + airline.
    """
    code_lower = airport_code.lower()
    url        = url_template.format(code=airport_code, code_lower=code_lower)

    records = []
    for year in range(START_YEAR, END_YEAR + 1):
        snapshot_url = get_wayback_snapshot_for_year(url, year)
        time.sleep(SLEEP_WAYBACK)

        if snapshot_url is None:
            records.append({"year": year, "airline": airline_key,
                             "airport": airport_code, "route_count": None,
                             "snapshot_found": False})
            continue

        count = count_destinations_in_page(snapshot_url)
        time.sleep(SLEEP_WAYBACK)

        records.append({
            "year":           year,
            "airline":        airline_key,
            "airport":        airport_code,
            "route_count":    count,
            "snapshot_found": True,
            "snapshot_url":   snapshot_url,
        })
        print(f"    {year}: {count} destinations ({airline_key})")

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    locations    = load_locations(CONFIG_PATH)
    airport_codes = list(set(
        loc["nearest_airport"].split("(")[1].rstrip(")")
        for loc in locations
    ))

    print(f"🛫 Building historical route data for: {airport_codes}")
    print("⚠️  This will take a while — Wayback Machine requests are rate-limited.\n")

    all_records = []

    for code in tqdm(airport_codes, desc="Airports"):
        for airline_key, url_template in AIRLINE_TIMETABLE_URLS.items():
            out_path = RAW_DIR / code / f"wayback_{airline_key}_history.csv"

            if out_path.exists():
                print(f"  [{code}] {airline_key} cached, skipping")
                df = pd.read_csv(out_path)
                all_records.append(df)
                continue

            print(f"\n  [{code}] Fetching {airline_key} route history...")
            df = build_route_history(code, airline_key, url_template)
            df.to_csv(out_path, index=False)
            all_records.append(df)

    if all_records:
        combined = pd.concat(all_records, ignore_index=True)

        # Pivot to airport × year, summing routes across airlines
        pivot = combined.groupby(["airport", "year"])["route_count"].sum().unstack("year")
        pivot.to_csv(OUTPUT_DIR / "_route_counts_by_year.csv")

        print("\n✅ Route count pivot (airports × years):")
        print(pivot.to_string())

        # YoY route growth per airport
        growth = pivot.pct_change(axis=1) * 100
        growth.to_csv(OUTPUT_DIR / "_route_growth_pct.csv")

        print("\n📈 YoY route growth (%):")
        print(growth.round(1).to_string())


if __name__ == "__main__":
    main()