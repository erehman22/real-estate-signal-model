"""
Airline route data collection for nearest airports to all case study locations.

Data source: OpenFlights (free, static) + Ryanair/Wizz route history via
web scraping where available.

Strategy:
  1. OpenFlights routes database — static snapshot but gives us a baseline
  2. Wayback Machine CDX API — find archived timetable pages for each airline
  3. For each airport, count unique routes per year as a proxy for
     accessibility growth

Output: data/raw/airline_routes/{airport_code}/
"""

import time
import yaml
import requests
import pandas as pd
from pathlib import Path
from io import StringIO
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("config/locations.yaml")
OUTPUT_DIR  = Path("data/raw/airline_routes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OpenFlights static data URLs
OPENFLIGHTS_ROUTES   = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
OPENFLIGHTS_AIRPORTS = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

SLEEP = 1

# LCC carriers we care about most as discovery proxies
LCC_CARRIERS = {
    "FR",  # Ryanair
    "U2",  # easyJet
    "W6",  # Wizz Air
    "VY",  # Vueling
    "PC",  # Pegasus
    "XW",  # NokScoot / Thai Lion
    "SL",  # Thai Lion Air
    "FD",  # Thai AirAsia
    "FZ",  # flydubai
    "G9",  # Air Arabia
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def fetch_openflights_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OpenFlights routes and airports databases."""
    print("↓ Downloading OpenFlights routes database...")
    routes_resp = requests.get(OPENFLIGHTS_ROUTES, timeout=30)
    routes_resp.raise_for_status()

    routes_cols = [
        "airline", "airline_id", "src_airport", "src_airport_id",
        "dst_airport", "dst_airport_id", "codeshare", "stops", "equipment"
    ]
    routes = pd.read_csv(
        StringIO(routes_resp.text),
        header=None,
        names=routes_cols,
        na_values=["\\N"],
    )

    print("↓ Downloading OpenFlights airports database...")
    airports_resp = requests.get(OPENFLIGHTS_AIRPORTS, timeout=30)
    airports_resp.raise_for_status()

    airports_cols = [
        "airport_id", "name", "city", "country", "iata", "icao",
        "lat", "lon", "altitude", "timezone", "dst", "tz_database",
        "type", "source"
    ]
    airports = pd.read_csv(
        StringIO(airports_resp.text),
        header=None,
        names=airports_cols,
        na_values=["\\N"],
    )

    return routes, airports


def get_routes_for_airport(
    airport_code: str,
    routes: pd.DataFrame,
) -> pd.DataFrame:
    """Filter routes where the airport is origin or destination."""
    mask = (
        (routes["src_airport"] == airport_code) |
        (routes["dst_airport"] == airport_code)
    )
    return routes[mask].copy()


def fetch_wayback_snapshots(url: str, from_year: int = 2008) -> list[dict]:
    """
    Use Wayback Machine CDX API to find archived snapshots of a URL.
    Returns list of {year, timestamp, snapshot_url}.
    """
    cdx_url = "http://web.archive.org/cdx/search/cdx"
    params  = {
        "url":        url,
        "output":     "json",
        "fl":         "timestamp,statuscode",
        "filter":     "statuscode:200",
        "from":       f"{from_year}0101",
        "to":         "20241231",
        "collapse":   "timestamp:6",  # one per month
        "limit":      200,
    }
    try:
        resp = requests.get(cdx_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if len(data) <= 1:
            return []
        # First row is headers
        snapshots = []
        for row in data[1:]:
            ts   = row[0]
            year = int(ts[:4])
            snapshots.append({
                "year":         year,
                "timestamp":    ts,
                "snapshot_url": f"https://web.archive.org/web/{ts}/{url}",
            })
        return snapshots
    except Exception as e:
        print(f"  ⚠️  Wayback CDX failed for {url}: {e}")
        return []


def count_ryanair_routes_from_wayback(airport_code: str) -> pd.DataFrame:
    """
    Check Wayback Machine for Ryanair timetable pages for an airport.
    Returns annual snapshot counts as a proxy for route history.
    Note: this gives us snapshot availability, not exact route counts.
    A full route scrape would require parsing each snapshot.
    """
    url = f"https://www.ryanair.com/gb/en/cheap-flights-from-{airport_code.lower()}.html"
    snapshots = fetch_wayback_snapshots(url)

    if not snapshots:
        return pd.DataFrame()

    df = pd.DataFrame(snapshots)
    annual = df.groupby("year").size().reset_index(name="snapshot_count")
    annual["airport"] = airport_code
    annual["source"]  = "wayback_ryanair"
    return annual


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    locations          = load_locations(CONFIG_PATH)
    routes, airports   = fetch_openflights_data()

    # Deduplicate airports to unique IATA codes
    airports_clean = airports.dropna(subset=["iata"]).copy()
    airports_clean = airports_clean[airports_clean["iata"] != "\\N"]

    # Get unique nearest airports across all locations
    airport_codes = list(set(
        loc["nearest_airport"].split("(")[1].rstrip(")")
        for loc in locations
    ))

    print(f"\n🛫 Processing {len(airport_codes)} airports: {airport_codes}")

    all_summaries = []

    for code in tqdm(airport_codes, desc="Airports"):
        airport_dir = OUTPUT_DIR / code
        airport_dir.mkdir(exist_ok=True)

        # ── 1. OpenFlights static routes ─────────────────────────────────────
        static_path = airport_dir / "openflights_routes.csv"
        if not static_path.exists():
            airport_routes = get_routes_for_airport(code, routes)
            airport_routes.to_csv(static_path, index=False)
            print(f"\n  [{code}] OpenFlights: {len(airport_routes)} routes found")
        else:
            airport_routes = pd.read_csv(static_path)
            print(f"\n  [{code}] OpenFlights: {len(airport_routes)} routes (cached)")

        # LCC-only subset
        lcc_routes = airport_routes[airport_routes["airline"].isin(LCC_CARRIERS)]
        lcc_path   = airport_dir / "lcc_routes.csv"
        lcc_routes.to_csv(lcc_path, index=False)
        print(f"  [{code}] LCC routes: {len(lcc_routes)}")

        # ── 2. Wayback Machine Ryanair timetable snapshots ───────────────────
        wayback_path = airport_dir / "wayback_ryanair_snapshots.csv"
        if not wayback_path.exists():
            print(f"  [{code}] Checking Wayback for Ryanair timetable...")
            wb_df = count_ryanair_routes_from_wayback(code)
            if not wb_df.empty:
                wb_df.to_csv(wayback_path, index=False)
                print(f"  [{code}] Wayback: {len(wb_df)} annual snapshots found")
            else:
                print(f"  [{code}] Wayback: no Ryanair timetable snapshots")
            time.sleep(SLEEP)
        else:
            print(f"  [{code}] Wayback (cached)")

        # ── 3. Summary stats per airport ─────────────────────────────────────
        all_summaries.append({
            "airport":        code,
            "total_routes":   len(airport_routes),
            "lcc_routes":     len(lcc_routes),
            "unique_airlines": airport_routes["airline"].nunique(),
        })

    # Save cross-airport summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_DIR / "_airport_summary.csv", index=False)
    print(f"\n✅ Airport summary:\n")
    print(summary_df.to_string(index=False))

    print("""
⚠️  NOTE: OpenFlights is a static snapshot (~2014 data).
   For historical route growth, use one of:
     - OAG (paid, best coverage)
     - Aviation Edge API (paid, affordable)
     - Flightradar24 statistics pages (manual)
     - Wayback Machine scraping of airline timetable pages (free, slow)
   
   The Wayback snapshot count above is a proxy — it tells us whether
   Ryanair was actively updating its timetable for that airport, which
   correlates with route activity, but doesn't give exact route counts.
   
   Next step: run 05b_airline_routes_historical.py to scrape 
   Wayback snapshots for route counts per year.
""")


if __name__ == "__main__":
    main()