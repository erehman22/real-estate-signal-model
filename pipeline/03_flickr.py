"""
Flickr geotagged photo upload pipeline for all case study locations.

For each location, pulls the number of publicly geotagged photos taken
within a defined radius, aggregated by quarter, from 2005 to present.

Why Flickr?
  - Skews heavily towards early/enthusiast travellers — precedes mass tourism
  - Fully timestamped (date taken, not date uploaded)
  - Free API with generous rate limits
  - Historical depth to 2004
  - Geotagged photos are non-manipulable (tied to physical presence)

We track TWO dates per photo:
  - date_taken: when the photographer was actually there (what we want)
  - date_upload: when they posted it (proxy for lag between visit and sharing)

The signal of interest is quarterly upload count normalised to a pre-boom
baseline — same approach as the Google Trends pipeline.

Setup:
  1. Get a free Flickr API key at https://www.flickr.com/services/apps/create/
  2. Add to .env: FLICKR_API_KEY=your_key_here
  3. pip install flickrapi python-dotenv tqdm pyyaml pandas

Output:
  data/raw/flickr/{location_id}/photos_quarterly.csv
    columns: year, quarter, photo_count, unique_users, avg_upload_lag_days

Rate limits:
  Flickr allows 3,600 requests/hour for free API keys.
  We use flickr.photos.search with bbox + date range, paginated.
  Each quarter = 1–5 API calls depending on photo volume.
  Full run for 10 locations × ~80 quarters ≈ ~800–4000 calls → well within limit.
  We still add a short sleep between calls to be polite.
"""

import os
import time
import yaml
import math
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import flickrapi
except ImportError:
    raise ImportError("Run: pip install flickrapi")

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

CONFIG_PATH = Path("config/locations.yaml")
OUTPUT_DIR  = Path("data/raw/flickr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLICKR_API_KEY    = os.getenv("FLICKR_API_KEY")
FLICKR_API_SECRET = os.getenv("FLICKR_API_SECRET", "")  # not needed for read-only

# Search radius in km around location coordinates
# Smaller = more precise but may miss nearby shots; 10km works well for towns
SEARCH_RADIUS_KM = 10

# Date range — Flickr has data from ~2004; we go 2005 to be safe
START_YEAR = 2005
END_YEAR   = datetime.now().year

# Flickr returns max 500 results per page; 250 is a safe page size
PAGE_SIZE  = 250

# Seconds between API calls — be polite even though limits are generous
SLEEP_BETWEEN_CALLS = 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def km_to_deg_lat(km: float) -> float:
    """Approximate km to degrees latitude (constant everywhere)."""
    return km / 111.0


def km_to_deg_lon(km: float, lat: float) -> float:
    """Approximate km to degrees longitude (varies with latitude)."""
    return km / (111.0 * math.cos(math.radians(lat)))


def build_bbox(lat: float, lon: float, radius_km: float) -> str:
    """
    Build a Flickr bbox string: min_lon,min_lat,max_lon,max_lat
    Flickr accepts up to 1 degree in any direction.
    """
    dlat = km_to_deg_lat(radius_km)
    dlon = km_to_deg_lon(radius_km, lat)
    min_lon = max(-180, lon - dlon)
    max_lon = min(180,  lon + dlon)
    min_lat = max(-90,  lat - dlat)
    max_lat = min(90,   lat + dlat)
    return f"{min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}"


def quarters_in_range(start_year: int, end_year: int) -> list[tuple[int, int, str, str]]:
    """
    Returns list of (year, quarter, start_date_str, end_date_str)
    for all quarters in range, inclusive.
    """
    quarters = []
    q_starts = ["-01-01", "-04-01", "-07-01", "-10-01"]
    q_ends   = ["-03-31", "-06-30", "-09-30", "-12-31"]
    for year in range(start_year, end_year + 1):
        for q_idx in range(4):
            # Skip future quarters
            q_start = date.fromisoformat(f"{year}{q_starts[q_idx]}")
            if q_start > date.today():
                break
            quarters.append((
                year,
                q_idx + 1,
                f"{year}{q_starts[q_idx]}",
                f"{year}{q_ends[q_idx]}",
            ))
    return quarters


def search_photos_in_period(
    flickr: "flickrapi.FlickrAPI",
    bbox: str,
    date_start: str,
    date_end: str,
) -> tuple[int, int, list[float]]:
    """
    Search for geotagged photos taken within bbox during date range.

    Returns:
        photo_count    — total number of public geotagged photos
        unique_users   — number of distinct photographers
        upload_lags    — list of (upload_ts - taken_ts) in days, for lag calc

    Handles pagination automatically.
    """
    photo_count  = 0
    unique_users = set()
    upload_lags  = []

    page     = 1
    per_page = PAGE_SIZE

    while True:
        try:
            result = flickr.photos.search(
                bbox=bbox,
                min_taken_date=date_start,
                max_taken_date=date_end,
                extras="date_taken,date_upload,owner_name",
                sort="date-taken-asc",
                per_page=per_page,
                page=page,
                has_geo=1,       # only geotagged photos
                content_type=1,  # photos only, not screenshots/illustrations
            )
        except Exception as e:
            print(f"    ⚠️  API error on page {page}: {e}")
            break

        photos    = result.find("photos")
        total     = int(photos.attrib.get("total", 0))
        pages     = int(photos.attrib.get("pages", 1))
        photo_els = photos.findall("photo")

        photo_count += len(photo_els)

        for photo in photo_els:
            owner = photo.attrib.get("owner", "")
            if owner:
                unique_users.add(owner)

            # Calculate upload lag if both dates available
            taken  = photo.attrib.get("datetaken", "")
            upload = photo.attrib.get("dateupload", "")
            if taken and upload:
                try:
                    taken_dt  = datetime.strptime(taken, "%Y-%m-%d %H:%M:%S")
                    upload_dt = datetime.fromtimestamp(int(upload))
                    lag_days  = (upload_dt - taken_dt).days
                    # Ignore implausible values (camera clock issues)
                    if 0 <= lag_days <= 3650:
                        upload_lags.append(lag_days)
                except (ValueError, OSError):
                    pass

        time.sleep(SLEEP_BETWEEN_CALLS)

        if page >= pages or total == 0:
            break

        # Safety: Flickr caps at 4000 results per query (page 16 × 250)
        # If a quarter has more than this, we'll undercount — log it
        if page >= 16:
            print(f"    ℹ️  Hit 4000-result cap for period {date_start}–{date_end}. "
                  f"True total ≥ {total}. Consider splitting into monthly queries.")
            break

        page += 1

    return photo_count, len(unique_users), upload_lags


# ── Main ──────────────────────────────────────────────────────────────────────

def process_location(
    flickr: "flickrapi.FlickrAPI",
    loc: dict,
) -> pd.DataFrame:
    """
    Pull quarterly Flickr photo counts for a single location.
    Returns a DataFrame ready to save.
    """
    lat  = loc["coordinates"]["lat"]
    lon  = loc["coordinates"]["lon"]
    bbox = build_bbox(lat, lon, SEARCH_RADIUS_KM)

    quarters = quarters_in_range(START_YEAR, END_YEAR)
    records  = []

    for year, quarter, date_start, date_end in tqdm(
        quarters,
        desc=f"  {loc['name']}",
        leave=False,
    ):
        count, users, lags = search_photos_in_period(
            flickr, bbox, date_start, date_end
        )
        avg_lag = round(sum(lags) / len(lags), 1) if lags else None

        records.append({
            "year":               year,
            "quarter":            quarter,
            "period_start":       date_start,
            "period_end":         date_end,
            "photo_count":        count,
            "unique_users":       users,
            "avg_upload_lag_days": avg_lag,
        })

    df = pd.DataFrame(records)
    df["year_quarter"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    return df


def main():
    if not FLICKR_API_KEY:
        raise ValueError(
            "FLICKR_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://www.flickr.com/services/apps/create/"
        )

    flickr = flickrapi.FlickrAPI(
        FLICKR_API_KEY,
        FLICKR_API_SECRET,
        format="etree",   # returns XML ElementTree — simpler than JSON for pagination
    )

    locations = load_locations(CONFIG_PATH)

    print(f"\n📷  Flickr photo collection")
    print(f"    Radius:     {SEARCH_RADIUS_KM} km")
    print(f"    Date range: {START_YEAR} – {END_YEAR}")
    print(f"    Locations:  {len(locations)}\n")

    for loc in tqdm(locations, desc="Locations"):
        loc_id   = loc["id"]
        loc_name = loc["name"]

        loc_dir  = OUTPUT_DIR / loc_id
        loc_dir.mkdir(exist_ok=True)
        out_path = loc_dir / "photos_quarterly.csv"

        if out_path.exists():
            print(f"\n  ✓  {loc_name}: already collected, skipping")
            continue

        print(f"\n📍  {loc_name}")

        df = process_location(flickr, loc)
        df.to_csv(out_path, index=False)

        total_photos = df["photo_count"].sum()
        peak_quarter = df.loc[df["photo_count"].idxmax(), "year_quarter"]

        print(f"  ✓  Saved {len(df)} quarters → {out_path}")
        print(f"     Total photos: {total_photos:,}   Peak quarter: {peak_quarter}")

    print("\n✅  Flickr collection complete.")


if __name__ == "__main__":
    main()