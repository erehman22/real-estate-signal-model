"""
YouTube travel video signal collection for all case study locations.

For each location, searches YouTube for travel content using a single
query per location ("[place] travel"), pulls video metadata, and
aggregates into quarterly time series.

Why YouTube?
  - Free API (10,000 units/day, search = 100 units, videos.list = 1 unit)
  - Historical content dating back to ~2006
  - View velocity on early videos = measure of whether content found an audience
  - Channel size at upload time = proxy for signal reach
  - Comment volume = proxy for tourism/relocation intent

Signal design (Option A — single query, maximum metadata per result):
  - Query: "[place] travel"
  - Pull up to 200 results per location (2 search pages × 50 results)
  - For each video: title, description, upload date, view count,
    like count, comment count, channel subscriber count at pull time
  - Aggregate into quarterly buckets by upload date
  - Key derived features:
      - upload_count: how many travel videos were published per quarter
      - total_views_on_early_videos: cumulative views on videos uploaded
        in each quarter (measured now — reveals which cohorts found audiences)
      - median_channel_subs: typical reach of channels covering the place
      - weighted_view_score: upload_count × median_channel_subs (reach × volume)

Quota usage:
  - 1 location = 2 search calls (100 units each) + up to 200 video detail
    calls batched into 2 calls (1 unit each, 50 videos/batch) = ~204 units
  - 10 locations = ~2,040 units — well within 10,000/day limit
  - Channel subscriber counts require 1 additional channels.list call
    per unique channel — typically 50–150 unique channels per location,
    batched 50/call = 1–3 calls per location at 1 unit each

Setup:
  1. Go to console.cloud.google.com
  2. Create a project → Enable "YouTube Data API v3"
  3. Credentials → Create API Key
  4. Add to .env: YOUTUBE_API_KEY=your_key_here
  5. pip install google-api-python-client python-dotenv tqdm pyyaml pandas

Output:
  data/raw/youtube/{location_id}/videos.csv         — one row per video
  data/raw/youtube/{location_id}/quarterly.csv      — aggregated by quarter
"""

import os
import time
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    raise ImportError("Run: pip install google-api-python-client")

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

CONFIG_PATH = Path("config/locations.yaml")
OUTPUT_DIR  = Path("data/raw/youtube")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Number of search result pages to pull per location
# 1 page = 50 results = 100 quota units
# 2 pages = 100 results = 200 quota units (safe for 10 locations/day)
SEARCH_PAGES = 2
RESULTS_PER_PAGE = 50

# Sleep between API calls — YouTube is generous but be polite
SLEEP_BETWEEN_CALLS = 0.3

# Date range — YouTube launched 2005; we start 2006 to be safe
SEARCH_PUBLISHED_AFTER = "2006-01-01T00:00:00Z"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["locations"]


def build_youtube_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def search_videos(
    youtube,
    query: str,
    page_token: str | None = None,
) -> tuple[list[dict], str | None]:
    """
    Execute one search page. Returns (items, next_page_token).
    Each call costs 100 quota units.
    """
    params = {
        "part":            "snippet",
        "q":               query,
        "type":            "video",
        "videoCategoryId": "19",          # Travel & Events category
        "maxResults":      RESULTS_PER_PAGE,
        "publishedAfter":  SEARCH_PUBLISHED_AFTER,
        "relevanceLanguage": "en",
        "order":           "relevance",   # relevance surfaces canonical content best
    }
    if page_token:
        params["pageToken"] = page_token

    try:
        response = youtube.search().list(**params).execute()
        time.sleep(SLEEP_BETWEEN_CALLS)
        return response.get("items", []), response.get("nextPageToken")
    except HttpError as e:
        print(f"  ⚠️  Search error: {e}")
        return [], None


def get_video_details(youtube, video_ids: list[str]) -> list[dict]:
    """
    Fetch full statistics for a batch of up to 50 video IDs.
    Costs 1 quota unit per call.
    """
    if not video_ids:
        return []
    try:
        response = youtube.videos().list(
            part="statistics,snippet,contentDetails",
            id=",".join(video_ids),
        ).execute()
        time.sleep(SLEEP_BETWEEN_CALLS)
        return response.get("items", [])
    except HttpError as e:
        print(f"  ⚠️  Video details error: {e}")
        return []


def get_channel_subs(youtube, channel_ids: list[str]) -> dict[str, int | None]:
    """
    Fetch subscriber counts for a batch of up to 50 channel IDs.
    Costs 1 quota unit per call.
    Returns {channel_id: subscriber_count}.
    """
    if not channel_ids:
        return {}
    try:
        response = youtube.channels().list(
            part="statistics",
            id=",".join(channel_ids),
            maxResults=50,
        ).execute()
        time.sleep(SLEEP_BETWEEN_CALLS)
        result = {}
        for item in response.get("items", []):
            cid   = item["id"]
            stats = item.get("statistics", {})
            subs  = stats.get("subscriberCount")
            result[cid] = int(subs) if subs else None
        return result
    except HttpError as e:
        print(f"  ⚠️  Channel stats error: {e}")
        return {}


def parse_video_record(item: dict, channel_subs: dict[str, int | None]) -> dict:
    """Extract a flat record from a video API response item."""
    snippet    = item.get("snippet", {})
    stats      = item.get("statistics", {})
    channel_id = snippet.get("channelId", "")

    published_at = snippet.get("publishedAt", "")
    try:
        published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        published_dt = None

    return {
        "video_id":          item.get("id", ""),
        "title":             snippet.get("title", ""),
        "channel_id":        channel_id,
        "channel_title":     snippet.get("channelTitle", ""),
        "published_at":      published_at,
        "published_year":    published_dt.year    if published_dt else None,
        "published_quarter": ((published_dt.month - 1) // 3 + 1) if published_dt else None,
        "view_count":        int(stats.get("viewCount", 0) or 0),
        "like_count":        int(stats.get("likeCount", 0) or 0),
        "comment_count":     int(stats.get("commentCount", 0) or 0),
        "channel_subs":      channel_subs.get(channel_id),
        "description_len":   len(snippet.get("description", "")),
    }


def aggregate_to_quarterly(videos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate video-level data into quarterly time series.

    Key metrics:
      upload_count         — number of travel videos published per quarter
      total_views          — sum of current view counts for videos from that quarter
                             (high = that cohort found an audience over time)
      median_views         — median views per video in that quarter
      median_channel_subs  — typical channel size covering the place that quarter
      weighted_reach       — upload_count × median_channel_subs
                             (volume × typical audience size)
      total_comments       — sum of comments (proxy for engagement/intent)
      pct_large_channels   — % of uploads from channels with >100k subs
                             (measures whether mainstream travel media has arrived)
    """
    df = videos_df.dropna(subset=["published_year", "published_quarter"]).copy()
    df["published_year"]    = df["published_year"].astype(int)
    df["published_quarter"] = df["published_quarter"].astype(int)

    def pct_large(subs_series):
        valid = subs_series.dropna()
        if len(valid) == 0:
            return None
        return (valid > 100_000).mean() * 100

    quarterly = df.groupby(["published_year", "published_quarter"]).agg(
        upload_count        =("video_id",       "count"),
        total_views         =("view_count",      "sum"),
        median_views        =("view_count",      "median"),
        total_comments      =("comment_count",   "sum"),
        median_channel_subs =("channel_subs",    "median"),
        pct_large_channels  =("channel_subs",    pct_large),
    ).reset_index()

    # weighted_reach: upload volume × typical channel audience
    quarterly["weighted_reach"] = (
        quarterly["upload_count"] * quarterly["median_channel_subs"].fillna(0)
    )

    quarterly["year_quarter"] = (
        quarterly["published_year"].astype(str) + "-Q" +
        quarterly["published_quarter"].astype(str)
    )

    return quarterly.sort_values(["published_year", "published_quarter"]).reset_index(drop=True)


# ── Main collection ───────────────────────────────────────────────────────────

def collect_location(youtube, loc: dict) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Collect all video data for a single location.
    Returns (videos_df, quarterly_df) or None on failure.
    """
    loc_name = loc["name"]
    country  = loc["country"]
    query    = f"{loc_name} {country} travel"

    print(f"\n📍  {loc_name}  |  query: \"{query}\"")

    # ── Step 1: Search for video IDs ──────────────────────────────────────
    all_search_items = []
    next_page_token  = None

    for page_num in range(1, SEARCH_PAGES + 1):
        print(f"  🔍  Search page {page_num}/{SEARCH_PAGES}...")
        items, next_page_token = search_videos(youtube, query, next_page_token)
        all_search_items.extend(items)
        if not next_page_token:
            break

    if not all_search_items:
        print(f"  ⚠️  No results returned for {loc_name}")
        return None

    video_ids = [
        item["id"]["videoId"]
        for item in all_search_items
        if item.get("id", {}).get("kind") == "youtube#video"
    ]
    print(f"  ✓  Found {len(video_ids)} videos")

    # ── Step 2: Get full video details (batched, 50/call) ─────────────────
    all_video_details = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        details = get_video_details(youtube, batch)
        all_video_details.extend(details)

    print(f"  ✓  Retrieved details for {len(all_video_details)} videos")

    # ── Step 3: Get channel subscriber counts ─────────────────────────────
    channel_ids = list({
        item.get("snippet", {}).get("channelId", "")
        for item in all_video_details
        if item.get("snippet", {}).get("channelId")
    })

    channel_subs = {}
    for i in range(0, len(channel_ids), 50):
        batch  = channel_ids[i:i+50]
        subs   = get_channel_subs(youtube, batch)
        channel_subs.update(subs)

    print(f"  ✓  Retrieved subscriber counts for {len(channel_subs)} channels")

    # ── Step 4: Build video-level DataFrame ───────────────────────────────
    records = [
        parse_video_record(item, channel_subs)
        for item in all_video_details
    ]
    videos_df = pd.DataFrame(records)

    # ── Step 5: Aggregate to quarterly ────────────────────────────────────
    quarterly_df = aggregate_to_quarterly(videos_df)

    return videos_df, quarterly_df


def main():
    if not YOUTUBE_API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY not set.\n"
            "1. Go to console.cloud.google.com\n"
            "2. Create project → Enable YouTube Data API v3\n"
            "3. Credentials → Create API Key\n"
            "4. Add to .env: YOUTUBE_API_KEY=your_key_here"
        )

    youtube   = build_youtube_client(YOUTUBE_API_KEY)
    locations = load_locations(CONFIG_PATH)

    print(f"\n📺  YouTube travel video collection")
    print(f"    Search pages per location: {SEARCH_PAGES} ({SEARCH_PAGES * RESULTS_PER_PAGE} results max)")
    print(f"    Estimated quota usage: ~{len(locations) * (SEARCH_PAGES * 100 + 5)} / 10,000 units")
    print(f"    Locations: {len(locations)}\n")

    for loc in tqdm(locations, desc="Locations"):
        loc_id   = loc["id"]
        loc_dir  = OUTPUT_DIR / loc_id
        loc_dir.mkdir(exist_ok=True)

        videos_path    = loc_dir / "videos.csv"
        quarterly_path = loc_dir / "quarterly.csv"

        if videos_path.exists() and quarterly_path.exists():
            print(f"\n  ✓  {loc['name']}: already collected, skipping")
            continue

        result = collect_location(youtube, loc)
        if result is None:
            continue

        videos_df, quarterly_df = result

        videos_df.to_csv(videos_path, index=False)
        quarterly_df.to_csv(quarterly_path, index=False)

        print(f"  ✓  Saved {len(videos_df)} videos → {videos_path}")
        print(f"  ✓  Saved {len(quarterly_df)} quarters → {quarterly_path}")

        # Quick sanity check
        if not quarterly_df.empty:
            earliest = quarterly_df.iloc[0]["year_quarter"]
            peak_row = quarterly_df.loc[quarterly_df["upload_count"].idxmax()]
            print(f"     Earliest content: {earliest}")
            print(f"     Peak upload quarter: {peak_row['year_quarter']} "
                  f"({int(peak_row['upload_count'])} videos)")

    print("\n✅  YouTube collection complete.")


if __name__ == "__main__":
    main()
