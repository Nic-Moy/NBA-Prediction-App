"""Team-level advanced stats fetch + cache layer.

Pulls OFF_RATING, DEF_RATING, NET_RATING, PACE, TS_PCT for all 30 teams in a
season via LeagueDashTeamStats and upserts to the team_advanced_stats Postgres
table. Used by model.py to attach opponent context features to game logs.
"""

from __future__ import annotations

import argparse
import time

from nba_api.stats.endpoints import leaguedashteamstats

from database import (
    ensure_tables,
    is_team_stats_fresh,
    upsert_team_advanced_stats,
)

DEFAULT_SEASON = "2025-26"


def fetch_and_cache_team_stats(
    season: str = DEFAULT_SEASON,
    force: bool = False,
    max_age_hours: float = 20.0,
) -> int:
    """Fetch advanced team stats and upsert to Postgres.

    Skips the API call if cache is fresh and force=False.
    Returns rows upserted (0 if skipped or empty response).
    """
    ensure_tables()

    if not force and is_team_stats_fresh(season, max_age_hours=max_age_hours):
        print(f"  Team advanced stats for {season} are fresh — skipping fetch.")
        return 0

    print(f"  Fetching team advanced stats for {season}...")
    time.sleep(0.6)
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=30,
        )
        df = stats.get_data_frames()[0]
    except Exception as e:
        print(f"  Error fetching team stats: {e}")
        return 0

    if df.empty:
        print("  No team stats returned.")
        return 0

    return upsert_team_advanced_stats(df, season)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch + cache team advanced stats.")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--force", action="store_true", help="Bypass freshness cache.")
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()
    fetch_and_cache_team_stats(season=args.season, force=args.force)


if __name__ == "__main__":
    main()
