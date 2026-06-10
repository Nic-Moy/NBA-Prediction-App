"""Team-level advanced stats fetch + cache layer.

Pulls OFF_RATING, DEF_RATING, NET_RATING, PACE, TS_PCT for all 30 teams in a
season via LeagueDashTeamStats and upserts to the team_advanced_stats Postgres
table. Used by model.py to attach opponent context features to game logs.
"""

from __future__ import annotations

import argparse
import time

from nba_api.stats.endpoints import (
    leaguedashteamstats,
    boxscoreadvancedv2,
    boxscoreadvancedv3,
)

from database import (
    ensure_tables,
    is_team_stats_fresh,
    load_cached_game_ids_for_season,
    load_cached_team_game_ids_for_season,
    upsert_team_game_advanced_stats,
    upsert_team_advanced_stats,
)

DEFAULT_SEASON = "2025-26"


def _normalize_game_id(game_id) -> str | None:
    """Normalize game IDs to nba_api's required 10-digit string format."""
    if game_id is None:
        return None

    text = str(game_id).strip()
    if not text or text.lower() == "nan":
        return None

    # Handle accidental float-like strings from CSV/SQL adapters (e.g. 22500001.0)
    if text.endswith(".0"):
        text = text[:-2]

    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None

    # nba_api GameID regex is ^\\d{10}$.
    if len(digits) < 10:
        digits = digits.zfill(10)
    elif len(digits) > 10:
        digits = digits[-10:]
    return digits


def _extract_team_stats_frame(frames: list) -> object:
    """Pick the TeamStats-like frame from endpoint data frames."""
    if not frames:
        raise ValueError("No data frames returned")

    # Prefer explicit TeamStats-like schema instead of assuming index 1.
    for df in frames:
        cols = set(getattr(df, "columns", []))
        if {"GAME_ID", "TEAM_ID", "OFF_RATING", "DEF_RATING", "PACE"}.issubset(cols):
            return df
        if {"gameId", "teamId", "offensiveRating", "defensiveRating", "pace"}.issubset(cols):
            return df

    # Fall back to historical index position if schema check missed.
    if len(frames) > 1:
        return frames[1]
    return frames[0]


def _fetch_team_advanced_boxscore(game_id: str):
    """Fetch team advanced boxscore with V2 then V3 fallback."""
    v2_error = None
    try:
        box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id, timeout=30)
        return _extract_team_stats_frame(box.get_data_frames()), "v2"
    except Exception as e:
        v2_error = e

    try:
        box = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id, timeout=30)
        return _extract_team_stats_frame(box.get_data_frames()), "v3"
    except Exception as e:
        raise RuntimeError(f"v2 failed ({v2_error}); v3 failed ({e})")


def fetch_and_cache_team_stats(
    season: str = DEFAULT_SEASON,
    force: bool = False,
    max_age_hours: float = 20.0,) -> int:
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


def fetch_and_cache_team_game_advanced_stats(
    season: str = DEFAULT_SEASON,
    force: bool = False,
    max_games: int | None = None,) -> int:
    """Backfill per-game team advanced stats for cached game IDs in a season."""
    ensure_tables()
    cached_games = load_cached_game_ids_for_season(season)
    if cached_games.empty:
        print(f"  No cached game IDs found for {season}; skipping game-level team stats.")
        return 0

    existing_game_ids = load_cached_team_game_ids_for_season(season)
    pending = cached_games if force else cached_games[~cached_games["game_id"].isin(existing_game_ids)]
    if max_games is not None:
        pending = pending.head(max(0, int(max_games)))

    if pending.empty:
        print(f"  Team game advanced stats for {season} are already cached.")
        return 0

    print(f"  Backfilling team game advanced stats for {len(pending)} games ({season})...")
    rows_upserted = 0
    for idx, row in pending.reset_index(drop=True).iterrows():
        raw_game_id = row["game_id"]
        game_id = _normalize_game_id(raw_game_id)
        game_date = row["game_date"]
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"    Progress: {idx + 1}/{len(pending)}")

        if not game_id:
            print(f"    Warning: invalid game_id '{raw_game_id}' — skipping.")
            continue

        try:
            time.sleep(0.6)
            team_df, endpoint_used = _fetch_team_advanced_boxscore(game_id)
            rows_upserted += upsert_team_game_advanced_stats(team_df, season=season, game_date=game_date)
            if (idx + 1) <= 3:
                print(f"    Retrieved {game_id} via {endpoint_used}")
        except Exception as e:
            print(f"    Warning: failed game {game_id} (raw={raw_game_id}): {e}")

    print(f"  Done. {rows_upserted} team game advanced rows upserted ({season}).")
    return rows_upserted


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch + cache team advanced stats.")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--force", action="store_true", help="Bypass freshness cache.")
    parser.add_argument(
        "--game-level",
        action="store_true",
        help="Backfill game-level team advanced stats from cached game IDs.",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Limit game backfill rows.")
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()
    fetch_and_cache_team_stats(season=args.season, force=args.force)
    if args.game_level:
        fetch_and_cache_team_game_advanced_stats(
            season=args.season,
            force=args.force,
            max_games=args.max_games,
        )


if __name__ == "__main__":
    main()
