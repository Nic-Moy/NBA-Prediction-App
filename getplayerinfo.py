"""
NBA player info utilities.

This module is intentionally focused on nba_api player data only:
- resolve player name -> player ID
- fetch raw game logs
- clean game logs into modeling-friendly columns
"""


import argparse
import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog, commonteamroster, leaguegamelog
from nba_api.stats.static import players, teams as nba_teams
from database import ensure_tables, cache_player, cache_team, get_all_player_ids, upsert_game_logs, is_cache_fresh, upsert_game_logs_bulk


DEFAULT_SEASON = "2025-26"
DEFAULT_SEASON_PHASES = ("Regular Season", "PlayIn", "Playoffs")


def find_player_id(name: str) -> int | None:
    """Resolve a player full name to NBA player ID."""
    matches = players.find_players_by_full_name(name)
    name_lower = name.strip().lower()
    exact = [p for p in matches if p["full_name"].lower() == name_lower]
    if not exact:
        return None
    return exact[0]["id"]


def get_player_stats(
    player_id: int,
    season: str = DEFAULT_SEASON,
    season_type: str = "Regular Season",
) -> pd.DataFrame | None:
    """Fetch raw game logs for a player from nba_api."""
    try:
        time.sleep(.6)
        game_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
            timeout=10,
        )
        return game_log.get_data_frames()[0]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def _minutes_to_float(value) -> float:
    """Convert MIN values like '32:45' to decimal minutes."""
    if pd.isna(value):
        return np.nan

    text = str(value)
    if ":" not in text:
        return float(text)

    minutes, seconds = text.split(":", maxsplit=1)
    return float(minutes) + (float(seconds) / 60.0)


def clean_player_games(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Keep raw stat columns and derive shooting percentages.

    Returns base box-score columns plus FG_PCT, FG3_PCT, FT_PCT, TS_PCT.
    Extra columns are kept only if present (cached rows may pre-date a column).
    """
    base_cols = ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"]
    id_cols = [c for c in ["Game_ID", "GAME_ID"] if c in raw_df.columns]
    extra_cols = [
        "WL", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "STL", "BLK", "TOV", "PLUS_MINUS",
    ]
    keep_cols = id_cols + base_cols + [c for c in extra_cols if c in raw_df.columns]
    df = raw_df[keep_cols].copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    df["MIN"] = df["MIN"].apply(_minutes_to_float)
    numeric_cols = [
        "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A",
        "FTM", "FTA", "STL", "BLK", "TOV", "PLUS_MINUS",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"FGM", "FGA"}.issubset(df.columns):
        df["FG_PCT"] = df["FGM"] / df["FGA"].replace(0, np.nan)
    if {"FG3M", "FG3A"}.issubset(df.columns):
        df["FG3_PCT"] = df["FG3M"] / df["FG3A"].replace(0, np.nan)
    if {"FTM", "FTA"}.issubset(df.columns):
        df["FT_PCT"] = df["FTM"] / df["FTA"].replace(0, np.nan)
    if {"PTS", "FGA", "FTA"}.issubset(df.columns):
        denom = 2.0 * (df["FGA"] + 0.44 * df["FTA"])
        df["TS_PCT"] = df["PTS"] / denom.replace(0, np.nan)

    return df


def parse_opponent_from_matchup(matchup: str) -> str | None:
    """Extract opponent abbreviation from MATCHUP string.

    'LAL vs. BOS' -> 'BOS'; 'BOS @ LAL' -> 'LAL'.
    Returns None for malformed input.
    """
    if not isinstance(matchup, str) or not matchup:
        return None
    normalized = matchup.replace("vs.", "@")
    parts = normalized.split("@")
    if len(parts) < 2:
        return None
    return parts[-1].strip() or None


def load_all_players_to_db(active_only: bool = True) -> int:
    """Fetch all NBA players from nba_api static data and upsert into the players table.

    Returns the number of players loaded.
    """
    ensure_tables()
    player_list = players.get_active_players() if active_only else players.get_players()
    for p in player_list:
        cache_player(
            player_id=p["id"],
            full_name=p["full_name"],
            is_active=p["is_active"],
        )
    return len(player_list)


def load_all_teams_to_db() -> int:
    """Load all 30 NBA teams into the teams table. Returns count loaded."""
    ensure_tables()
    team_list = nba_teams.get_teams()
    for t in team_list:
        cache_team(
            team_id=t["id"],
            full_name=t["full_name"],
            abbreviation=t["abbreviation"],
            city=t["city"],
            nickname=t["nickname"],
        )
    return len(team_list)


def load_player_game_logs(player_name: str, season: str = DEFAULT_SEASON, max_retries: int = 3) -> int:
    """Fetch and store game logs for a single player by name.

    Validates the player name, resolves it to an ID, fetches their game logs,
    and upserts them into the database. Returns the number of rows upserted.
    Raises ValueError for invalid or unrecognized player names.
    """
    if not player_name or not player_name.strip():
        raise ValueError("Player name cannot be empty.")

    player_name = player_name.strip()
    player_id = find_player_id(player_name)
    if player_id is None:
        raise ValueError(f"Player '{player_name}' not found. Check spelling and try again.")

    ensure_tables()
    print(f"  Fetching game logs for {player_name} (ID: {player_id}, season: {season})...")

    phase_frames: list[pd.DataFrame] = []
    for phase in DEFAULT_SEASON_PHASES:
        print(f"    Pulling phase: {phase}")
        raw_df = None
        for attempt in range(1, max_retries + 1):
            time.sleep(0.6)
            try:
                raw_df = get_player_stats(player_id, season=season, season_type=phase)
                break
            except Exception as e:
                print(f"    Attempt {attempt}/{max_retries} failed: {e or type(e).__name__}")
                if attempt < max_retries:
                    time.sleep(2.0 * attempt)

        if raw_df is not None and not raw_df.empty:
            phase_frames.append(raw_df)
        else:
            print(f"    No rows returned for phase: {phase}")

    if not phase_frames:
        print(f"  No game log data found for {player_name} in {season}.")
        return 0

    combined_df = pd.concat(phase_frames, ignore_index=True)
    rows = upsert_game_logs(combined_df, player_id, season)
    print(f"  Done. {rows} rows upserted for {player_name}.")
    return rows


def load_all_game_logs_bulk(season: str = DEFAULT_SEASON) -> int:
    """Fetch all player game logs for a season across all phases via LeagueGameLog.

    Pulls Regular Season + PlayIn + Playoffs, then upserts combined results.
    """
    ensure_tables()
    phase_frames: list[pd.DataFrame] = []

    for phase in DEFAULT_SEASON_PHASES:
        print(f"  Fetching all player game logs for {season} ({phase}) via LeagueGameLog...")
        time.sleep(0.6)
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=phase,
            player_or_team_abbreviation="P",
            timeout=60,
        )
        df = log.get_data_frames()[0]
        if df.empty:
            print(f"    No data returned for phase: {phase}")
            continue
        phase_frames.append(df)

    if not phase_frames:
        print("  No data returned from LeagueGameLog for any phase.")
        return 0

    combined_df = pd.concat(phase_frames, ignore_index=True)
    total = upsert_game_logs_bulk(combined_df, season)
    print(f"\nDone. {total} rows upserted via bulk load ({', '.join(DEFAULT_SEASON_PHASES)}).")
    return total


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and preview NBA player game logs from nba_api."
    )
    parser.add_argument("--player", help="Player full name (example: Stephen Curry)")
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help=f"NBA season string (default: {DEFAULT_SEASON})",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="How many recent cleaned rows to display (default: 10)",
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()
    valid_player = False

    while valid_player != True:
        player_name = args.player or input("Enter player name (ex: Stephen Curry): ").strip()
        season = args.season
    
        if not player_name:
            print("No player name provided. Please try again")
            continue

        player_id = find_player_id(player_name)
        if player_id is None:
            print(f"Player '{player_name}' not found. Please try again")
            continue

            

        valid_player = True

    print(f"Resolved player ID: {player_id}")
    print(f"Fetching season: {season}")

    raw_df = get_player_stats(player_id, season=season)
    if raw_df is None or raw_df.empty:
        print("No game log data returned.")
        print("Check the season format or nba_api connectivity.")
        return

    cleaned_df = clean_player_games(raw_df)

    print(f"\nTotal games fetched: {len(cleaned_df)}")
    print(f"Date range: {cleaned_df['GAME_DATE'].min().date()} to {cleaned_df['GAME_DATE'].max().date()}")
    print("\nRecent cleaned rows:")
    print(
        cleaned_df.tail(max(1, args.rows)).to_string(index=False)
    )


if __name__ == "__main__":
    main()
