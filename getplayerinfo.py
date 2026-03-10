"""
NBA player info utilities.

This module is intentionally focused on nba_api player data only:
- resolve player name -> player ID
- fetch raw game logs
- clean game logs into modeling-friendly columns
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog, commonteamroster
from nba_api.stats.static import players, teams as nba_teams
from database import ensure_tables, cache_player, cache_team, update_player_team_position


DEFAULT_SEASON = "2025-26"


def find_player_id(name: str) -> int | None:
    """Resolve a player full name to NBA player ID."""
    matches = players.find_players_by_full_name(name)
    name_lower = name.strip().lower()
    exact = [p for p in matches if p["full_name"].lower() == name_lower]
    if not exact:
        return None
    return exact[0]["id"]


def get_player_stats(player_id: int, season: str = DEFAULT_SEASON) -> pd.DataFrame | None:
    """Fetch raw game logs for a player from nba_api."""
    try:
        time.sleep(.6)
        game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=10)
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
    """Keep required columns and normalize datatypes."""
    keep_cols = ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"]
    df = raw_df[keep_cols].copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    df["MIN"] = df["MIN"].apply(_minutes_to_float)
    for col in ["PTS", "REB", "AST"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


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


def backfill_player_team_and_position(max_retries: int = 3) -> int:
    """Fetch each team's roster and update team_id + position for all players.

    Makes 30 API calls (one per team) rather than one per player.
    Retries up to max_retries times on timeout before skipping a team.
    Returns total player rows updated.
    """
    team_list = nba_teams.get_teams()
    total = 0
    for t in team_list:
        time.sleep(1.0)
        df = None
        for attempt in range(1, max_retries + 1):
            try:
                roster = commonteamroster.CommonTeamRoster(team_id=t["id"], timeout=30)
                df = roster.get_data_frames()[0]
                break
            except Exception as e:
                print(f"  Attempt {attempt}/{max_retries} failed for {t['full_name']}: {e or type(e).__name__}")
                if attempt < max_retries:
                    time.sleep(2.0 * attempt)
        if df is None:
            print(f"  Skipping {t['full_name']} after {max_retries} failed attempts.")
            continue
        for _, row in df.iterrows():
            update_player_team_position(
                player_id=int(row["PLAYER_ID"]),
                team_id=t["id"],
                position=str(row.get("POSITION", "") or ""),
            )
            total += 1
        print(f"  ✓ {t['full_name']}: {len(df)} players updated")
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
