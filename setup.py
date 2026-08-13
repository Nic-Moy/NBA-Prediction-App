"""Interactive database setup workflow for the NBA props application.

Loads the shared player and team reference data, season-level team advanced
stats, player game logs, and per-game team advanced stats into Postgres. The
user can choose either a complete league game-log load or one player's logs.

Provides:
- load_all_data_to_db() -> runs the complete cache-loading workflow.

Uses: getplayerinfo.py for player, team, and game-log loaders, and
      team_stats.py for team advanced-stat loaders.
    
Run this script directly before main.py or server.py since those scripts read the data this places in the database.
"""

from getplayerinfo import (
    DEFAULT_SEASON,
    load_all_players_to_db,
    load_all_teams_to_db,
    load_all_game_logs_bulk,
    load_player_game_logs,
)
from team_stats import fetch_and_cache_team_stats, fetch_and_cache_team_game_advanced_stats


def load_all_data_to_db(userchoice: int, season: str):
    """Load all shared data plus either league-wide or one-player game logs.

    Called by this file's interactive command-line flow. Its cached database
    records are subsequently read by main.py and server.py.
    """
    print("Loading players...")
    count = load_all_players_to_db(active_only=True)
    print(f"  {count} players loaded to players table.")

    print("Loading teams...")
    count = load_all_teams_to_db()
    print(f"  {count} teams loaded to teams table.")

    print("Loading team advanced stats...")
    count = fetch_and_cache_team_stats(season=season)
    print(f"  {count} team advanced stat rows upserted.")

    if userchoice == 1:
        print("Loading game logs for all players (bulk, ~seconds)...")
        count = load_all_game_logs_bulk(season=season)
        print(f"  {count} total game log rows upserted to player logs table.")


    if userchoice == 2:
        player_name = input("Enter player name (ex: Stephen Curry): ").strip()
        count = load_player_game_logs(player_name, season=season)
        print(f"  {count} total game log rows upserted to player logs table.")

    print("Loading game-level team advanced stats...")
    count = fetch_and_cache_team_game_advanced_stats(season=season)
    print(f"  {count} team game advanced stat rows upserted.")


if __name__ == "__main__":
    # Testing code for setup.py only
    choice = 0
    season_input = input(f"Enter season [{DEFAULT_SEASON}]: ").strip()
    season = season_input or DEFAULT_SEASON

    while choice not in (1, 2):
        choice = int(input("Enter '1' to fill db with entire leagues game logs, enter '2' to insert a specific player: "))

    load_all_data_to_db(choice, season=season)
