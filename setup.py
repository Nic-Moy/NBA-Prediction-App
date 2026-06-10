from getplayerinfo import (
    DEFAULT_SEASON,
    load_all_players_to_db,
    load_all_teams_to_db,
    load_all_game_logs_bulk,
    load_player_game_logs,
)
from team_stats import fetch_and_cache_team_stats, fetch_and_cache_team_game_advanced_stats


def load_all_data_to_db(userchoice: int, season: str):
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
    choice = 0
    season_input = input(f"Enter season [{DEFAULT_SEASON}]: ").strip()
    season = season_input or DEFAULT_SEASON

    while choice not in (1, 2):
        choice = int(input("Enter '1' to fill db with entire leagues game logs, enter '2' to insert a specific player: "))

    load_all_data_to_db(choice, season=season)
