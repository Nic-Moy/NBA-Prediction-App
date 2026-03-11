from getplayerinfo import (
    load_all_players_to_db,
    load_all_teams_to_db,
    backfill_player_team_and_position,
    load_all_game_logs,
)

if __name__ == "__main__":
    print("Loading players...")
    count = load_all_players_to_db(active_only=True)
    print(f"  {count} players loaded.\n")

    print("Loading teams...")
    count = load_all_teams_to_db()
    print(f"  {count} teams loaded.\n")

    print("Backfilling player team_id and position...")
    count = backfill_player_team_and_position()
    print(f"  {count} player rows updated.\n")

    print("Loading game logs for all players (this takes ~5-10 min)...")
    count = load_all_game_logs()
    print(f"  {count} total game log rows upserted.")
