from getplayerinfo import (load_all_players_to_db, load_all_teams_to_db, load_all_game_logs_bulk, load_all_game_logs)

def load_all_data_to_db():
    print("Loading players...")
    count = load_all_players_to_db(active_only=True)
    print(f"  {count} players loaded.")

    print("Loading teams...")
    count = load_all_teams_to_db()
    print(f"  {count} teams loaded.")

    # print("Loading game logs for 1 player")
    # count = load_all_game_logs()
    # print(f"  {count} total game log rows upserted.")

    print("Loading game logs for all players (bulk, ~seconds)...")
    count = load_all_game_logs_bulk()
    print(f"  {count} total game log rows upserted.")


if __name__ == "__main__":
    load_all_data_to_db()