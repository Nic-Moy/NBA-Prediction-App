from getplayerinfo import load_all_players_to_db, load_all_teams_to_db, backfill_player_team_and_position

if __name__ == "__main__":
    print("Loading players...")
    count = load_all_players_to_db(active_only=True)
    print(f"  {count} players loaded.")

    print("Loading teams...")
    count = load_all_teams_to_db()
    print(f"  {count} teams loaded.")

    print("Backfilling player team_id and position (~30 API calls)...")
    count = backfill_player_team_and_position()
    print(f"  {count} player rows updated.")
