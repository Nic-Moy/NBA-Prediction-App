from getplayerinfo import (load_all_players_to_db, load_all_teams_to_db, load_all_game_logs_bulk, load_player_game_logs)

def load_all_data_to_db(userchoice: int):
    print("Loading players...")
    count = load_all_players_to_db(active_only=True)
    print(f"  {count} players loaded to players table.")

    print("Loading teams...")
    count = load_all_teams_to_db()
    print(f"  {count} teams loaded to teams table.")

    if userchoice == 1:
        print("Loading game logs for all players (bulk, ~seconds)...")
        count = load_all_game_logs_bulk()
        print(f"  {count} total game log rows upserted to player logs table.")


    if userchoice == 2:
        player_name = input("Enter player name (ex: Stephen Curry): ").strip()
        count = load_player_game_logs(player_name)
        print(f"  {count} total game log rows upserted to player logs table.")
    


if __name__ == "__main__":
    choice = 0

    while choice not in (1, 2):
        choice = int(input("Enter '1' to fill db with entire leagues game logs, enter '2' to insert a specific player: "))

    load_all_data_to_db(choice)