import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


def find_player_id(name: str):
    hits = players.find_players_by_full_name(name)
    if hits:
        return hits[0]["id"]
    else:
        None


def get_player_stats(player_id, season='2024-25'):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        return df
    except:
        return None
    

def isolate_key_stats(df):

    key_stats = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT']
    cleaned_df = df[key_stats].copy()
    
    # Convert GAME_DATE to datetime
    cleaned_df['GAME_DATE'] = pd.to_datetime(cleaned_df['GAME_DATE'])
    
    # Sort by date (oldest to newest) for proper time-series analysis
    cleaned_df = cleaned_df.sort_values('GAME_DATE')
    
    # Convert MIN from string "32:45" to float (32.75 minutes)
    cleaned_df['MIN'] = cleaned_df['MIN'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 if ':' in str(x) else float(x))
    
    return cleaned_df


def calculate_averages(df, player_name):

    print(f"STATS SUMMARY FOR {player_name.upper()}")
    last_5_games = df.tail(5)

    print(f"\nTotal Games Played: {len(df)}")
    print(f"\nSeason Averages:")
    print(f"  Points:   {df['PTS'].mean():.1f} PPG")
    print(f"  Rebounds: {df['REB'].mean():.1f} RPG")
    print(f"  Assists:  {df['AST'].mean():.1f} APG")
    print(f"  Minutes:  {df['MIN'].mean():.1f} MPG")
    
    print(f"\nLast 5 Games Average:")
    print(f"  Points:   {last_5_games['PTS'].mean():.1f} PPG")
    print(f"  Rebounds: {last_5_games['REB'].mean():.1f} RPG")
    print(f"  Assists:  {last_5_games['AST'].mean():.1f} APG")
    
    print("\n" + "="*60)
    print("RECENT GAMES")
    print("="*60)
    print(df[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST']].tail(10).to_string(index=False))


def main():
    userInputtedPlayer = input("Please type a players first and last name: ")
    playerID = find_player_id(userInputtedPlayer)

    if playerID is None:
        print(f"Player '{userInputtedPlayer}' not found. Check the spelling.")
        return
    
    raw_stats_dataframe = get_player_stats(playerID)
    if raw_stats_dataframe is None:
        print("Failed to get player stats.")
        return

    
    cleaned_stats_dataframe = isolate_key_stats(raw_stats_dataframe)
    calculate_averages(cleaned_stats_dataframe, userInputtedPlayer)

    print("\n", cleaned_stats_dataframe)
    

    
main()

