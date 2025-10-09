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
    

def main():
    userInputtedPlayer = input("Please type a players first and last name: ")
    playerID = find_player_id(userInputtedPlayer)
    stats_dataframe = get_player_stats(playerID)
    print(stats_dataframe)

    
main()

