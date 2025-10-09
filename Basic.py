import streamlit as st
import requests
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
#from sklearn.linear_model import LinearRegression

API_KEY = "ODDS_API_KEY"  # Replace with your actual API key
CACHE_EXPIRY = 60 * 15  # 15 minutes cache

# ------------------ Data Collection ------------------

# Function to get player stats from NBA API
def get_player_stats(player_id, season='2023-24'):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        return df
    except:
        return None

# Function to fetch Vegas odds (mocked example)
def get_vegas_lines(player_name):
    vegas_data = {
        "LeBron James": {"Points": 26.5, "Assists": 6.5, "Rebounds": 7.5},
        "Stephen Curry": {"Points": 29.5, "Assists": 5.5, "Rebounds": 4.5},
    }
    return vegas_data.get(player_name, {})

# ------------------ Expected Value (EV) Calculation ------------------

# Convert odds to implied probability
def implied_probability(odds):
    return 1 / odds if odds > 0 else -odds / (-odds + 100)

# Calculate expected value (EV)
def calculate_ev(probability, payout, wager=1):
    return (probability * payout) - ((1 - probability) * wager)

# ------------------ Line Shopping ------------------

# Mocked sportsbook odds comparison
def best_line(player_prop, odds_data):
    best_platform = min(odds_data, key=lambda x: odds_data[x].get(player_prop, float('inf')))
    best_odds = odds_data[best_platform].get(player_prop)
    return best_platform, best_odds

# ------------------ Predictive Modeling ------------------

# Train a basic regression model to predict points based on past games
# def train_model(data):
#     if data is None or data.empty:
#         return None
#     X = data[['MIN']].astype(float)
#     y = data['PTS'].astype(float)
#     model = LinearRegression()
#     model.fit(X, y)
#     return model

# # Predict player performance
# def predict_points(model, minutes):
#     return model.predict([[minutes]])[0] if model else "N/A"

# ------------------ Streamlit Interface ------------------

st.title("NBA Player Prop Betting Tool")

# User input
player_name = st.text_input("Enter Player Name:")
player_id_map = {"LeBron James": "2544", "Stephen Curry": "201939"}  # Add more players as needed

if player_name in player_id_map:
    player_id = player_id_map[player_name]
    stats_df = get_player_stats(player_id)

    if stats_df is not None:
        st.write(f"### Last 5 Games for {player_name}")
        st.write(stats_df[['GAME_DATE', 'PTS', 'REB', 'AST', 'MIN']].head())

        vegas_lines = get_vegas_lines(player_name)
        st.write(f"### Vegas Lines for {player_name}")
        st.write(vegas_lines)

        # Train model and predict
        #model = train_model(stats_df)
        #predicted_points = predict_points(model, stats_df.iloc[0]['MIN']) if model else "N/A"

        #st.write(f"### Predicted Points: {predicted_points}")

        # Line shopping
        odds_data = {
            "PrizePicks": {"Points": vegas_lines.get("Points", 0)},
            "SportsbookA": {"Points": vegas_lines.get("Points", 0) + 1},
            "SportsbookB": {"Points": vegas_lines.get("Points", 0) - 1},
        }
        best_platform, best_odds = best_line("Points", odds_data)
        st.write(f"### Best Line Found at {best_platform}: {best_odds}")

else:
    st.write("Player not found. Try entering 'LeBron James' or 'Stephen Curry'.")

