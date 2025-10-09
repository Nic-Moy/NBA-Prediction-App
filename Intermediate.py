import streamlit as st
import requests
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# ------------------ Configuration ------------------
API_KEY = "ODDS_API_KEY"  # Replace with your actual API key
CACHE_EXPIRY = 60 * 15  # 15 minutes cache


# ------------------ Sportsbook API Integration ------------------
def fetch_odds(player_name):
    """
    Fetch betting odds for a player from a sportsbook API.
    """
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={API_KEY}&regions=us&markets=player_points&oddsFormat=american"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        odds_data = response.json()

        player_odds = {}
        for game in odds_data:
            for bookmaker in game['bookmakers']:
                for market in bookmaker['markets']:
                    if market['key'] == "player_points":
                        for outcome in market['outcomes']:
                            if player_name.lower() in outcome['description'].lower():
                                player_odds[bookmaker['title']] = {
                                    'line': outcome['point'],
                                    'price': outcome['price']
                                }
        return player_odds
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return {}
    except Exception as e:
        st.error(f"Error fetching odds: {str(e)}")
        return {}


# ------------------ NBA Stats Integration ------------------
@st.cache_data(ttl=CACHE_EXPIRY)
def get_player_stats(player_id, season='2023-24'):
    """
    Fetch player game logs using the NBA API.
    """
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return df.sort_values('GAME_DATE', ascending=False)
    except Exception as e:
        st.error(f"Error fetching player stats: {str(e)}")
        return None


# ------------------ Expected Value Calculation ------------------
def calculate_ev(predicted_points, line, odds):
    """
    Calculate expected value for a bet.
    """
    if odds > 0:
        decimal_odds = 1 + (odds / 100)
    else:
        decimal_odds = 1 - (100 / odds)

    probability = 1 / decimal_odds
    ev = (predicted_points - line) * decimal_odds
    return ev


# ------------------ Streamlit Interface ------------------
st.title("🏀 NBA Prop Betting Tool")
st.markdown("⚠️ Warning: Sports betting involves risk. Only gamble with funds you can afford to lose.")

# Player search
player_name = st.text_input("Enter Player Name:", "LeBron James")

# Resolve player ID
player_info = players.find_players_by_full_name(player_name)
if not player_info:
    st.error("Player not found. Try entering a valid player name.")
    st.stop()

player_id = player_info[0]['id']

# Fetch player stats
stats_df = get_player_stats(player_id)
if stats_df is None:
    st.error("Failed to fetch player stats.")
    st.stop()

# Display recent stats
st.subheader(f"Last 5 Games for {player_name}")
st.write(stats_df[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'MIN']].head())

# Calculate rolling average for points
stats_df['3GAME_PTS_AVG'] = stats_df['PTS'].rolling(3).mean()
predicted_points = stats_df['3GAME_PTS_AVG'].iloc[0]

# Display predicted points
st.subheader("Predicted Performance")
st.write(f"**3-Game Rolling Average Points:** {predicted_points:.1f}")

# Fetch betting odds
odds_data = fetch_odds(player_name)
if not odds_data:
    st.warning("No betting odds available for this player.")
    st.stop()

# Display odds
st.subheader("Available Betting Lines")
for bookmaker, odds in odds_data.items():
    st.write(f"**{bookmaker}:** {odds['line']} points @ {odds['price']}")

# Calculate expected value for each line
st.subheader("Expected Value Analysis")
for bookmaker, odds in odds_data.items():
    line = odds['line']
    price = odds['price']
    ev = calculate_ev(predicted_points, line, price)
    st.write(f"**{bookmaker} ({line} points):** EV = {ev:.2f}")

# Best line recommendation
best_line = min(odds_data.items(), key=lambda x: abs(x[1]['line'] - predicted_points))
st.subheader("Best Line Recommendation")
st.write(f"**{best_line[0]}:** {best_line[1]['line']} points @ {best_line[1]['price']}")

# Historical performance chart
st.subheader("Historical Performance")
st.line_chart(stats_df.set_index('GAME_DATE')['PTS'])