import streamlit as st
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import playergamelog, leaguedashptteamdefend
from nba_api.stats.static import players, teams
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# ------------------ Configuration ------------------
API_KEY = st.secrets["ODDS_API_KEY"]  # Store secrets properly
CACHE_EXPIRY = timedelta(minutes=15)


# ------------------ Advanced Sportsbook API Integration ------------------
def fetch_odds(player_name):
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={API_KEY}&regions=us&markets=player_points&oddsFormat=american"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        odds_data = response.json()

        player_odds = {'over': {}, 'under': {}}
        for bookmaker in odds_data:
            for market in bookmaker['markets']:
                if market['key'] == "player_points":
                    for outcome in market['outcomes']:
                        if player_name.lower() in outcome['name'].lower():
                            key = 'over' if 'over' in outcome['name'].lower() else 'under'
                            player_odds[key][bookmaker['title']] = {
                                'line': outcome['point'],
                                'price': outcome['price'],
                                'last_updated': datetime.fromisoformat(market['last_update'][:-1])
                            }
        return player_odds
    except Exception as e:
        st.error(f"Odds API Error: {str(e)}")
        return {}


# ------------------ Enhanced Data Pipeline ------------------
@st.cache_data(ttl=CACHE_EXPIRY)
def get_player_stats(player_id, season='2023-24'):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return df.sort_values('GAME_DATE').reset_index(drop=True)
    except Exception as e:
        st.error(f"Stats Error: {str(e)}")
        return None


@st.cache_data(ttl=timedelta(days=1))
def get_team_defense_stats():
    try:
        defense_stats = leaguedashptteamdefend.LeagueDashPTTeamDefend().get_data_frames()[0]
        return defense_stats[['TEAM_NAME', 'DEF_RATING', 'DREB', 'STL', 'BLK']]
    except Exception as e:
        st.error(f"Defense Stats Error: {str(e)}")
        return None


# ------------------ Advanced Feature Engineering ------------------
def create_features(stats_df, defense_df):
    # Rolling features
    stats_df['3GAME_PTS_AVG'] = stats_df['PTS'].rolling(3).mean()
    stats_df['5GAME_MIN_AVG'] = stats_df['MIN'].rolling(5).mean()

    # Opponent defense features
    stats_df = pd.merge(stats_df, defense_df, left_on='MATCHUP', right_on='TEAM_NAME', how='left')

    # Time between games
    stats_df['DAYS_REST'] = stats_df['GAME_DATE'].diff().dt.days.fillna(3)

    # Home/Away
    stats_df['HOME_GAME'] = stats_df['MATCHUP'].apply(lambda x: 'vs.' in x).astype(int)

    return stats_df[['MIN', 'REB', 'AST', '3GAME_PTS_AVG', '5GAME_MIN_AVG',
                     'DEF_RATING', 'DAYS_REST', 'HOME_GAME']].dropna()


# ------------------ Machine Learning Pipeline ------------------
def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        'mse': mean_squared_error(y_test, preds),
        'r2': r2_score(y_test, preds),
        'feature_importances': dict(zip(X_test.columns, model.feature_importances_))
    }


# ------------------ Bankroll Management ------------------
def kelly_criterion(prob, decimal_odds):
    return (prob * decimal_odds - 1) / (decimal_odds - 1)


# ------------------ Injury Monitoring ------------------
@st.cache_data(ttl=timedelta(minutes=30))
def check_injuries(player_name):
    try:
        url = f"https://www.rotowire.com/basketball/search.php?search={player_name.replace(' ', '+')}"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        injury_div = soup.find('div', class_='news-update__report')
        return injury_div.get_text(strip=True) if injury_div else "No recent injury news"
    except Exception as e:
        return "Injury check unavailable"


# ------------------ Streamlit Interface ------------------
st.title("🏀 NBA Prop Predictor Pro")
st.markdown("⚠️ Warning: Sports betting involves risk. Only gamble with funds you can afford to lose.")

with st.sidebar:
    st.header("Settings")
    bankroll = st.number_input("Bankroll ($)", min_value=0, value=1000)
    risk_tolerance = st.select_slider("Risk Tolerance", options=["Low", "Medium", "High"], value="Medium")

player_name = st.text_input("Search Player:", "LeBron James")

# Player ID resolution
player_info = players.find_players_by_full_name(player_name)
if not player_info:
    st.error("Player not found")
    st.stop()

player_id = player_info[0]['id']

# Data loading
with st.spinner("Loading data..."):
    stats_df = get_player_stats(player_id)
    defense_df = get_team_defense_stats()

    if stats_df is None or defense_df is None:
        st.error("Failed to load required data")
        st.stop()

# Feature engineering
enhanced_df = create_features(stats_df, defense_df)
if enhanced_df.empty:
    st.error("Insufficient data for analysis")
    st.stop()

# Model training
X = enhanced_df.drop(columns=['3GAME_PTS_AVG'])  # Avoid target leakage
y = enhanced_df['3GAME_PTS_AVG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(X_train, y_train)
eval_results = evaluate_model(model, X_test, y_test)

# Latest game features
latest_features = X.iloc[-1].values.reshape(1, -1)
prediction = model.predict(latest_features)[0]

# Odds analysis
odds_data = fetch_odds(player_name)
best_over = max(odds_data.get('over', {}).items(), key=lambda x: x[1]['price'], default=None)

# Interface
col1, col2 = st.columns(2)
with col1:
    st.subheader("Player Insights")
    st.metric("Predicted Points", f"{prediction:.1f}")
    st.write(f"**Model Accuracy (R²):** {eval_results['r2']:.2%}")

    st.subheader("Injury Status")
    st.write(check_injuries(player_name))

with col2:
    st.subheader("Betting Analysis")
    if best_over:
        decimal_odds = best_over[1]['price'] / 100 + 1 if best_over[1]['price'] > 0 else 1 - 100 / best_over[1]['price']
        ev = (prediction - best_over[1]['line']) * decimal_odds
        kelly = kelly_criterion(prediction / best_over[1]['line'], decimal_odds)

        st.write(f"**Best Over Odds:** {best_over[1]['line']} @ {best_over[1]['price']} ({best_over[0]})")
        st.write(f"**Expected Value:** ${ev:.2f}")
        st.write(f"**Kelly Criterion:** {kelly:.1%} of bankroll")
        st.progress(min(max(kelly, 0), 1))
    else:
        st.write("No odds available")

st.subheader("Model Insights")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(y_test, model.predict(X_test))
ax[0].set_title("Actual vs Predicted")
ax[1].barh(list(eval_results['feature_importances'].keys()), list(eval_results['feature_importances'].values()))
ax[1].set_title("Feature Importance")
st.pyplot(fig)

st.subheader("Historical Performance")
st.line_chart(stats_df.set_index('GAME_DATE')['PTS'])