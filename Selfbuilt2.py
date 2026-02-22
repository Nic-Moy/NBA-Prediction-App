import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from nba_api.stats.static import players
from sklearn.linear_model import LogisticRegression, LinearRegression
import time

# ==================== NBA API FUNCTIONS ====================

def find_player_id(name: str):
    hits = players.find_players_by_full_name(name)
    return hits[0]["id"] if hits else None


def get_player_stats(player_id, season='2024-25'):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return gamelog.get_data_frames()[0]
    except:
        return None


def get_team_defensive_stats(season='2024-25'):
    """Fetch defensive stats for all teams - used for opponent context."""
    try:
        time.sleep(0.6)  # Rate limiting for NBA API
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Base'
        )
        df = stats.get_data_frames()[0]
        
        team_stats = {}
        for _, row in df.iterrows():
            team_stats[row['TEAM_ABBREVIATION']] = {
                'opp_pts_allowed': row.get('PTS', 110),  # Points they allow
                'pace': row.get('PACE', 100) if 'PACE' in row else 100
            }
        return team_stats
    except Exception as e:
        print(f"Warning: Could not fetch team stats: {e}")
        return {}


def isolate_key_stats(df):
    """Clean and prepare player game log data."""
    key_stats = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT']
    cleaned_df = df[key_stats].copy()
    
    cleaned_df['GAME_DATE'] = pd.to_datetime(cleaned_df['GAME_DATE'])
    cleaned_df = cleaned_df.sort_values('GAME_DATE')
    
    # Convert MIN from "32:45" to 32.75
    cleaned_df['MIN'] = cleaned_df['MIN'].apply(
        lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 if ':' in str(x) else float(x)
    )
    
    # Create combined stat (Points + Rebounds + Assists)
    cleaned_df['PRA'] = cleaned_df['PTS'] + cleaned_df['REB'] + cleaned_df['AST']
    
    return cleaned_df


# ==================== FEATURE ENGINEERING ====================

def extract_opponent(matchup):
    """Extract opponent team abbreviation from matchup string."""
    if ' vs. ' in matchup:
        return matchup.split(' vs. ')[1]
    elif ' @ ' in matchup:
        return matchup.split(' @ ')[1]
    return None


def engineer_features(df, team_defensive_stats=None):
    """
    Add predictive features knowable BEFORE the game.
    Includes opponent defensive context if available.
    """
    df = df.copy()
    
    # Home/Away
    df['is_home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    # Rest days
    df['rest_days'] = df['GAME_DATE'].diff().dt.days.fillna(3)
    df['back_to_back'] = (df['rest_days'] <= 1).astype(int)
    
    # Recent performance (lagged - knowable before game)
    for stat in ['PTS', 'REB', 'AST', 'PRA']:
        df[f'{stat.lower()}_last_3'] = df[stat].shift(1).rolling(window=3, min_periods=1).mean()
        df[f'{stat.lower()}_last_5'] = df[stat].shift(1).rolling(window=5, min_periods=1).mean()
    
    # Minutes trend
    df['minutes_last_3'] = df['MIN'].shift(1).rolling(window=3, min_periods=1).mean()
    
    # Shooting form
    df['fg_pct_last_3'] = df['FG_PCT'].shift(1).rolling(window=3, min_periods=1).mean()
    df['fg3_pct_last_3'] = df['FG3_PCT'].shift(1).rolling(window=3, min_periods=1).mean()
    
    # Hot/cold trend
    last_3 = df['PTS'].shift(1).rolling(window=3, min_periods=1).mean()
    prev_3 = df['PTS'].shift(4).rolling(window=3, min_periods=1).mean()
    df['pts_trend'] = (last_3 - prev_3).fillna(0)
    
    # Win streak
    df['win'] = (df['WL'] == 'W').astype(int)
    streaks = []
    streak = 0
    for w in df['win'].values:
        streak = streak + 1 if w == 1 and streak >= 0 else (1 if w == 1 else (streak - 1 if streak <= 0 else -1))
        streaks.append(streak)
    df['win_streak'] = streaks
    
    # Opponent defensive stats (new feature)
    df['opponent'] = df['MATCHUP'].apply(extract_opponent)
    if team_defensive_stats:
        df['opp_pts_allowed'] = df['opponent'].map(
            lambda x: team_defensive_stats.get(x, {}).get('opp_pts_allowed', 110)
        )
        df['opp_pace'] = df['opponent'].map(
            lambda x: team_defensive_stats.get(x, {}).get('pace', 100)
        )
    else:
        df['opp_pts_allowed'] = 110  # League average default
        df['opp_pace'] = 100
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    return df


# ==================== DISPLAY FUNCTIONS ====================

def display_player_summary(df, player_name):
    """Show basic stats summary."""
    print(f"\n{'='*60}")
    print(f"STATS SUMMARY: {player_name.upper()}")
    print(f"{'='*60}")
    
    last_5 = df.tail(5)
    
    print(f"\nGames Played: {len(df)}")
    print(f"\nSeason Averages:")
    print(f"  Points:   {df['PTS'].mean():.1f}")
    print(f"  Rebounds: {df['REB'].mean():.1f}")
    print(f"  Assists:  {df['AST'].mean():.1f}")
    print(f"  PRA:      {df['PRA'].mean():.1f}")
    
    print(f"\nLast 5 Games:")
    print(f"  Points:   {last_5['PTS'].mean():.1f}")
    print(f"  Rebounds: {last_5['REB'].mean():.1f}")
    print(f"  Assists:  {last_5['AST'].mean():.1f}")
    
    # Show variance for each stat (helps explain why some are easier to predict)
    print(f"\nVolatility (Std Dev):")
    print(f"  Points:   {df['PTS'].std():.1f}  (hardest to predict)")
    print(f"  Rebounds: {df['REB'].std():.1f}")
    print(f"  Assists:  {df['AST'].std():.1f}")
    print(f"  PRA:      {df['PRA'].std():.1f}")


# ==================== BACKTESTING FRAMEWORK ====================

def create_splits(df, window=15):
    """Create time-series train/test splits for backtesting."""
    if len(df) <= window:
        return []
    return [(df.iloc[i-window:i], df.iloc[i]) for i in range(window, len(df))]


def get_feature_columns(include_opponent=True):
    """Get list of features to use in models."""
    features = [
        'is_home', 'rest_days', 'back_to_back',
        'pts_last_3', 'pts_last_5', 'minutes_last_3',
        'fg_pct_last_3', 'fg3_pct_last_3', 'pts_trend', 'win_streak',
        'reb_last_3', 'ast_last_3', 'pra_last_3'
    ]
    if include_opponent:
        features.extend(['opp_pts_allowed', 'opp_pace'])
    return features


# ==================== CLASSIFICATION BACKTEST (OVER/UNDER) ====================

def classification_backtest(df, stat_column, window=15, include_opponent=True):
    """
    Backtest over/under predictions using logistic regression.
    Uses rolling average as proxy for betting line.
    
    Returns DataFrame with prediction results.
    """
    splits = create_splits(df, window)
    if not splits:
        return None
    
    feature_cols = get_feature_columns(include_opponent)
    results = []
    
    for train_df, test_row in splits:
        # Use rolling average as betting line proxy
        line = train_df[stat_column].mean()
        
        # Create binary target
        train_df = train_df.copy()
        train_df['target'] = (train_df[stat_column] > line).astype(int)
        
        # Prepare training data
        X_train = train_df[feature_cols].dropna()
        y_train = train_df.loc[X_train.index, 'target']
        
        # Skip if not enough data or only one class
        if len(X_train) < 5 or y_train.nunique() < 2:
            continue
        
        # Train logistic regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predict
        X_test = test_row[feature_cols].values.reshape(1, -1)
        if np.isnan(X_test).any():
            continue
        
        pred_proba = model.predict_proba(X_test)[0]
        pred_class = model.predict(X_test)[0]
        actual_class = 1 if test_row[stat_column] > line else 0
        
        results.append({
            'date': test_row['GAME_DATE'],
            'stat': stat_column,
            'line': line,
            'actual_value': test_row[stat_column],
            'predicted': pred_class,
            'actual': actual_class,
            'confidence': max(pred_proba),
            'correct': pred_class == actual_class
        })
    
    return pd.DataFrame(results)


# ==================== ENSEMBLE BACKTEST ====================

def ensemble_backtest(df, stat_column, window=15):
    """
    Combine multiple methods for prediction:
    1. Recent 5-game average vs line
    2. Logistic regression with features
    3. Weighted recent games (exponential)
    
    Returns DataFrame with comparison of all methods.
    """
    splits = create_splits(df, window)
    if not splits:
        return None
    
    feature_cols = get_feature_columns(include_opponent=True)
    results = []
    
    for train_df, test_row in splits:
        line = train_df[stat_column].mean()
        actual_class = 1 if test_row[stat_column] > line else 0
        
        # Method 1: Recent 5-game average
        recent_avg = train_df[stat_column].tail(5).mean()
        method1_pred = 1 if recent_avg > line else 0
        
        # Method 2: Logistic regression
        train_copy = train_df.copy()
        train_copy['target'] = (train_copy[stat_column] > line).astype(int)
        X_train = train_copy[feature_cols].dropna()
        y_train = train_copy.loc[X_train.index, 'target']
        
        if len(X_train) < 5 or y_train.nunique() < 2:
            continue
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        X_test = test_row[feature_cols].values.reshape(1, -1)
        if np.isnan(X_test).any():
            continue
        
        method2_proba = model.predict_proba(X_test)[0][1]
        method2_pred = 1 if method2_proba > 0.5 else 0
        
        # Method 3: Exponentially weighted average
        weights = np.exp(np.linspace(0, 1, len(train_df)))
        weights = weights / weights.sum()
        weighted_avg = np.average(train_df[stat_column], weights=weights)
        method3_pred = 1 if weighted_avg > line else 0
        
        # Ensemble: majority vote
        votes = [method1_pred, method2_pred, method3_pred]
        ensemble_pred = 1 if sum(votes) >= 2 else 0
        
        # Soft ensemble: probability averaging
        m1_conf = 0.5 + 0.1 * (1 if method1_pred else -1)
        m3_conf = 0.5 + 0.15 * (1 if method3_pred else -1)
        avg_proba = (m1_conf + method2_proba + m3_conf) / 3
        ensemble_soft_pred = 1 if avg_proba > 0.5 else 0
        
        results.append({
            'date': test_row['GAME_DATE'],
            'method1_correct': method1_pred == actual_class,
            'method2_correct': method2_pred == actual_class,
            'method3_correct': method3_pred == actual_class,
            'ensemble_vote_correct': ensemble_pred == actual_class,
            'ensemble_soft_correct': ensemble_soft_pred == actual_class,
            'confidence': avg_proba
        })
    
    return pd.DataFrame(results)


# ==================== EVALUATION FUNCTIONS ====================

def evaluate_classification(results_df, stat_name):
    """Evaluate classification performance with betting context."""
    if results_df is None or len(results_df) == 0:
        print(f"No results for {stat_name}")
        return None
    
    accuracy = results_df['correct'].mean()
    
    print(f"\n{'-'*50}")
    print(f"{stat_name} - OVER/UNDER CLASSIFICATION")
    print(f"{'-'*50}")
    print(f"Predictions: {len(results_df)}")
    print(f"Accuracy: {accuracy:.1%}")
    
    # High confidence predictions
    high_conf = results_df[results_df['confidence'] > 0.6]
    if len(high_conf) > 5:
        print(f"High Confidence (>60%): {high_conf['correct'].mean():.1%} ({len(high_conf)} bets)")
    
    # Betting context
    print(f"\nBreak-even needed: 52.4%")
    if accuracy > 0.524:
        print(f"✅ Beats the vig!")
    else:
        print(f"❌ Below break-even")
    
    return accuracy


def evaluate_ensemble(results_df, stat_name):
    """Compare all ensemble methods."""
    if results_df is None or len(results_df) == 0:
        return
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE COMPARISON: {stat_name}")
    print(f"{'='*60}")
    print(f"Method 1 (Recent 5 avg):    {results_df['method1_correct'].mean():.1%}")
    print(f"Method 2 (Logistic Reg):    {results_df['method2_correct'].mean():.1%}")
    print(f"Method 3 (Weighted Recent): {results_df['method3_correct'].mean():.1%}")
    print(f"Ensemble (Majority Vote):   {results_df['ensemble_vote_correct'].mean():.1%}")
    print(f"Ensemble (Soft Average):    {results_df['ensemble_soft_correct'].mean():.1%}")
    
    # Best method
    methods = {
        'Recent 5 avg': results_df['method1_correct'].mean(),
        'Logistic Reg': results_df['method2_correct'].mean(),
        'Weighted Recent': results_df['method3_correct'].mean(),
        'Ensemble Vote': results_df['ensemble_vote_correct'].mean(),
        'Ensemble Soft': results_df['ensemble_soft_correct'].mean()
    }
    best = max(methods, key=methods.get)
    print(f"\n✅ Best method: {best} ({methods[best]:.1%})")


def compare_stat_difficulty(df, window=15):
    """Compare prediction difficulty across different stats."""
    print(f"\n{'='*60}")
    print("STAT COMPARISON: WHICH IS EASIEST TO PREDICT?")
    print(f"{'='*60}")
    
    stats = ['PTS', 'REB', 'AST', 'PRA']
    results = {}
    
    for stat in stats:
        result_df = classification_backtest(df, stat, window=window)
        if result_df is not None and len(result_df) > 0:
            acc = result_df['correct'].mean()
            results[stat] = acc
            volatility = df[stat].std()
            print(f"{stat:4s}: {acc:.1%} accuracy | Volatility: {volatility:.1f}")
    
    if results:
        best_stat = max(results, key=results.get)
        print(f"\n✅ Easiest to predict: {best_stat} ({results[best_stat]:.1%})")
        print(f"   Consider focusing on {best_stat} props for betting")
    
    return results


# ==================== MAIN FUNCTION ====================

def main():
    # Get player
    playerID = None
    while playerID is None:
        player_name = input("Enter player name: ")
        playerID = find_player_id(player_name)
        if playerID is None:
            print(f"'{player_name}' not found. Try again.")
    
    print(f"\n🔄 Fetching data for {player_name}...")
    
    # Fetch player stats
    raw_df = get_player_stats(playerID)
    if raw_df is None:
        print("Failed to fetch player stats.")
        return
    
    cleaned_df = isolate_key_stats(raw_df)
    
    # Fetch team defensive stats (for opponent features)
    print("🔄 Fetching team defensive stats...")
    team_stats = get_team_defensive_stats()
    
    # Engineer features
    print("🔧 Engineering features...")
    df = engineer_features(cleaned_df, team_stats)
    
    # Display summary
    display_player_summary(df, player_name)
    
    # Step 1: Compare stat difficulty
    print("\n" + "="*60)
    print("STEP 1: COMPARING STAT PREDICTION DIFFICULTY")
    print("="*60)
    stat_results = compare_stat_difficulty(df, window=15)
    
    # Step 2: Classification backtest on each stat
    print("\n" + "="*60)
    print("STEP 2: CLASSIFICATION BACKTEST (OVER/UNDER)")
    print("="*60)
    
    for stat in ['PTS', 'REB', 'AST', 'PRA']:
        results = classification_backtest(df, stat, window=15)
        evaluate_classification(results, stat)
    
    # Step 3: Ensemble comparison on best stat
    print("\n" + "="*60)
    print("STEP 3: ENSEMBLE METHOD COMPARISON")
    print("="*60)
    
    if stat_results:
        best_stat = max(stat_results, key=stat_results.get)
        print(f"Running ensemble on {best_stat} (easiest to predict)...")
        ensemble_results = ensemble_backtest(df, best_stat, window=15)
        evaluate_ensemble(ensemble_results, best_stat)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    print("1. Lower volatility stats (REB, AST) are often easier to predict")
    print("2. Classification (over/under) is more practical than exact prediction")
    print("3. Ensemble methods can improve accuracy by combining signals")
    print("4. Need >52.4% accuracy to profit with standard -110 odds")
    print("5. Focus on high-confidence predictions for actual betting")


main()
