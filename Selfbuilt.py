import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from nba_api.stats.static import players
from LinearRegression import predict_next_stat

ODDS_API_KEY =  "41d21edd04f753851b4dfd7421a40341"

# /////////////////  Accessing NBA api functions  ///////////////////////////////////////
def find_player_id(name: str):
    hits = players.find_players_by_full_name(name)
    if hits:
        return hits[0]["id"]
    else:
        None


def get_player_stats(player_id, season='2025-26'):
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


def calculate_and_display_averages(df, player_name):
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



# /////////////////////////  Exploratory Analysis of data  ////////////////////////////////
def analyze_correlations(df):
    """Analyze correlations between key stats to identify predictive relationships."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numeric columns for correlation
    numeric_cols = ['MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT']
    correlation_matrix = df[numeric_cols].corr(method='spearman')
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3))
    
    # Identify strong predictors for PTS
    print("\n" + "-"*60)
    print("STRONG PREDICTORS FOR POINTS (|correlation| > 0.3):")
    print("-"*60)
    pts_correlations = correlation_matrix['PTS'].drop('PTS').sort_values(ascending=False)
    for stat, corr in pts_correlations.items():
        if abs(corr) > 0.3:
            direction = "↑ Positive" if corr > 0 else "↓ Negative"
            print(f"  {stat:8s}: {corr:+.3f}  {direction}")
    
    # Similar analysis for REB and AST
    print("\nSTRONG PREDICTORS FOR REBOUNDS (|correlation| > 0.3):")
    reb_correlations = correlation_matrix['REB'].drop('REB').sort_values(ascending=False)
    for stat, corr in reb_correlations.items():
        if abs(corr) > 0.3:
            direction = "↑ Positive" if corr > 0 else "↓ Negative"
            print(f"  {stat:8s}: {corr:+.3f}  {direction}")
    
    print("\nSTRONG PREDICTORS FOR ASSISTS (|correlation| > 0.3):")
    ast_correlations = correlation_matrix['AST'].drop('AST').sort_values(ascending=False)
    for stat, corr in ast_correlations.items():
        if abs(corr) > 0.3:
            direction = "↑ Positive" if corr > 0 else "↓ Negative"
            print(f"  {stat:8s}: {corr:+.3f}  {direction}")


def plot_relationships(df, player_name):
    """Create visualizations to understand patterns in player performance."""
    print("\n" + "="*60)
    print(f"VISUAL ANALYSIS FOR {player_name.upper()}")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{player_name} Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Points vs Minutes (scatter plot)
    axes[0, 0].scatter(df['MIN'], df['PTS'], alpha=0.6, color='blue', edgecolors='black')
    axes[0, 0].set_xlabel('Minutes Played', fontsize=10)
    axes[0, 0].set_ylabel('Points Scored', fontsize=10)
    axes[0, 0].set_title('Points vs Minutes (Correlation Check)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['MIN'], df['PTS'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['MIN'], p(df['MIN']), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    axes[0, 0].legend()
    
    # 2. Points over time (trend analysis)
    axes[0, 1].plot(df['GAME_DATE'], df['PTS'], marker='o', linestyle='-', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Game Date', fontsize=10)
    axes[0, 1].set_ylabel('Points', fontsize=10)
    axes[0, 1].set_title('Points Trend Over Season', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add rolling average line
    df_temp = df.copy()
    df_temp['rolling_avg'] = df_temp['PTS'].rolling(window=5, min_periods=1).mean()
    axes[0, 1].plot(df_temp['GAME_DATE'], df_temp['rolling_avg'], 'r--', linewidth=2, label='5-Game Avg')
    axes[0, 1].legend()
    
    # 3. Points distribution (histogram)
    axes[1, 0].hist(df['PTS'], bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(df['PTS'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["PTS"].mean():.1f}')
    axes[1, 0].axvline(df['PTS'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["PTS"].median():.1f}')
    axes[1, 0].set_xlabel('Points', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title('Points Distribution (Consistency Check)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Multi-stat comparison (recent form)
    recent_games = df.tail(10)
    x = range(len(recent_games))
    axes[1, 1].plot(x, recent_games['PTS'].values, marker='o', label='Points', linewidth=2)
    axes[1, 1].plot(x, recent_games['REB'].values * 3, marker='s', label='Rebounds (x3)', linewidth=2)
    axes[1, 1].plot(x, recent_games['AST'].values * 3, marker='^', label='Assists (x3)', linewidth=2)
    axes[1, 1].set_xlabel('Last 10 Games', fontsize=10)
    axes[1, 1].set_ylabel('Value', fontsize=10)
    axes[1, 1].set_title('Recent Performance Trends', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    print(f"\nKey Insights:")
    print(f"  - Points std deviation: {df['PTS'].std():.2f} (lower = more consistent)")
    print(f"  - MIN-PTS correlation: {df['MIN'].corr(df['PTS']):.3f}")
    print(f"  - Points range: {df['PTS'].min():.0f} to {df['PTS'].max():.0f}")


def compare_game_window_sizes(df):
    """Compare prediction accuracy of different rolling window sizes."""
    print("\n" + "="*60)
    print("ROLLING WINDOW COMPARISON")
    print("="*60)
    print("Testing which lookback period best predicts next game performance\n")
    
    if len(df) < 15:
        print("Not enough games to perform window comparison (need at least 15)")
        return
    
    # Test different window sizes
    windows = [3, 5, 7, 10]
    results = {}
    
    for window in windows:
        errors = []
        
        # For each game after the window period, predict using previous games
        for i in range(window, len(df)):
            # Get previous 'window' games
            lookback = df.iloc[i-window:i]
            
            # Simple prediction: average of previous games
            predicted_pts = lookback['PTS'].mean()
            
            # Actual points in this game
            actual_pts = df.iloc[i]['PTS']
            
            # Calculate error
            error = abs(predicted_pts - actual_pts)
            errors.append(error)
        
        # Calculate metrics for this window size
        mae = np.mean(errors)  # Mean Absolute Error
        results[window] = {
            'mae': mae,
            'predictions_made': len(errors),
            'errors': errors
        }
    
    # Display results
    print(f"{'Window':<10} {'MAE':<12} {'Predictions':<15}")
    print("-" * 40)
    for window in windows:
        mae = results[window]['mae']
        count = results[window]['predictions_made']
        print(f"{window}-game    {mae:>6.2f} pts   {count:>3} predictions")
    
    # Find best window
    best_window = min(results.keys(), key=lambda w: results[w]['mae'])
    print("\n" + "-"*60)
    print(f"BEST WINDOW: {best_window} games (MAE = {results[best_window]['mae']:.2f} pts)")
    print("-"*60)
    
    # Show insight
    improvement = ((results[max(windows)]['mae'] - results[best_window]['mae']) / results[max(windows)]['mae']) * 100
    if improvement > 5:
        print(f"\n💡 Using {best_window}-game window is {improvement:.1f}% more accurate than {max(windows)}-game window")
    else:
        print(f"\n💡 Window size doesn't significantly impact accuracy (variance < 5%)")
    
    return results


# //////////////////////////////  Validation Functions  ////////////////////////////////

def create_training_testing_split(df, window=5):
    """
    Split dataframe into training and testing sets for time-series validation.
    Uses walk-forward validation where each test point uses only prior data.
    
    Returns:
    - List of (train_df, test_row) tuples for each prediction
    """
    if len(df) < window + 5:
        print(f"Not enough data. Need at least {window + 5} games, have {len(df)}")
        return []
    
    splits = []
    
    # Start predictions after 'window' games
    for i in range(window, len(df)):
        # Training data: previous 'window' games
        train_df = df.iloc[i-window:i].copy()
        
        # Test data: current game (what we're trying to predict)
        test_row = df.iloc[i]
        
        splits.append((train_df, test_row))
    
    return splits

def rolling_avg_prediction(df, window=5):
    """
    Predict next game points using simple rolling average (baseline method).
    
    Returns:
    - predictions: List of predicted points
    - actuals: List of actual points
    - errors: List of absolute errors
    """
    splits = create_training_testing_split(df, window)
    
    if not splits:
        return None, None, None
    
    predictions = []
    actuals = []
    errors = []
    
    for train_df, test_row in splits:
        # Baseline prediction: simple average of training window
        predicted_pts = train_df['PTS'].mean()
        actual_pts = test_row['PTS']
        error = abs(predicted_pts - actual_pts)
        
        predictions.append(predicted_pts)
        actuals.append(actual_pts)
        errors.append(error)
    
    return predictions, actuals, errors

def regression_prediction(df, window=10, model_type='linear'):
    """
    Predict next game points using regression model with multiple features.
    
    Returns:
    - predictions: List of predicted points
    - actuals: List of actual points
    - errors: List of absolute errors
    """
    
    splits = create_training_testing_split(df, window)
    
    if not splits:
        return None, None, None
    
    predictions = []
    actuals = []
    errors = []
    
    # Define features to use for prediction
    feature_columns = ['MIN', 'FG_PCT', 'FG3_PCT', 'AST']
    
    for train_df, test_row in splits:
        try:
            # Use LinearRegression module to predict points
            predicted_pts, model = predict_next_stat(
                train_df,
                feature_columns=feature_columns,
                target_column='PTS',
                model_type=model_type,
                alpha=1.0,
                window=None  # Already using windowed data from splits
            )
            
            actual_pts = test_row['PTS']
            error = abs(predicted_pts - actual_pts)
            
            predictions.append(predicted_pts)
            actuals.append(actual_pts)
            errors.append(error)
            
        except (ValueError, KeyError) as e:
            # Skip predictions where data is insufficient or features missing
            continue
    
    return predictions, actuals, errors

def evaluate_predictions(actual, prediction, method_name="Model"):
    """
    Evaluate prediction accuracy using multiple metrics.
    
    Parameters:
    - actual: List of actual point values
    - prediction: List of predicted point values
    - method_name: Name of prediction method (for display)
    
    Returns:
    - Dictionary of evaluation metrics
    """
    if not actual or not prediction or len(actual) != len(prediction):
        print(f"Cannot evaluate {method_name}: Invalid data")
        return None
    
    actual = np.array(actual)
    prediction = np.array(prediction)
    
    # Calculate metrics
    errors = np.abs(actual - prediction)
    mae = np.mean(errors)  # Mean Absolute Error
    rmse = np.sqrt(np.mean((actual - prediction) ** 2))  # Root Mean Squared Error
    
    # Accuracy within certain thresholds
    within_3pts = np.sum(errors <= 3) / len(errors) * 100
    within_5pts = np.sum(errors <= 5) / len(errors) * 100
    within_7pts = np.sum(errors <= 7) / len(errors) * 100
    
    # Average actual vs predicted
    avg_actual = np.mean(actual)
    avg_predicted = np.mean(prediction)
    bias = avg_predicted - avg_actual
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'within_3pts': within_3pts,
        'within_5pts': within_5pts,
        'within_7pts': within_7pts,
        'avg_actual': avg_actual,
        'avg_predicted': avg_predicted,
        'bias': bias,
        'n_predictions': len(actual)
    }
    
    # Display results
    print(f"\n{'='*60}")
    print(f"{method_name.upper()} EVALUATION")
    print(f"{'='*60}")
    print(f"Predictions Made: {metrics['n_predictions']}")
    print(f"\nAccuracy Metrics:")
    print(f"  Mean Absolute Error (MAE):  {mae:.2f} points")
    print(f"  Root Mean Squared Error:    {rmse:.2f} points")
    print(f"\nPrediction Accuracy:")
    print(f"  Within ±3 points: {within_3pts:.1f}%")
    print(f"  Within ±5 points: {within_5pts:.1f}%")
    print(f"  Within ±7 points: {within_7pts:.1f}%")
    print(f"\nBias Analysis:")
    print(f"  Average Actual Points:    {avg_actual:.1f}")
    print(f"  Average Predicted Points: {avg_predicted:.1f}")
    print(f"  Bias (over/under):        {bias:+.1f} points")
    
    if abs(bias) > 2:
        direction = "overestimating" if bias > 0 else "underestimating"
        print(f"  ⚠️  Model is {direction} by {abs(bias):.1f} points on average")
    
    return metrics


# ////////////////////////////  Main Function  ////////////////////////////////
def main():
    playerID, raw_stats_dataframe = None , None
    
    # Getting player stats
    while playerID is None:
        userInputtedPlayer = input("Please type a players first and last name: ")
        playerID = find_player_id(userInputtedPlayer)
        if playerID == None:
            print(f"Player '{userInputtedPlayer}' not found. Check the spelling.")
        
    
   
    raw_stats_dataframe = get_player_stats(playerID)
    cleaned_stats_dataframe = isolate_key_stats(raw_stats_dataframe)

    # Basic mean averages
    calculate_and_display_averages(cleaned_stats_dataframe, userInputtedPlayer)
    print()

    # New exploratory analysis functions
    analyze_correlations(cleaned_stats_dataframe)
    print()
    plot_relationships(cleaned_stats_dataframe, userInputtedPlayer)
    print()
    compare_game_window_sizes(cleaned_stats_dataframe)

    # New Validation functions
    # Baseline: Rolling Average
    print("\n" + "="*60)
    print("VALIDATION: BASELINE VS ML MODEL")
    print("="*60)

    # pred_avg, actual_avg, err_avg = rolling_avg_prediction(cleaned_stats_dataframe, window=7)
    # metrics_baseline = evaluate_predictions(actual_avg, pred_avg, "Rolling Average (7-game)")

    # # ML Model: Regression
    # pred_ml, actual_ml, err_ml = regression_prediction(cleaned_stats_dataframe, window=7, model_type='linear')
    # metrics_ml = evaluate_predictions(actual_ml, pred_ml, "Linear Regression (10-game)")

    # # Comparison
    # if metrics_baseline and metrics_ml:
    #     improvement = ((metrics_baseline['mae'] - metrics_ml['mae']) / metrics_baseline['mae']) * 100
    #     print(f"\n{'='*60}")
    #     print(f"ML IMPROVEMENT: {improvement:+.1f}%")
    #     if improvement > 0:
    #         print(f"✅ ML model is {improvement:.1f}% more accurate!")
    #     else:
    #         print(f"❌ ML model is {abs(improvement):.1f}% worse than baseline")
    #     print(f"{'='*60}")


    
main()
