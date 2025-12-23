import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from LinearRegression import forecast_player_stats

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

    # New functions
    analyze_correlations(cleaned_stats_dataframe)
    print()
    plot_relationships(cleaned_stats_dataframe, userInputtedPlayer)
    print()
    compare_game_window_sizes(cleaned_stats_dataframe)

    
main()
