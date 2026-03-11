from getplayerinfo import find_player_id, clean_player_games, DEFAULT_SEASON
from database import load_cached_logs
from model import (
    build_features,
    walk_forward_backtest,
    print_player_summary,
    print_feature_preview,
    evaluate_results,
    DEFAULT_LOOKBACK_GAMES,
)

if __name__ == "__main__":
    print("NBA Player Props — Model")

    # Resolve player with validation loop
    player_id = None
    player_name = ""
    while player_id is None:
        player_name = input("Enter player name (ex: Stephen Curry): ").strip()
        if not player_name:
            print("Player name cannot be empty.")
            continue
        player_id = find_player_id(player_name)
        if player_id is None:
            print(f"Player '{player_name}' not found. Check spelling and try again.")

    season_input = input(f"Enter season [{DEFAULT_SEASON}]: ").strip()
    season = season_input or DEFAULT_SEASON

    raw_df = load_cached_logs(player_id, season)
    if raw_df.empty:
        print(f"\nNo cached game logs for {player_name} ({season}).")
        print("Run setup.py first to load game logs into the database.")
        raise SystemExit(1)

    clean_df = clean_player_games(raw_df)
    feature_df = build_features(clean_df)

    print_player_summary(clean_df, player_name)
    print_feature_preview(feature_df)

    results_df = walk_forward_backtest(feature_df, lookback_games=DEFAULT_LOOKBACK_GAMES)
    evaluate_results(results_df)
