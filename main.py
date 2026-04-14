from getplayerinfo import find_player_id, clean_player_games, DEFAULT_SEASON
from database import load_cached_logs
from model import (
    build_features,
    walk_forward_backtest,
    walk_forward_regression,
    print_player_summary,
    print_feature_preview,
    evaluate_results,
    evaluate_regression_results,
    predict_next_game,
    DEFAULT_LOOKBACK_GAMES,
)
from getprizepicks import get_nba_props

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

    # --- Regression backtest (predicts actual stat value, no proxy line needed) ---
    reg_df = walk_forward_regression(feature_df, stat_col="PTS", lookback_games=DEFAULT_LOOKBACK_GAMES)
    evaluate_regression_results(reg_df, stat_col="PTS")

    # --- Next-game prediction vs real PrizePicks line ---
    print(f"\n{'=' * 60}")
    print("NEXT GAME PREDICTION")
    print(f"{'=' * 60}")

    prediction = predict_next_game(feature_df, stat_col="PTS", lookback_games=DEFAULT_LOOKBACK_GAMES)

    if prediction["predicted"] is None:
        print("Not enough data to predict next game.")
    else:
        pred_pts = prediction["predicted"]
        moe = prediction["margin_of_error"]
        print(f"Model prediction:  {pred_pts} pts  (± {moe})")

        # Fetch today's real PrizePicks line for this player
        try:
            props_df = get_nba_props(stats=["pts"], player=player_name)
            standard = props_df[props_df["tier"] == "standard"]

            if standard.empty:
                print("PrizePicks line:   No line found for this player today.")
                print("(Player may not have a game today, or the line isn't posted yet.)")
            else:
                line = float(standard["line"].iloc[0])
                margin = pred_pts - line
                direction = "OVER" if margin > 0 else "UNDER"
                print(f"PrizePicks line:   {line} pts")
                print(f"Recommendation:    {direction}  ({margin:+.1f} margin)")
                if abs(margin) < 2.0:
                    print("Confidence:        Low  (margin < 2 pts — close call)")
                elif abs(margin) < 4.0:
                    print("Confidence:        Moderate")
                else:
                    print("Confidence:        High  (margin > 4 pts)")
        except Exception as e:
            print(f"Could not fetch PrizePicks line: {e}")
