"""
Learning version: NBA player prop classification (Over / Under points)

What this script does:
1. Fetch one player's game log from nba_api
2. Build only PRE-GAME features (things you know before tipoff)
3. Create a proxy betting line (previous 5-game average points)
4. Train a Logistic Regression model in a walk-forward backtest
5. Report accuracy vs betting break-even (52.4% for -110 odds)

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from getplayerinfo import (
    DEFAULT_SEASON as PLAYERINFO_DEFAULT_SEASON,
    clean_player_games,
    find_player_id,
    get_player_stats,
)


DEFAULT_SEASON = PLAYERINFO_DEFAULT_SEASON
DEFAULT_LOOKBACK_GAMES = 15
LINE_PROXY_WINDOW = 5

# Only include features we could know before the game starts.
FEATURE_COLUMNS = [
    "is_home",
    "rest_days",
    "pts_avg_3",
    "pts_avg_5",
    "reb_avg_3",
    "ast_avg_3",
    "min_avg_3",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-game features and labels.

    Important:
    - We use .shift(1) so features come from PRIOR games only.
    - That avoids data leakage.
    """
    df = df.copy()

    # Schedule-based features (known before the game)
    df["is_home"] = df["MATCHUP"].astype(str).str.contains("vs.", regex=False).astype(int)
    df["rest_days"] = df["GAME_DATE"].diff().dt.days
    df["rest_days"] = df["rest_days"].fillna(3).clip(lower=0)

    # Rolling averages from PRIOR games only (shifted)
    df["pts_avg_3"] = df["PTS"].shift(1).rolling(window=3, min_periods=1).mean()
    df["pts_avg_5"] = df["PTS"].shift(1).rolling(window=5, min_periods=1).mean()
    df["reb_avg_3"] = df["REB"].shift(1).rolling(window=3, min_periods=1).mean()
    df["ast_avg_3"] = df["AST"].shift(1).rolling(window=3, min_periods=1).mean()
    df["min_avg_3"] = df["MIN"].shift(1).rolling(window=3, min_periods=1).mean()

    # Learning-only proxy for a betting line:
    # "What if the line were the player's previous 5-game average points?"
    df["line_proxy_pts"] = df["PTS"].shift(1).rolling(
        window=LINE_PROXY_WINDOW,
        min_periods=LINE_PROXY_WINDOW,
    ).mean()

    # Classification label:
    # 1 = over the line, 0 = under (or equal)
    df["target_over_pts"] = np.nan
    valid_line = df["line_proxy_pts"].notna()
    df.loc[valid_line, "target_over_pts"] = (
        df.loc[valid_line, "PTS"] > df.loc[valid_line, "line_proxy_pts"]
    ).astype(int)

    return df


def walk_forward_backtest(
    df: pd.DataFrame,
    lookback_games: int = DEFAULT_LOOKBACK_GAMES,
) -> pd.DataFrame:
    """
    Walk-forward backtest for points over/under classification.

    For each test game:
    - train on the previous N games only
    - predict OVER/UNDER for the next game
    """
    results: list[dict] = []

    for i in range(lookback_games, len(df)):
        train_window = df.iloc[i - lookback_games : i].copy()
        test_row = df.iloc[i]

        # Skip if we don't have a proxy line for the test game yet
        if pd.isna(test_row["line_proxy_pts"]):
            continue

        # Prepare training data
        train_model_df = train_window[FEATURE_COLUMNS + ["target_over_pts"]].dropna()
        if len(train_model_df) < 8:
            continue

        X_train = train_model_df[FEATURE_COLUMNS]
        y_train = train_model_df["target_over_pts"].astype(int)

        # Logistic regression needs both classes in the training window
        if y_train.nunique() < 2:
            continue

        # Prepare test row
        X_test = test_row[FEATURE_COLUMNS]
        if X_test.isna().any():
            continue
        X_test = X_test.to_frame().T  # 1-row DataFrame

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        prob_over = float(model.predict_proba(X_test)[0][1])
        pred_over = int(prob_over >= 0.5)
        actual_over = int(test_row["PTS"] > test_row["line_proxy_pts"])

        results.append(
            {
                "game_date": test_row["GAME_DATE"],
                "matchup": test_row["MATCHUP"],
                "actual_pts": float(test_row["PTS"]),
                "line_proxy_pts": float(test_row["line_proxy_pts"]),
                "pred_prob_over": prob_over,
                "pred_over": pred_over,
                "actual_over": actual_over,
                "correct": pred_over == actual_over,
            }
        )

    return pd.DataFrame(results)


def print_player_summary(df: pd.DataFrame, player_name: str) -> None:
    """Display a small summary so you can inspect the raw data."""
    print(f"\n{'=' * 60}")
    print(f"PLAYER SUMMARY: {player_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Games loaded: {len(df)}")
    print(f"Average points: {df['PTS'].mean():.1f}")
    print(f"Average rebounds: {df['REB'].mean():.1f}")
    print(f"Average assists: {df['AST'].mean():.1f}")
    print("\nLast 5 games:")
    print(df[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "MIN"]].tail(5).to_string(index=False))


def print_feature_preview(df: pd.DataFrame) -> None:
    """Show a few engineered columns so you can learn what the model sees."""
    print(f"\n{'=' * 60}")
    print("FEATURE PREVIEW (what the model sees before a game)")
    print(f"{'=' * 60}")
    preview_cols = ["GAME_DATE", "MATCHUP"] + FEATURE_COLUMNS + ["line_proxy_pts", "target_over_pts"]
    print(df[preview_cols].tail(8).to_string(index=False))


def evaluate_results(results_df: pd.DataFrame) -> None:
    """Print simple metrics and a sample of predictions."""
    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 60}")

    if results_df.empty:
        print("No predictions were generated.")
        print("Try a player with more games or reduce the lookback window.")
        return

    accuracy = float(results_df["correct"].mean())
    n_predictions = len(results_df)
    avg_confidence = float(results_df["pred_prob_over"].apply(lambda p: max(p, 1 - p)).mean())

    print(f"Predictions made: {n_predictions}")
    print(f"Accuracy: {accuracy:.1%}")
    print("Break-even accuracy for -110 odds: 52.4%")
    if accuracy > 0.524:
        print("Status: ABOVE break-even (on this backtest)")
    else:
        print("Status: BELOW break-even (on this backtest)")
    print(f"Average confidence: {avg_confidence:.1%}")

    high_conf = results_df[results_df["pred_prob_over"].apply(lambda p: max(p, 1 - p)) >= 0.60]
    if not high_conf.empty:
        print(f"High-confidence picks (>= 60%): {len(high_conf)}")
        print(f"High-confidence accuracy: {high_conf['correct'].mean():.1%}")

    print(f"\nMost recent predictions (last 10):")
    display_cols = [
        "game_date",
        "matchup",
        "actual_pts",
        "line_proxy_pts",
        "pred_prob_over",
        "pred_over",
        "actual_over",
        "correct",
    ]
    recent = results_df[display_cols].tail(10).copy()
    recent["game_date"] = pd.to_datetime(recent["game_date"]).dt.date
    recent["pred_prob_over"] = recent["pred_prob_over"].round(3)
    recent["actual_pts"] = recent["actual_pts"].round(1)
    recent["line_proxy_pts"] = recent["line_proxy_pts"].round(1)
    print(recent.to_string(index=False))


def walk_forward_regression(
    df: pd.DataFrame,
    stat_col: str = "PTS",
    lookback_games: int = DEFAULT_LOOKBACK_GAMES,
) -> pd.DataFrame:
    """
    Walk-forward regression backtest: predict actual stat value for each game.

    For each test game, trains Ridge regression on the previous N games only,
    then predicts the actual stat value (e.g. points scored).

    stat_col should be an uppercase column name: "PTS", "REB", or "AST".
    """
    results: list[dict] = []

    for i in range(lookback_games, len(df)):
        train_window = df.iloc[i - lookback_games : i].copy()
        test_row = df.iloc[i]

        train_model_df = train_window[FEATURE_COLUMNS + [stat_col]].dropna()
        if len(train_model_df) < 8:
            continue

        X_train = train_model_df[FEATURE_COLUMNS]
        y_train = train_model_df[stat_col]

        X_test = test_row[FEATURE_COLUMNS]
        if X_test.isna().any():
            continue
        X_test = X_test.to_frame().T

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        predicted = float(model.predict(X_test)[0])
        actual = float(test_row[stat_col])

        results.append({
            "game_date": test_row["GAME_DATE"],
            "matchup":   test_row["MATCHUP"],
            "actual":    round(actual, 1),
            "predicted": round(predicted, 1),
            "error":     round(actual - predicted, 1),
        })

    return pd.DataFrame(results)


def evaluate_regression_results(results_df: pd.DataFrame, stat_col: str = "PTS") -> None:
    """Print regression backtest metrics: MAE, within-N accuracy, and bias."""
    print(f"\n{'=' * 60}")
    print(f"REGRESSION BACKTEST — {stat_col}")
    print(f"{'=' * 60}")

    if results_df.empty:
        print("No predictions generated. Player may have too few games.")
        return

    mae = float(results_df["error"].abs().mean())
    bias = float(results_df["error"].mean())
    within_2_5 = float((results_df["error"].abs() <= 2.5).mean())
    within_5 = float((results_df["error"].abs() <= 5.0).mean())

    print(f"Predictions made: {len(results_df)}")
    print(f"MAE:              {mae:.1f} pts")
    print(f"Within ±2.5:      {within_2_5:.1%}")
    print(f"Within ±5.0:      {within_5:.1%}")
    bias_dir = "undershoots" if bias > 0 else "overshoots"
    print(f"Bias:             {bias:+.1f}  (model {bias_dir} actual)")

    print("\nMost recent predictions (last 10):")
    recent = results_df.tail(10).copy()
    recent["game_date"] = pd.to_datetime(recent["game_date"]).dt.date
    print(recent[["game_date", "matchup", "actual", "predicted", "error"]].to_string(index=False))


def predict_next_game(
    df: pd.DataFrame,
    stat_col: str = "PTS",
    lookback_games: int = DEFAULT_LOOKBACK_GAMES,
) -> dict:
    """
    Train on the last N games and predict the next unplayed game's stat value.

    Called at inference time — run this before a game, then compare the returned
    prediction to the real PrizePicks line to get an Over/Under recommendation.

    Returns:
        {"predicted": float, "margin_of_error": float}
        where margin_of_error is ±1 std dev of training residuals.
        Returns {"predicted": None, "margin_of_error": None} if data is insufficient.
    """
    train_df = df[FEATURE_COLUMNS + [stat_col]].dropna().tail(lookback_games)
    if len(train_df) < 8:
        return {"predicted": None, "margin_of_error": None}

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[stat_col]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Margin of error = ±1 std dev of training residuals
    residuals = y_train.values - model.predict(X_train)
    margin_of_error = round(float(np.std(residuals)), 1)

    # Build next-game feature vector from most recent completed games.
    # is_home is unknown before seeing the schedule, so use 0.5 (neutral).
    last_game_date = df["GAME_DATE"].iloc[-1]
    days_since_last = (pd.Timestamp.now().normalize() - pd.Timestamp(last_game_date)).days

    if days_since_last > 7:
        # Data is likely stale — clamp to typical in-season rest to avoid
        # extrapolation error. User should run setup.py to refresh.
        print(f"  Warning: last cached game was {days_since_last} days ago.")
        print("  Run setup.py to refresh game logs for a more accurate prediction.")
        rest_days = 2.0  # typical back-to-back/short rest default
    else:
        rest_days = float(max(0, days_since_last))

    next_features = {
        "is_home":   0.5,
        "rest_days": rest_days,
        "pts_avg_3": float(df["PTS"].tail(3).mean()),
        "pts_avg_5": float(df["PTS"].tail(5).mean()),
        "reb_avg_3": float(df["REB"].tail(3).mean()),
        "ast_avg_3": float(df["AST"].tail(3).mean()),
        "min_avg_3": float(df["MIN"].tail(3).mean()),
    }

    X_next = pd.DataFrame([next_features])[FEATURE_COLUMNS]
    predicted = round(float(model.predict(X_next)[0]), 1)

    return {"predicted": predicted, "margin_of_error": margin_of_error}


def main() -> None:
    print("NBA Player Props Learning Script (PTS Over/Under Classification)")
    print("- Uses logistic regression")
    print(f"- Uses previous {LINE_PROXY_WINDOW}-game average as a proxy betting line")
    print("- Uses walk-forward backtesting (train on past, test on next game)\n")

    player_name = input("Enter player name (ex: Stephen Curry): ").strip()
    season_input = input(f"Enter season [{DEFAULT_SEASON}]: ").strip()
    season = season_input or DEFAULT_SEASON

    player_id = find_player_id(player_name)
    if player_id is None:
        print(f"Player '{player_name}' not found.")
        return

    raw_df = get_player_stats(player_id, season=season)
    if raw_df is None or raw_df.empty:
        print("Failed to fetch player game log.")
        print("Check your internet connection, season string, or nba_api availability.")
        return

    clean_df = clean_player_games(raw_df)
    feature_df = build_features(clean_df)

    print_player_summary(clean_df, player_name)
    print_feature_preview(feature_df)

    results_df = walk_forward_backtest(feature_df, lookback_games=DEFAULT_LOOKBACK_GAMES)
    evaluate_results(results_df)

    print(f"\nNext steps to improve this model:")
    print("1. Replace proxy lines with real sportsbook lines")
    print("2. Add opponent/team context features")
    print("3. Tune lookback window and probability threshold")
    print("4. Backtest across many players, not just one")


if __name__ == "__main__":
    main()
