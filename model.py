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
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from getplayerinfo import (
    DEFAULT_SEASON as PLAYERINFO_DEFAULT_SEASON,
    clean_player_games,
    find_player_id,
    get_player_stats,
    parse_opponent_from_matchup,
)
from database import (
    load_opponent_stats_by_abbrev,
    load_opponent_game_advanced_history_by_abbrev,
)


DEFAULT_SEASON = PLAYERINFO_DEFAULT_SEASON
DEFAULT_MIN_TRAIN_SIZE = 15  # minimum prior games before first prediction
LINE_PROXY_WINDOW = 5


def _make_classifier(C: float = 0.3) -> Pipeline:
    """Scaled, L2-regularized LogReg with balanced class weights.

    StandardScaler removes scale dominance (opp_def_rating ~113 vs is_home 0/1).
    C=0.3 is tighter than sklearn default to combat p>n overfitting.
    class_weight='balanced' counters the OVER-heavy label distribution from the
    rolling proxy line.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, class_weight="balanced",
            random_state=42, max_iter=1000,
        )),
    ])


def _make_regressor(alpha: float = 1.0) -> Pipeline:
    """Scaled Ridge regressor. Same scaler so coefficients are comparable."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=alpha)),
    ])

# Only include features we could know before the game starts.
# Every rolling stat is built with .shift(1) so it depends on prior games only.
BASE_FEATURE_COLUMNS = [
    # Schedule
    "is_home",
    "rest_days",
    "rest_bucket",
    "month",
    # Points form
    "pts_avg_3",
    "pts_avg_5",
    "pts_avg_10",
    "pts_form_delta",
    # Other base stats
    "reb_avg_3",
    "ast_avg_3",
    "min_avg_3",
    "min_avg_5",
    # Volume (prior 5)
    "fga_avg_5",
    "fg3a_avg_5",
    "fta_avg_5",
    "tov_avg_5",
    # Efficiency (prior 5)
    "fg_pct_avg_5",
    "fg3_pct_avg_5",
    "ft_pct_avg_5",
    "ts_pct_avg_5",
    # Usage proxy
    "usage_proxy_avg_5",
    # Variance / consistency
    "pts_std_5",
    "min_std_5",
    "pts_min_5",
    "pts_max_5",
    # Team momentum
    "wl_avg_5",
    "plus_minus_avg_5",
]

OPPONENT_STATIC_FEATURE_COLUMNS = [
    "opp_def_rating",
    "opp_off_rating",
    "opp_pace",
]

OPPONENT_ROLLING_FEATURE_COLUMNS = [
    "opp_def_avg_5",
    "opp_off_avg_5",
    "opp_pace_avg_5",
    "opp_def_trend_5",
    "opp_pace_trend_5",
]

# Default training set uses game-level rolling opponent context (with static fallback).
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + OPPONENT_ROLLING_FEATURE_COLUMNS

def _add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Schedule context known before tipoff: home/away, rest, month."""
    df["is_home"] = df["MATCHUP"].astype(str).str.contains("vs.", regex=False).astype(int)
    df["rest_days"] = df["GAME_DATE"].diff().dt.days
    df["rest_days"] = df["rest_days"].fillna(3).clip(lower=0)
    df["rest_bucket"] = pd.cut(
        df["rest_days"],
        bins=[-0.001, 0.999, 1.999, 3.999, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(float)
    df["month"] = df["GAME_DATE"].dt.month
    return df


def _add_rolling_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling averages of base stats from PRIOR games only (shifted)."""
    df["pts_avg_3"]  = df["PTS"].shift(1).rolling(window=3,  min_periods=1).mean()
    df["pts_avg_5"]  = df["PTS"].shift(1).rolling(window=5,  min_periods=1).mean()
    df["pts_avg_10"] = df["PTS"].shift(1).rolling(window=10, min_periods=1).mean()
    df["pts_form_delta"] = df["pts_avg_3"] - df["pts_avg_10"]

    df["reb_avg_3"] = df["REB"].shift(1).rolling(window=3, min_periods=1).mean()
    df["ast_avg_3"] = df["AST"].shift(1).rolling(window=3, min_periods=1).mean()
    df["min_avg_3"] = df["MIN"].shift(1).rolling(window=3, min_periods=1).mean()
    df["min_avg_5"] = df["MIN"].shift(1).rolling(window=5, min_periods=1).mean()
    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume averages over prior 5 games: shots taken, turnovers."""
    for src, out in [
        ("FGA",  "fga_avg_5"),
        ("FG3A", "fg3a_avg_5"),
        ("FTA",  "fta_avg_5"),
        ("TOV",  "tov_avg_5"),
    ]:
        if src in df.columns:
            df[out] = df[src].shift(1).rolling(window=5, min_periods=1).mean()
        else:
            df[out] = np.nan
    return df


def _add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shooting efficiency averages over prior 5 games."""
    for src, out in [
        ("FG_PCT",  "fg_pct_avg_5"),
        ("FG3_PCT", "fg3_pct_avg_5"),
        ("FT_PCT",  "ft_pct_avg_5"),
        ("TS_PCT",  "ts_pct_avg_5"),
    ]:
        if src in df.columns:
            df[out] = df[src].shift(1).rolling(window=5, min_periods=1).mean()
        else:
            df[out] = np.nan
    return df


def _add_usage_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Usage proxy: (FGA + 0.44*FTA + TOV) / MIN averaged over prior 5 games."""
    if {"FGA", "FTA", "TOV", "MIN"}.issubset(df.columns):
        minutes = df["MIN"].replace(0, np.nan)
        usage = (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / minutes
        df["usage_proxy_avg_5"] = usage.shift(1).rolling(window=5, min_periods=1).mean()
    else:
        df["usage_proxy_avg_5"] = np.nan
    return df


def _add_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variance / range of recent performance — captures volatility for over/under."""
    df["pts_std_5"] = df["PTS"].shift(1).rolling(window=5, min_periods=2).std()
    df["min_std_5"] = df["MIN"].shift(1).rolling(window=5, min_periods=2).std()
    df["pts_min_5"] = df["PTS"].shift(1).rolling(window=5, min_periods=1).min()
    df["pts_max_5"] = df["PTS"].shift(1).rolling(window=5, min_periods=1).max()
    return df


def _add_team_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Team-level momentum signals from prior 5 games."""
    if "WL" in df.columns:
        wins = (df["WL"] == "W").astype(int)
        df["wl_avg_5"] = wins.shift(1).rolling(window=5, min_periods=1).mean()
    else:
        df["wl_avg_5"] = np.nan

    if "PLUS_MINUS" in df.columns:
        df["plus_minus_avg_5"] = df["PLUS_MINUS"].shift(1).rolling(window=5, min_periods=1).mean()
    else:
        df["plus_minus_avg_5"] = np.nan
    return df


def _rolling_avg_and_trend(series: pd.Series) -> tuple[float, float]:
    """Return (last-5 average, short-term trend = mean(last3)-mean(prev2))."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan"), float("nan")

    last5 = s.tail(5)
    avg_5 = float(last5.mean()) if not last5.empty else float("nan")

    recent3 = last5.tail(3)
    prev2 = last5.head(max(0, len(last5) - 3))
    if recent3.empty or prev2.empty:
        trend = float("nan")
    else:
        trend = float(recent3.mean() - prev2.mean())
    return avg_5, trend


def _add_opponent_features(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Attach opponent context with game-level rolling features + static fallback."""
    try:
        opp_static = load_opponent_stats_by_abbrev(season)
    except Exception as e:
        print(f"  Warning: could not load season opponent stats ({e}).")
        opp_static = {}

    try:
        opp_history = load_opponent_game_advanced_history_by_abbrev(season)
    except Exception as e:
        print(f"  Warning: could not load game-level opponent stats ({e}).")
        opp_history = {}

    opps = df["MATCHUP"].apply(parse_opponent_from_matchup)
    df["opp_def_rating"] = opps.map(lambda a: (opp_static.get(a) or {}).get("def_rating"))
    df["opp_off_rating"] = opps.map(lambda a: (opp_static.get(a) or {}).get("off_rating"))
    df["opp_pace"] = opps.map(lambda a: (opp_static.get(a) or {}).get("pace"))

    roll_def_avg: list[float] = []
    roll_off_avg: list[float] = []
    roll_pace_avg: list[float] = []
    roll_def_trend: list[float] = []
    roll_pace_trend: list[float] = []

    game_dates = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    for opp_abbrev, game_date in zip(opps, game_dates):
        hist = opp_history.get(opp_abbrev or "")
        if hist is None or pd.isna(game_date):
            roll_def_avg.append(float("nan"))
            roll_off_avg.append(float("nan"))
            roll_pace_avg.append(float("nan"))
            roll_def_trend.append(float("nan"))
            roll_pace_trend.append(float("nan"))
            continue

        prior = hist[hist["game_date"] < game_date]
        def_avg, def_trend = _rolling_avg_and_trend(prior["def_rating"])
        off_avg, _ = _rolling_avg_and_trend(prior["off_rating"])
        pace_avg, pace_trend = _rolling_avg_and_trend(prior["pace"])
        roll_def_avg.append(def_avg)
        roll_off_avg.append(off_avg)
        roll_pace_avg.append(pace_avg)
        roll_def_trend.append(def_trend)
        roll_pace_trend.append(pace_trend)

    df["opp_def_avg_5"] = roll_def_avg
    df["opp_off_avg_5"] = roll_off_avg
    df["opp_pace_avg_5"] = roll_pace_avg
    df["opp_def_trend_5"] = roll_def_trend
    df["opp_pace_trend_5"] = roll_pace_trend

    # Static season-level values are fallback when game-level context is missing.
    df["opp_def_avg_5"] = df["opp_def_avg_5"].fillna(df["opp_def_rating"])
    df["opp_off_avg_5"] = df["opp_off_avg_5"].fillna(df["opp_off_rating"])
    df["opp_pace_avg_5"] = df["opp_pace_avg_5"].fillna(df["opp_pace"])
    df["opp_def_trend_5"] = df["opp_def_trend_5"].fillna(0.0)
    df["opp_pace_trend_5"] = df["opp_pace_trend_5"].fillna(0.0)
    return df


def _add_proxy_line_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy betting line + binary over/under label.

    Line = previous LINE_PROXY_WINDOW-game points average.
    Label = 1 if actual PTS > line, else 0. NaN until enough history exists.
    """
    df["line_proxy_pts"] = df["PTS"].shift(1).rolling(
        window=LINE_PROXY_WINDOW,
        min_periods=LINE_PROXY_WINDOW,
    ).mean()

    df["target_over_pts"] = np.nan
    valid_line = df["line_proxy_pts"].notna()
    df.loc[valid_line, "target_over_pts"] = (
        df.loc[valid_line, "PTS"] > df.loc[valid_line, "line_proxy_pts"]
    ).astype(int)
    return df


def build_features(df: pd.DataFrame, season: str = DEFAULT_SEASON) -> pd.DataFrame:
    """
    Add pre-game features and labels.

    Important:
    - We use .shift(1) so features come from PRIOR games only.
    - That avoids data leakage.
    """
    df = df.copy()
    df = _add_schedule_features(df)
    df = _add_rolling_form_features(df)
    df = _add_volume_features(df)
    df = _add_efficiency_features(df)
    df = _add_usage_proxy(df)
    df = _add_consistency_features(df)
    df = _add_team_momentum_features(df)
    df = _add_opponent_features(df, season=season)
    df = _add_proxy_line_and_label(df)
    return df


def walk_forward_backtest(
    df: pd.DataFrame,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Walk-forward backtest for points over/under classification.

    Expanding window: each fold trains on ALL prior games (df.iloc[:i]) rather
    than the most recent N. Train set grows over the season — by fold 60 the
    model sees 60 rows instead of 15, easing the p>n overfitting problem.
    """
    results: list[dict] = []
    feature_columns = feature_columns or FEATURE_COLUMNS

    for i in range(min_train_size, len(df)):
        train_window = df.iloc[:i].copy()  # expanding: every prior game
        test_row = df.iloc[i]

        # Skip if we don't have a proxy line for the test game yet
        if pd.isna(test_row["line_proxy_pts"]):
            continue

        # Prepare training data
        train_model_df = train_window[feature_columns + ["target_over_pts"]].dropna()
        if len(train_model_df) < 8:
            continue

        X_train = train_model_df[feature_columns]
        y_train = train_model_df["target_over_pts"].astype(int)

        # Logistic regression needs both classes in the training window
        if y_train.nunique() < 2:
            continue

        # Prepare test row
        X_test = test_row[feature_columns]
        if X_test.isna().any():
            continue
        X_test = X_test.to_frame().T  # 1-row DataFrame

        model = _make_classifier()
        model.fit(X_train, y_train)

        # Train accuracy on the same window we just fit on — a large gap vs
        # test accuracy is the textbook overfitting signal.
        train_accuracy = float((model.predict(X_train) == y_train.values).mean())

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
                "train_accuracy": train_accuracy,
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


def print_feature_preview(df: pd.DataFrame, rows: int = 8) -> None:
    """Quick peek at engineered feature columns."""
    cols = ["GAME_DATE", "MATCHUP"] + FEATURE_COLUMNS + ["line_proxy_pts", "target_over_pts"]
    keep = [c for c in cols if c in df.columns]
    print(f"\nFeature preview (last {rows} rows):")
    print(df[keep].tail(max(1, rows)).to_string(index=False))


def _print_confusion_matrix(results_df: pd.DataFrame) -> None:
    """2x2 confusion matrix + class balance gap warning."""
    cm = confusion_matrix(
        results_df["actual_over"].astype(int),
        results_df["pred_over"].astype(int),
        labels=[0, 1],
    )
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix:")
    print(f"                   pred_UNDER   pred_OVER")
    print(f"  actual_UNDER       {tn:5d}       {fp:5d}")
    print(f"  actual_OVER        {fn:5d}       {tp:5d}")

    pred_over_rate   = float((results_df["pred_over"] == 1).mean())
    actual_over_rate = float((results_df["actual_over"] == 1).mean())
    print(f"  Predicted OVER:   {pred_over_rate:.1%}")
    print(f"  Actual OVER:      {actual_over_rate:.1%}")
    gap = abs(pred_over_rate - actual_over_rate)
    if gap > 0.15:
        print(f"  ⚠ Class-balance gap {gap:.1%} — model leaning toward majority.")


def _print_brier(results_df: pd.DataFrame) -> None:
    """Brier score: mean squared error of probability vs outcome.

    0.00 = perfect, 0.25 = coin flip, > 0.25 = worse than random.
    """
    brier = brier_score_loss(
        results_df["actual_over"].astype(int),
        results_df["pred_prob_over"].astype(float),
    )
    print(f"\nBrier score: {brier:.3f}   (0 = perfect, 0.25 = coin flip)")
    if brier < 0.20:
        print("  Probabilities carry real signal.")
    elif brier < 0.25:
        print("  Probabilities marginally useful.")
    else:
        print("  ⚠ Probabilities worse than guessing — model is miscalibrated.")


def _print_train_vs_test(results_df: pd.DataFrame) -> None:
    """Compare per-fold train accuracy vs test accuracy to spot overfit."""
    if "train_accuracy" not in results_df.columns:
        return
    train_acc = float(results_df["train_accuracy"].mean())
    test_acc  = float(results_df["correct"].mean())
    gap = train_acc - test_acc
    print(f"\nTrain accuracy (avg per fold): {train_acc:.1%}")
    print(f"Test  accuracy (overall):      {test_acc:.1%}")
    print(f"Gap (train - test):            {gap:+.1%}")
    if gap > 0.20:
        print("  ⚠ Gap > 20% — overfitting (memorizing train, missing test).")
    elif gap < 0.05 and test_acc < 0.50:
        print("  ⚠ Both low — underfitting (no signal in features).")


def evaluate_results(results_df: pd.DataFrame) -> None:
    """Print backtest metrics: accuracy, confusion, Brier, train-vs-test, sample."""
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

    _print_confusion_matrix(results_df)
    _print_brier(results_df)
    _print_train_vs_test(results_df)

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


def _ablation_accuracy(
    df: pd.DataFrame,
    feature_columns: list[str],
    min_train_size: int,
) -> pd.DataFrame:
    """Run walk-forward and return game-level correctness for a feature subset."""
    out = walk_forward_backtest(
        df,
        min_train_size=min_train_size,
        feature_columns=feature_columns,
    )
    if out.empty:
        return pd.DataFrame(columns=["game_date", "matchup", "correct"])
    return out[["game_date", "matchup", "correct"]].copy()


def print_opponent_feature_diagnostics(
    feature_df: pd.DataFrame,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
) -> None:
    """Print opponent feature coverage and same-split ablation checks."""
    print(f"\n{'=' * 60}")
    print("OPPONENT FEATURE DIAGNOSTICS")
    print(f"{'=' * 60}")

    opp_series = feature_df["MATCHUP"].apply(parse_opponent_from_matchup)
    unique_opps = opp_series.dropna().nunique()
    print(f"Unique opponents seen: {unique_opps}")

    for col in OPPONENT_ROLLING_FEATURE_COLUMNS:
        if col in feature_df.columns:
            null_rate = float(feature_df[col].isna().mean())
            print(f"{col}: null rate {null_rate:.1%}")

    baseline_cols = BASE_FEATURE_COLUMNS
    static_opp_cols = OPPONENT_STATIC_FEATURE_COLUMNS
    rolling_opp_cols = OPPONENT_ROLLING_FEATURE_COLUMNS
    full_static_cols = baseline_cols + static_opp_cols
    full_rolling_cols = baseline_cols + rolling_opp_cols

    baseline = _ablation_accuracy(feature_df, baseline_cols, min_train_size)
    opp_only = _ablation_accuracy(feature_df, rolling_opp_cols, min_train_size)

    merged = baseline.merge(
        opp_only,
        on=["game_date", "matchup"],
        suffixes=("_baseline", "_opp_only"),
    )
    if merged.empty:
        print("Baseline vs opp-only ablation: not enough overlapping folds.")
    else:
        base_acc = float(merged["correct_baseline"].mean())
        opp_acc = float(merged["correct_opp_only"].mean())
        print(
            "Baseline vs opp-only (same folds): "
            f"{base_acc:.1%} vs {opp_acc:.1%} (delta {opp_acc - base_acc:+.1%})"
        )

    # Controlled A/B on the same folds: baseline vs full static vs full rolling.
    full_static = _ablation_accuracy(feature_df, full_static_cols, min_train_size)
    full_rolling = _ablation_accuracy(feature_df, full_rolling_cols, min_train_size)
    ab = baseline.merge(full_static, on=["game_date", "matchup"], suffixes=("_base", "_static"))
    abr = ab.merge(full_rolling, on=["game_date", "matchup"])
    if abr.empty:
        print("A/B (baseline/static/rolling): not enough overlapping folds.")
    else:
        base_acc = float(abr["correct_base"].mean())
        static_acc = float(abr["correct_static"].mean())
        rolling_acc = float(abr["correct"].mean())
        print("A/B on identical folds:")
        print(f"  Baseline (no opp): {base_acc:.1%}")
        print(f"  Static opponent:    {static_acc:.1%}")
        print(f"  Rolling opponent:   {rolling_acc:.1%}")


def walk_forward_regression(
    df: pd.DataFrame,
    stat_col: str = "PTS",
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
) -> pd.DataFrame:
    """
    Walk-forward regression backtest: predict actual stat value for each game.

    Expanding window — each fold trains a scaled Ridge on every prior game.
    stat_col should be an uppercase column name: "PTS", "REB", or "AST".
    """
    results: list[dict] = []

    for i in range(min_train_size, len(df)):
        train_window = df.iloc[:i].copy()  # expanding window
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

        model = _make_regressor()
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


def compute_permutation_importance(
    feature_df: pd.DataFrame,
    n_repeats: int = 15,
    test_frac: float = 0.25,
) -> pd.DataFrame:
    """Time-respecting permutation importance for the over/under classifier.

    Fits LogReg on first (1 - test_frac) of valid rows, then measures accuracy
    drop when each feature is shuffled in the held-out tail. Positive importance
    = feature helps; near-zero or negative = feature is noise.
    """
    valid = feature_df[FEATURE_COLUMNS + ["target_over_pts"]].dropna()
    if len(valid) < 20:
        print("Not enough rows for permutation importance (need >= 20).")
        return pd.DataFrame()

    split = int(len(valid) * (1.0 - test_frac))
    X_train = valid[FEATURE_COLUMNS].iloc[:split]
    y_train = valid["target_over_pts"].astype(int).iloc[:split]
    X_test  = valid[FEATURE_COLUMNS].iloc[split:]
    y_test  = valid["target_over_pts"].astype(int).iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print("Permutation importance needs both classes in train and test.")
        return pd.DataFrame()

    model = _make_classifier()
    model.fit(X_train, y_train)

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=42, scoring="accuracy",
    )

    return pd.DataFrame({
        "feature":         FEATURE_COLUMNS,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_permutation_importance(importance_df: pd.DataFrame, save_path: str | None = None) -> None:
    """Horizontal bar plot of permutation importance (mean ± std).

    Shows plt window unless save_path is given.
    """
    if importance_df.empty:
        return

    import matplotlib.pyplot as plt  # local import — only loaded if user wants the plot

    print(f"\n{'=' * 60}")
    print("PERMUTATION FEATURE IMPORTANCE (held-out tail)")
    print(f"{'=' * 60}")
    print("Bar = accuracy drop when feature is shuffled. Positive = helps.\n")

    ordered = importance_df.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(ordered))))
    ax.barh(
        ordered["feature"],
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color=["#2ca02c" if v > 0 else "#d62728" for v in ordered["importance_mean"]],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean accuracy drop when shuffled")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"  Plot saved to {save_path}")
    else:
        plt.show()


def _rest_bucket_from_days(rest_days: float) -> float:
    """Match the bucketing used in _add_schedule_features for live prediction."""
    if rest_days < 1:
        return 0.0
    if rest_days < 2:
        return 1.0
    if rest_days < 4:
        return 2.0
    return 3.0


def _opt_tail_mean(df: pd.DataFrame, col: str, n: int) -> float:
    """Mean of last n values of col if present; NaN otherwise."""
    if col not in df.columns:
        return float("nan")
    series = df[col].tail(n).dropna()
    return float(series.mean()) if not series.empty else float("nan")


def _opt_tail_std(df: pd.DataFrame, col: str, n: int) -> float:
    if col not in df.columns:
        return float("nan")
    series = df[col].tail(n).dropna()
    return float(series.std()) if len(series) >= 2 else float("nan")


def _opt_tail_min(df: pd.DataFrame, col: str, n: int) -> float:
    if col not in df.columns:
        return float("nan")
    series = df[col].tail(n).dropna()
    return float(series.min()) if not series.empty else float("nan")


def _opt_tail_max(df: pd.DataFrame, col: str, n: int) -> float:
    if col not in df.columns:
        return float("nan")
    series = df[col].tail(n).dropna()
    return float(series.max()) if not series.empty else float("nan")


def _build_next_game_features(
    df: pd.DataFrame,
    rest_days: float,
    next_opponent_abbrev: str | None,
    season: str,
) -> dict:
    """Construct the pregame feature row for the next unplayed game.

    Mirrors the .shift(1).rolling() logic used in build_features by pulling the
    last N completed games. is_home is unknown without a schedule, so 0.5 is
    used as a neutral default.
    """
    last_date = pd.Timestamp(df["GAME_DATE"].iloc[-1])
    next_month = last_date.month  # crude default — caller may override later

    pts_avg_3  = _opt_tail_mean(df, "PTS", 3)
    pts_avg_10 = _opt_tail_mean(df, "PTS", 10)

    # Usage proxy uses raw columns, not the pre-rolled feature column.
    if {"FGA", "FTA", "TOV", "MIN"}.issubset(df.columns):
        tail = df.tail(5)
        minutes = tail["MIN"].replace(0, np.nan)
        usage_series = (tail["FGA"] + 0.44 * tail["FTA"] + tail["TOV"]) / minutes
        usage_proxy = float(usage_series.dropna().mean()) if not usage_series.dropna().empty else float("nan")
    else:
        usage_proxy = float("nan")

    if "WL" in df.columns:
        wl_avg_5 = float((df["WL"].tail(5) == "W").astype(int).mean())
    else:
        wl_avg_5 = float("nan")

    try:
        opp_stats = load_opponent_stats_by_abbrev(season)
    except Exception:
        opp_stats = {}
    opp = opp_stats.get(next_opponent_abbrev or "", {}) if next_opponent_abbrev else {}

    try:
        opp_history = load_opponent_game_advanced_history_by_abbrev(season)
    except Exception:
        opp_history = {}
    opp_hist_df = opp_history.get(next_opponent_abbrev or "", pd.DataFrame())
    opp_def_avg_5, opp_def_trend_5 = _rolling_avg_and_trend(opp_hist_df.get("def_rating", pd.Series(dtype=float)))
    opp_off_avg_5, _ = _rolling_avg_and_trend(opp_hist_df.get("off_rating", pd.Series(dtype=float)))
    opp_pace_avg_5, opp_pace_trend_5 = _rolling_avg_and_trend(opp_hist_df.get("pace", pd.Series(dtype=float)))

    if pd.isna(opp_def_avg_5):
        opp_def_avg_5 = opp.get("def_rating", float("nan"))
    if pd.isna(opp_off_avg_5):
        opp_off_avg_5 = opp.get("off_rating", float("nan"))
    if pd.isna(opp_pace_avg_5):
        opp_pace_avg_5 = opp.get("pace", float("nan"))
    if pd.isna(opp_def_trend_5):
        opp_def_trend_5 = 0.0
    if pd.isna(opp_pace_trend_5):
        opp_pace_trend_5 = 0.0

    return {
        "is_home":            0.5,
        "rest_days":          rest_days,
        "rest_bucket":        _rest_bucket_from_days(rest_days),
        "month":              float(next_month),

        "pts_avg_3":          pts_avg_3,
        "pts_avg_5":          _opt_tail_mean(df, "PTS", 5),
        "pts_avg_10":         pts_avg_10,
        "pts_form_delta":     pts_avg_3 - pts_avg_10,

        "reb_avg_3":          _opt_tail_mean(df, "REB", 3),
        "ast_avg_3":          _opt_tail_mean(df, "AST", 3),
        "min_avg_3":          _opt_tail_mean(df, "MIN", 3),
        "min_avg_5":          _opt_tail_mean(df, "MIN", 5),

        "fga_avg_5":          _opt_tail_mean(df, "FGA", 5),
        "fg3a_avg_5":         _opt_tail_mean(df, "FG3A", 5),
        "fta_avg_5":          _opt_tail_mean(df, "FTA", 5),
        "tov_avg_5":          _opt_tail_mean(df, "TOV", 5),

        "fg_pct_avg_5":       _opt_tail_mean(df, "FG_PCT", 5),
        "fg3_pct_avg_5":      _opt_tail_mean(df, "FG3_PCT", 5),
        "ft_pct_avg_5":       _opt_tail_mean(df, "FT_PCT", 5),
        "ts_pct_avg_5":       _opt_tail_mean(df, "TS_PCT", 5),

        "usage_proxy_avg_5":  usage_proxy,

        "pts_std_5":          _opt_tail_std(df, "PTS", 5),
        "min_std_5":          _opt_tail_std(df, "MIN", 5),
        "pts_min_5":          _opt_tail_min(df, "PTS", 5),
        "pts_max_5":          _opt_tail_max(df, "PTS", 5),

        "wl_avg_5":           wl_avg_5,
        "plus_minus_avg_5":   _opt_tail_mean(df, "PLUS_MINUS", 5),

        "opp_def_avg_5":      opp_def_avg_5,
        "opp_off_avg_5":      opp_off_avg_5,
        "opp_pace_avg_5":     opp_pace_avg_5,
        "opp_def_trend_5":    opp_def_trend_5,
        "opp_pace_trend_5":   opp_pace_trend_5,
    }


def predict_next_game(
    df: pd.DataFrame,
    stat_col: str = "PTS",
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    next_opponent_abbrev: str | None = None,
    season: str = DEFAULT_SEASON,
) -> dict:
    """
    Train on ALL prior games and predict the next unplayed game's stat value.

    Called at inference time — run this before a game, then compare the returned
    prediction to the real PrizePicks line to get an Over/Under recommendation.

    Uses an expanding-window training set (consistent with walk_forward_*).
    min_train_size is a floor — if fewer valid rows exist, returns None.

    Pass next_opponent_abbrev (e.g. "BOS") to attach opponent defensive context;
    otherwise opponent features fall back to NaN (filled from train means).

    Returns:
        {"predicted": float, "margin_of_error": float}
        where margin_of_error is ±1 std dev of training residuals.
        Returns {"predicted": None, "margin_of_error": None} if data is insufficient.
    """
    train_df = df[FEATURE_COLUMNS + [stat_col]].dropna()
    if len(train_df) < min_train_size:
        return {"predicted": None, "margin_of_error": None}

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[stat_col]

    model = _make_regressor()
    model.fit(X_train, y_train)

    # Margin of error = ±1 std dev of training residuals
    residuals = y_train.values - model.predict(X_train)
    margin_of_error = round(float(np.std(residuals)), 1)

    # Build next-game feature vector from most recent completed games.
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

    next_features = _build_next_game_features(
        df,
        rest_days=rest_days,
        next_opponent_abbrev=next_opponent_abbrev,
        season=season,
    )

    X_next = pd.DataFrame([next_features])[FEATURE_COLUMNS]
    # Fill any remaining NaN (e.g. opponent stats when no abbrev passed, or
    # rolling features with too-short history) using training-set column means
    # so Ridge / LogReg can score the row.
    X_next = X_next.fillna(X_train.mean())

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
    feature_df = build_features(clean_df, season=season)

    print_player_summary(clean_df, player_name)
    print_feature_preview(feature_df)

    results_df = walk_forward_backtest(feature_df, min_train_size=DEFAULT_MIN_TRAIN_SIZE)
    evaluate_results(results_df)

    print(f"\nNext steps to improve this model:")
    print("1. Replace proxy lines with real sportsbook lines")
    print("2. Tune regularization C and threshold")
    print("3. Pool training across many players")
    print("4. Try HistGradientBoosting after pooling")


if __name__ == "__main__":
    main()
