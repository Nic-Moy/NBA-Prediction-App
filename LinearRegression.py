import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_DEFAULT_FEATURES: Dict[str, List[str]] = {
    "PTS": ["MIN", "REB", "AST", "FG_PCT", "FG3_PCT"],
    "REB": ["MIN", "PTS", "AST", "FG_PCT"],
    "AST": ["MIN", "PTS", "REB", "FG3_PCT"],
}


def _instantiate_model(model_type: str = "linear", alpha: float = 1.0):
    """Return an sklearn regression model based on the selected type."""
    model_type = (model_type or "linear").lower()

    if model_type == "linear":
        return LinearRegression()
    if model_type == "ridge":
        return Ridge(alpha=alpha)
    if model_type == "lasso":
        # Higher max_iter helps convergence when data is limited.
        return Lasso(alpha=alpha, max_iter=10_000)

    raise ValueError(f"Unsupported model_type '{model_type}'.")


def _clean_numeric_subset(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Cast a subset of columns to numeric and drop rows with missing values.

    Parameters
    ----------
    df:
        Input dataframe.
    columns:
        Columns that should be retained and coerced to numeric.
    """
    subset = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return subset.dropna()


def train_stat_model(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    *,
    model_type: str = "linear",
    alpha: float = 1.0,
) -> Tuple[RegressorMixin, pd.DataFrame]:
    """
    Train a regression model for a single target stat.

    Parameters
    ----------
    df:
        Player game log dataframe.
    feature_columns:
        Columns to use as predictors.
    target_column:
        Column to predict.
    model_type:
        One of ``linear``, ``ridge``, or ``lasso``.
    alpha:
        Regularisation strength for Ridge/Lasso models.

    Returns
    -------
    model, training_frame:
        The fitted model and the cleaned numeric frame that was actually used.
    """
    features = list(feature_columns)
    required_columns = features + [target_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Dataframe is missing required columns: {missing}")

    numeric = _clean_numeric_subset(df, required_columns)
    if len(numeric) < 2:
        raise ValueError("Not enough valid rows to train the regression model.")

    model = _instantiate_model(model_type=model_type, alpha=alpha)
    model.fit(numeric[features].values, numeric[target_column].values)
    return model, numeric


def predict_next_stat(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    *,
    model_type: str = "linear",
    alpha: float = 1.0,
    window: Optional[int] = None,
) -> Tuple[float, RegressorMixin]:
    """
    Predict the next value of ``target_column`` using a regression model.

    Parameters
    ----------
    df:
        Player game log dataframe.
    feature_columns:
        Columns to use as predictors.
    target_column:
        Column to predict.
    model_type:
        One of ``linear``, ``ridge``, or ``lasso``.
    alpha:
        Regularisation strength for Ridge/Lasso models.
    window:
        Optional number of most recent games to use when training.

    Returns
    -------
    prediction, model:
        The predicted stat value and the fitted regression model.
    """
    features = list(feature_columns)
    required_columns = features + [target_column]

    lookback_df = df.sort_values("GAME_DATE") if "GAME_DATE" in df.columns else df.copy()
    if window is not None and window > 0:
        lookback_df = lookback_df.tail(window)

    model, numeric = train_stat_model(
        lookback_df,
        features,
        target_column,
        model_type=model_type,
        alpha=alpha,
    )

    feature_frame = numeric[features]
    feature_row = feature_frame.mean(axis=0).to_numpy().reshape(1, -1)
    prediction = float(model.predict(feature_row)[0])
    return prediction, model


def forecast_player_stats(
    df: pd.DataFrame,
    *,
    stats: Optional[Iterable[str]] = None,
    model_type: str = "linear",
    alpha: float = 1.0,
    window: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Build regression models for multiple stats and return their predictions.

    Parameters
    ----------
    df:
        Player game log dataframe (ideally cleaned with numeric columns).
    stats:
        Iterable of stat names you want predictions for (defaults to PTS, REB, AST).
    model_type:
        One of ``linear``, ``ridge``, or ``lasso``.
    alpha:
        Regularisation strength for Ridge/Lasso models.
    window:
        Optional number of most recent games to use when training each stat model.

    Returns
    -------
    Dict[str, Dict[str, object]]:
        Mapping of stat -> {"prediction": float, "model": sklearn estimator, "features": List[str]}.
    """
    desired_stats = list(stats) if stats is not None else list(_DEFAULT_FEATURES.keys())
    outputs: Dict[str, Dict[str, object]] = {}

    for stat in desired_stats:
        default_features = _DEFAULT_FEATURES.get(stat, [])
        available_features = [col for col in default_features if col in df.columns and col != stat]

        # Skip stats where we have no usable features.
        if not available_features or stat not in df.columns:
            continue

        try:
            prediction, model = predict_next_stat(
                df,
                available_features,
                stat,
                model_type=model_type,
                alpha=alpha,
                window=window,
            )
        except (KeyError, ValueError):
            # Not enough data or required columns missing; skip prediction.
            continue

        outputs[stat] = {
            "prediction": prediction,
            "model": model,
            "features": available_features,
        }

    return outputs
