"""
NBA player info utilities.

This module is intentionally focused on nba_api player data only:
- resolve player name -> player ID
- fetch raw game logs
- clean game logs into modeling-friendly columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


DEFAULT_SEASON = "2025-26"


def find_player_id(name: str) -> int | None:
    """Resolve a player full name to NBA player ID."""
    matches = players.find_players_by_full_name(name)
    if not matches:
        return None
    return matches[0]["id"]


def get_player_stats(player_id: int, season: str = DEFAULT_SEASON) -> pd.DataFrame | None:
    """Fetch raw game logs for a player from nba_api."""
    try:
        game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return game_log.get_data_frames()[0]
    except Exception:
        return None


def _minutes_to_float(value) -> float:
    """Convert MIN values like '32:45' to decimal minutes."""
    if pd.isna(value):
        return np.nan

    text = str(value)
    if ":" not in text:
        return float(text)

    minutes, seconds = text.split(":", maxsplit=1)
    return float(minutes) + (float(seconds) / 60.0)


def clean_player_games(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Keep required columns and normalize datatypes."""
    keep_cols = ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"]
    df = raw_df[keep_cols].copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    df["MIN"] = df["MIN"].apply(_minutes_to_float)
    for col in ["PTS", "REB", "AST"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
