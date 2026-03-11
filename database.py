"""
database.py — Postgres cache layer for NBA player data.

Provides:
- get_engine()        → shared SQLAlchemy engine
- ensure_tables()     → creates tables if they don't exist (safety net)
- upsert_game_logs()  → insert/update player game logs (no duplicates)
- load_cached_logs()  → read cached logs back as a DataFrame
- is_cache_fresh()    → check if we need to re-fetch from nba_api
- cache_player()      → store a player_id ↔ name mapping
- lookup_player_id()  → resolve name → id from local cache
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text


# ─── Connection config ────────────────────────────────────────

DB_CONFIG = {
    "host": "localhost",
    "database": "Betting_App",
    "user": "Nic",
    "password": "",
    "port": 5432,
}

_engine = None  # module-level singleton


def get_engine():
    """Return a shared SQLAlchemy engine (created once)."""
    global _engine
    if _engine is None:
        c = DB_CONFIG
        conn_str = (
            f"postgresql://{c['user']}:{c['password']}"
            f"@{c['host']}:{c['port']}/{c['database']}"
        )
        _engine = create_engine(conn_str, pool_pre_ping=True)
    return _engine


# ─── Schema creation (safety net) ─────────────────────────────

_CREATE_PLAYERS = """
CREATE TABLE IF NOT EXISTS players (
    player_id    INTEGER PRIMARY KEY,
    full_name    VARCHAR(100) NOT NULL,
    is_active    BOOLEAN,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_GAME_LOGS = """
CREATE TABLE IF NOT EXISTS player_game_logs (
    player_id    INTEGER      NOT NULL,
    game_id      VARCHAR(20)  NOT NULL,
    game_date    DATE         NOT NULL,
    matchup      VARCHAR(20)  NOT NULL,
    wl           CHAR(1),
    min          REAL,
    pts          REAL,
    reb          REAL,
    ast          REAL,
    fgm          REAL,
    fga          REAL,
    fg3m         REAL,
    fg3a         REAL,
    ftm          REAL,
    fta          REAL,
    stl          REAL,
    blk          REAL,
    tov          REAL,
    plus_minus   REAL,
    season       VARCHAR(10)  NOT NULL,
    fetched_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, game_id)
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_player_season
    ON player_game_logs (player_id, season);
CREATE INDEX IF NOT EXISTS idx_game_date
    ON player_game_logs (game_date);
"""

_CREATE_TEAMS = """
CREATE TABLE IF NOT EXISTS teams (
    team_id      INTEGER PRIMARY KEY,
    full_name    VARCHAR(50)  NOT NULL,
    abbreviation VARCHAR(5),
    city         VARCHAR(30),
    nickname     VARCHAR(30)
);
"""

_ALTER_PLAYERS_TEAM_POSITION = """
ALTER TABLE players ADD COLUMN IF NOT EXISTS team_id  INTEGER;
ALTER TABLE players ADD COLUMN IF NOT EXISTS position VARCHAR(10);
"""


def ensure_tables() -> None:
    """Create tables + indexes if they don't already exist. No-op if they do."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(_CREATE_PLAYERS))
        conn.execute(text(_CREATE_GAME_LOGS))
        conn.execute(text(_CREATE_INDEXES))
        conn.execute(text(_CREATE_TEAMS))
        conn.execute(text(_ALTER_PLAYERS_TEAM_POSITION))
    print("✓ Tables verified")


# ─── Teams cache ──────────────────────────────────────────────

def cache_team(team_id: int, full_name: str, abbreviation: str, city: str, nickname: str) -> None:
    """Insert or update a team row in the teams table."""
    sql = text("""
        INSERT INTO teams (team_id, full_name, abbreviation, city, nickname)
        VALUES (:team_id, :full_name, :abbreviation, :city, :nickname)
        ON CONFLICT (team_id) DO UPDATE SET
            full_name    = EXCLUDED.full_name,
            abbreviation = EXCLUDED.abbreviation,
            city         = EXCLUDED.city,
            nickname     = EXCLUDED.nickname;
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, dict(team_id=team_id, full_name=full_name,
                               abbreviation=abbreviation, city=city, nickname=nickname))


def update_player_team_position(player_id: int, team_id: int, position: str) -> None:
    """Update team_id and position for an existing player row."""
    sql = text("""
        UPDATE players SET team_id = :team_id, position = :position
        WHERE player_id = :player_id;
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, dict(player_id=player_id, team_id=team_id, position=position))


# ─── Players cache ────────────────────────────────────────────

def cache_player(player_id: int, full_name: str, is_active: bool = True) -> None:
    """Insert or update the local player name ↔ ID mapping."""
    engine = get_engine()
    sql = text("""
        INSERT INTO players (player_id, full_name, is_active, updated_at)
        VALUES (:pid, :name, :active, CURRENT_TIMESTAMP)
        ON CONFLICT (player_id)
        DO UPDATE SET full_name  = EXCLUDED.full_name,
                      is_active  = EXCLUDED.is_active,
                      updated_at = CURRENT_TIMESTAMP;
    """)
    with engine.begin() as conn:
        conn.execute(sql, {"pid": player_id, "name": full_name, "active": is_active})


def lookup_player_id(full_name: str) -> Optional[int]:
    """Try to resolve a player name from the local cache (case-insensitive)."""
    engine = get_engine()
    sql = text(
        "SELECT player_id FROM players WHERE LOWER(full_name) = LOWER(:name) LIMIT 1;"
    )
    with engine.begin() as conn:
        row = conn.execute(sql, {"name": full_name.strip()}).fetchone()
    return row[0] if row else None


# ─── Game log upsert ──────────────────────────────────────────

_LOG_COLS_MAP = {
    "Game_ID":    "game_id",
    "GAME_DATE":  "game_date",
    "MATCHUP":    "matchup",
    "WL":         "wl",
    "MIN":        "min",
    "PTS":        "pts",
    "REB":        "reb",
    "AST":        "ast",
    "FGM":        "fgm",
    "FGA":        "fga",
    "FG3M":       "fg3m",
    "FG3A":       "fg3a",
    "FTM":        "ftm",
    "FTA":        "fta",
    "STL":        "stl",
    "BLK":        "blk",
    "TOV":        "tov",
    "PLUS_MINUS": "plus_minus",
}

_UPSERT_SQL = text("""
    INSERT INTO player_game_logs
        (player_id, game_id, game_date, matchup, wl,
         min, pts, reb, ast, fgm, fga, fg3m, fg3a,
         ftm, fta, stl, blk, tov, plus_minus,
         season, fetched_at)
    VALUES
        (:player_id, :game_id, :game_date, :matchup, :wl,
         :min, :pts, :reb, :ast, :fgm, :fga, :fg3m, :fg3a,
         :ftm, :fta, :stl, :blk, :tov, :plus_minus,
         :season, CURRENT_TIMESTAMP)
    ON CONFLICT (player_id, game_id)
    DO UPDATE SET
         pts        = EXCLUDED.pts,
         reb        = EXCLUDED.reb,
         ast        = EXCLUDED.ast,
         min        = EXCLUDED.min,
         fgm        = EXCLUDED.fgm,
         fga        = EXCLUDED.fga,
         fg3m       = EXCLUDED.fg3m,
         fg3a       = EXCLUDED.fg3a,
         ftm        = EXCLUDED.ftm,
         fta        = EXCLUDED.fta,
         stl        = EXCLUDED.stl,
         blk        = EXCLUDED.blk,
         tov        = EXCLUDED.tov,
         plus_minus = EXCLUDED.plus_minus,
         wl         = EXCLUDED.wl,
         fetched_at = CURRENT_TIMESTAMP;
""")


def _minutes_to_float(value) -> float | None:
    """Convert '32:45' → 32.75."""
    if pd.isna(value):
        return None
    text_val = str(value)
    if ":" not in text_val:
        try:
            return float(text_val)
        except ValueError:
            return None
    parts = text_val.split(":", maxsplit=1)
    try:
        return float(parts[0]) + float(parts[1]) / 60.0
    except ValueError:
        return None


def upsert_game_logs(raw_df: pd.DataFrame, player_id: int,season: str) -> int:
    """
    Take a raw nba_api game-log DataFrame and upsert every row into Postgres.
    Returns the number of rows upserted.
    """
    if raw_df.empty:
        return 0

    engine = get_engine()
    rows = []

    for _, row in raw_df.iterrows():
        record = {"player_id": player_id, "season": season}
        for api_col, db_col in _LOG_COLS_MAP.items():
            val = row.get(api_col)
            if db_col == "min":
                val = _minutes_to_float(val)
            if db_col == "game_date":
                val = pd.to_datetime(val).date() if pd.notna(val) else None
            if db_col in (
                "pts", "reb", "ast", "fgm", "fga", "fg3m", "fg3a",
                "ftm", "fta", "stl", "blk", "tov", "plus_minus",
            ):
                val = float(val) if pd.notna(val) else None
            record[db_col] = val
        rows.append(record)

    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, rows)

    print(f"✓ Upserted {len(rows)} game logs for player {player_id} ({season})")
    return len(rows)


# ─── Cache reads ──────────────────────────────────────────────

def load_cached_logs(player_id: int, season: str) -> pd.DataFrame:
    """
    Load cached game logs from Postgres.
    Column aliases match raw nba_api output so downstream code
    (clean_player_games, build_features) works unchanged.
    """
    engine = get_engine()
    sql = text("""
        SELECT
            game_id    AS "Game_ID",
            game_date  AS "GAME_DATE",
            matchup    AS "MATCHUP",
            wl         AS "WL",
            min        AS "MIN",
            pts        AS "PTS",
            reb        AS "REB",
            ast        AS "AST",
            fgm        AS "FGM",
            fga        AS "FGA",
            fg3m       AS "FG3M",
            fg3a       AS "FG3A",
            ftm        AS "FTM",
            fta        AS "FTA",
            stl        AS "STL",
            blk        AS "BLK",
            tov        AS "TOV",
            plus_minus AS "PLUS_MINUS"
        FROM player_game_logs
        WHERE player_id = :pid AND season = :season
        ORDER BY game_date;
    """)
    with engine.begin() as conn:
        df = pd.read_sql(sql, conn, params={"pid": player_id, "season": season})
    return df


# ─── Freshness check ─────────────────────────────────────────

def is_cache_fresh(player_id: int, season: str, max_age_hours: float = 20.0,
) -> bool:
    """
    Return True if we have cached data for this player+season
    and the most recent fetch was within max_age_hours.
    """
    engine = get_engine()
    sql = text("""
        SELECT MAX(fetched_at)
        FROM player_game_logs
        WHERE player_id = :pid AND season = :season;
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {"pid": player_id, "season": season}).fetchone()

    if row is None or row[0] is None:
        return False

    last_fetch = row[0]
    if last_fetch.tzinfo is None:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    return last_fetch >= cutoff


# ─── Player ID list ──────────────────────────────────────────

def get_all_player_ids() -> list[int]:
    """Return all player_ids currently in the players table."""
    sql = text("SELECT player_id FROM players ORDER BY player_id;")
    with get_engine().begin() as conn:
        rows = conn.execute(sql).fetchall()
    return [r[0] for r in rows]


# ─── Bulk game log upsert (LeagueGameLog) ────────────────────

_BULK_LOG_COLS_MAP = {
    "PLAYER_ID":  "player_id",
    "GAME_ID":    "game_id",
    "GAME_DATE":  "game_date",
    "MATCHUP":    "matchup",
    "WL":         "wl",
    "MIN":        "min",
    "PTS":        "pts",
    "REB":        "reb",
    "AST":        "ast",
    "FGM":        "fgm",
    "FGA":        "fga",
    "FG3M":       "fg3m",
    "FG3A":       "fg3a",
    "FTM":        "ftm",
    "FTA":        "fta",
    "STL":        "stl",
    "BLK":        "blk",
    "TOV":        "tov",
    "PLUS_MINUS": "plus_minus",
}

_NUMERIC_COLS = {
    "pts", "reb", "ast", "fgm", "fga", "fg3m", "fg3a",
    "ftm", "fta", "stl", "blk", "tov", "plus_minus",
}


def upsert_game_logs_bulk(df: pd.DataFrame, season: str) -> int:
    """
    Upsert all rows from a LeagueGameLog DataFrame (all players, one season).
    PLAYER_ID is read from the DataFrame itself.
    Returns total rows upserted.
    """
    if df.empty:
        return 0

    rows = []
    for _, row in df.iterrows():
        record = {"season": season}
        for api_col, db_col in _BULK_LOG_COLS_MAP.items():
            val = row.get(api_col)
            if db_col == "min":
                val = _minutes_to_float(val)
            elif db_col == "game_date":
                val = pd.to_datetime(val).date() if pd.notna(val) else None
            elif db_col == "player_id":
                val = int(val) if pd.notna(val) else None
            elif db_col in _NUMERIC_COLS:
                val = float(val) if pd.notna(val) else None
            record[db_col] = val
        rows.append(record)

    with get_engine().begin() as conn:
        conn.execute(_UPSERT_SQL, rows)

    print(f"✓ Bulk upserted {len(rows)} game log rows ({season})")
    return len(rows)


# ─── Quick sanity test ────────────────────────────────────────

if __name__ == "__main__":
    ensure_tables()
    print("✓ database.py ready — tables exist in Betting_App")