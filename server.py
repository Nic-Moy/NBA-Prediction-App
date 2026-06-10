"""
PROPSDESK — FastAPI + HTMX frontend for the NBA props model.

Run with:
    uvicorn server:app --reload --port 8000

Reuses the existing scraping/model stack unchanged:
    getprizepicks.get_props / get_nba_props   -> live PrizePicks lines
    getplayerinfo / database / model          -> next-game prediction
"""

from __future__ import annotations

import time
import datetime
import traceback
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from getprizepicks import get_props, get_nba_props, LEAGUES
from getplayerinfo import find_player_id, clean_player_games, DEFAULT_SEASON
from database import load_cached_logs
from model import build_features, predict_next_game, DEFAULT_MIN_TRAIN_SIZE

app = FastAPI(title="PROPSDESK")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Stat mapping: PrizePicks stat_type / alias  <->  game-log column
# ---------------------------------------------------------------------------

PREDICT_STATS = {
    # game-log col : (display label, prizepicks stat_type, get_nba_props alias)
    "PTS": ("Points", "Points", "pts"),
    "REB": ("Rebounds", "Rebounds", "reb"),
    "AST": ("Assists", "Assists", "ast"),
}

# ---------------------------------------------------------------------------
# Props cache — replaces Streamlit's @st.cache_data(ttl=300)
# ---------------------------------------------------------------------------

_PROPS_TTL = 300  # seconds
_props_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def load_props(league: str, *, refresh: bool = False) -> pd.DataFrame:
    """Fetch + cache non-promo props for a league (5 min TTL)."""
    now = time.time()
    if not refresh and league in _props_cache:
        ts, df = _props_cache[league]
        if now - ts < _PROPS_TTL:
            return df
    df = get_props(league=league, include_promos=False)
    df = df.reset_index(drop=True) if not df.empty else df
    _props_cache[league] = (now, df)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(value) -> str:
    """Format a start_time value to a short local clock string."""
    if value is None or pd.isna(value):
        return "TBD"
    try:
        ts = pd.Timestamp(value)
        return ts.strftime("%-I:%M %p")
    except Exception:
        return str(value)


def stats_for_league(league: str) -> list[str]:
    """All stat_types offered for a league, most-bet first (alpha tiebreak)."""
    try:
        df = load_props(league)
    except Exception:  # noqa: BLE001 — fetch/bot-block; caller renders empty chips
        return []
    if df.empty or "stat_type" not in df.columns:
        return []
    counts = df["stat_type"].dropna().value_counts()
    # value_counts is already count-desc; stabilize ties alphabetically.
    return sorted(counts.index.tolist(), key=lambda s: (-int(counts[s]), s))


def _records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        out.append(
            {
                "player_name": row.get("player_name", "—"),
                "team": row.get("team") or "",
                "position": row.get("position") or "",
                "stat_type": row.get("stat_type", "—"),
                "line": row.get("line"),
                "tier": (row.get("tier") or "standard"),
                "start_time": _fmt_time(row.get("start_time")),
                "status": row.get("status") or "",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "leagues": sorted(LEAGUES),
            "default_league": "NBA" if "NBA" in LEAGUES else sorted(LEAGUES)[0],
            "predict_stats": PREDICT_STATS,
        },
    )


@app.get("/api/stats", response_class=HTMLResponse)
def api_stats(request: Request, league: str = "NBA"):
    """HTMX partial: stat filter chips for the selected league.

    Emits HX-Trigger=propsReload so finishing the chip swap kicks a props fetch.
    """
    return templates.TemplateResponse(
        request,
        "partials/_stat_chips.html",
        {"request": request, "stats": stats_for_league(league)},
        headers={"HX-Trigger": "propsReload"},
    )


@app.get("/api/props", response_class=HTMLResponse)
def api_props(
    request: Request,
    league: str = "NBA",
    player: str = "",
    stats: Optional[list[str]] = Query(None),
    tier: Optional[list[str]] = Query(None),
    refresh: int = 0,
):
    """HTMX partial: filtered prop cards. Filters mirror the old app.py logic."""
    try:
        df = load_props(league, refresh=bool(refresh))
    except Exception as exc:  # scraping failed
        traceback.print_exc()
        return templates.TemplateResponse(
            request,
            "partials/_props.html",
            {"request": request, "props": [], "count": 0, "error": str(exc),
             "fetched_at": ""},
        )

    all_stats = sorted(df["stat_type"].dropna().unique()) if not df.empty else []

    if not df.empty:
        filtered = df.copy()

        if player.strip():
            filtered = filtered[
                filtered["player_name"].str.contains(player.strip(), case=False, na=False)
            ]

        selected_stats = [s for s in (stats or []) if s]
        if selected_stats:
            filtered = filtered[filtered["stat_type"].isin(selected_stats)]

        selected_tiers = [t for t in (tier or []) if t]
        if selected_tiers:
            filtered = filtered[filtered["tier"].isin(selected_tiers)]
    else:
        filtered = df

    props = _records(filtered)
    fetched_at = datetime.datetime.now().strftime("%-I:%M %p")

    return templates.TemplateResponse(
        request,
        "partials/_props.html",
        {
            "request": request,
            "props": props,
            "count": len(props),
            "all_stats": all_stats,
            "error": None,
            "fetched_at": fetched_at,
            "league": league,
        },
    )


@app.get("/api/predict", response_class=HTMLResponse)
def api_predict(
    request: Request,
    player: str = "",
    opponent: str = "",
    stat: str = "PTS",
    season: str = DEFAULT_SEASON,
):
    """HTMX partial: next-game prediction vs real PrizePicks line."""
    ctx: dict = {"request": request, "player": player, "stat": stat,
                 "opponent": opponent.upper()}

    def fail(message: str, kind: str = "info"):
        ctx.update({"state": kind, "message": message})
        return templates.TemplateResponse(request, "partials/_prediction.html", ctx)

    player = player.strip()
    if not player:
        return fail("Enter a player name to run a projection.", "empty")

    stat_col = stat if stat in PREDICT_STATS else "PTS"
    label, pp_stat_type, alias = PREDICT_STATS[stat_col]

    try:
        player_id = find_player_id(player)
        if player_id is None:
            return fail(f"Player '{player}' not found. Check spelling.", "error")

        raw_df = load_cached_logs(player_id, season)
        if raw_df.empty:
            return fail(
                f"No cached game logs for {player} ({season}). "
                "Run setup.py to load logs into the database.",
                "error",
            )

        clean_df = clean_player_games(raw_df)
        feature_df = build_features(clean_df, season=season)

        opp = opponent.strip().upper() or None
        prediction = predict_next_game(
            feature_df,
            stat_col=stat_col,
            min_train_size=DEFAULT_MIN_TRAIN_SIZE,
            next_opponent_abbrev=opp,
            season=season,
        )

        predicted = prediction.get("predicted")
        moe = prediction.get("margin_of_error")
        if predicted is None:
            return fail(
                f"Not enough game history for {player} to project {label}.",
                "error",
            )

        # Real PrizePicks line (standard tier) for the over/under call.
        line = None
        line_note = None
        try:
            props_df = get_nba_props(stats=[alias], player=player)
            standard = props_df[props_df["tier"] == "standard"] if not props_df.empty else props_df
            if standard.empty:
                line_note = "No standard line posted for this player today."
            else:
                line = float(standard["line"].iloc[0])
        except Exception as exc:  # noqa: BLE001
            line_note = f"Couldn't fetch live line ({exc})."

        rec = None
        if line is not None:
            margin = predicted - line
            direction = "OVER" if margin >= 0 else "UNDER"
            abs_margin = abs(margin)
            if abs_margin < 2.0:
                confidence = "Low"
            elif abs_margin < 4.0:
                confidence = "Moderate"
            else:
                confidence = "High"
            rec = {
                "direction": direction,
                "margin": round(margin, 1),
                "abs_margin": round(abs_margin, 1),
                "confidence": confidence,
            }

        ctx.update(
            {
                "state": "ok",
                "label": label,
                "predicted": predicted,
                "moe": moe,
                "line": line,
                "line_note": line_note,
                "rec": rec,
                "opponent": opp or "",
            }
        )
        return templates.TemplateResponse(request, "partials/_prediction.html", ctx)

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return fail(f"Prediction failed: {exc}", "error")
