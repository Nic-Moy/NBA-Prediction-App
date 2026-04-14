"""
Fetch player prop lines from PrizePicks for any sport.

PrizePicks is a DFS platform that publishes player projections (Over/Under lines)
across many sports. Their API is publicly accessible with no API key required.

NBA stat aliases:
    pts  -> Points
    reb  -> Rebounds
    ast  -> Assists
    3pm  -> 3-PT Made
    pra  -> Pts+Rebs+Asts
    pr   -> Pts+Rebs
    pa   -> Pts+Asts

Key flow:
1) GET https://api.prizepicks.com/projections?league_id=<id>&per_page=250
   Returns all current projections in JSON:API format.
2) Join `data` (projections) with `included` (player metadata) on player ID.
3) Filter by stat type and/or player name as needed.

Importable convenience functions:
    from getprizepicks import get_props, get_nba_props
    df = get_props("NHL", player="McDavid")
    df = get_nba_props(stats=["pts", "pra"], player="LeBron")
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Optional, Union

import pandas as pd
from curl_cffi import requests


BASE_URL = "https://api.prizepicks.com"
DEFAULT_PER_PAGE = 250

# Friendly name -> PrizePicks league_id
LEAGUES: dict[str, int] = {
    "NBA":    7,
    "MLB":    2,
    "NHL":    8,
    "NFL":    9,
    "WNBA":   3,
    "CBB":   20,
    "CFB":   15,
    "PGA":    1,
    "SOCCER": 82,
    "TENNIS": 5,
    "MMA":   12,
    "BOXING": 42,
}

# Kept for backward compatibility
NBA_LEAGUE_ID = LEAGUES["NBA"]

# NBA short alias -> PrizePicks stat_type string
STAT_ALIASES: dict[str, str] = {
    "pts": "Points",
    "reb": "Rebounds",
    "ast": "Assists",
    "3pm": "3-PT Made",
    "pra": "Pts+Rebs+Asts",
    "pr":  "Pts+Rebs",
    "pa":  "Pts+Asts",
}

# NBA default stat filter (used by get_nba_props and CLI when no --stats given)
NBA_DEFAULT_STATS = ["Points", "Rebounds", "Assists", "Pts+Rebs+Asts", "Pts+Rebs", "Pts+Asts"]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://app.prizepicks.com/",
}


class PrizePicksError(RuntimeError):
    """Raised when a PrizePicks API request fails."""


def _resolve_league_id(league: Union[str, int]) -> int:
    """Accept a league name (e.g. 'NBA') or raw integer ID and return the ID."""
    if isinstance(league, int):
        return league
    upper = league.strip().upper()
    if upper not in LEAGUES:
        known = ", ".join(sorted(LEAGUES))
        raise ValueError(f"Unknown league '{league}'. Known leagues: {known}")
    return LEAGUES[upper]


# ---------------------------------------------------------------------------
# Core fetch + parse
# ---------------------------------------------------------------------------

def fetch_projections(
    league: Union[str, int] = "NBA",
    per_page: int = DEFAULT_PER_PAGE,
) -> dict:
    """Fetch raw projections JSON from PrizePicks. No API key required.

    Args:
        league:   League name (e.g. 'NBA', 'MLB', 'NHL') or raw league_id integer.
        per_page: Max rows per request (default 250 covers most leagues in one call).
    """
    league_id = _resolve_league_id(league)
    url = f"{BASE_URL}/projections"
    params = {"league_id": league_id, "per_page": per_page}
    response = requests.get(url, params=params, headers=_HEADERS, timeout=20, impersonate="chrome120")
    if not response.ok:
        raise PrizePicksError(
            f"PrizePicks request failed ({response.status_code}): {response.text.strip()}"
        )
    return response.json()


def parse_projections(raw: dict) -> pd.DataFrame:
    """
    Flatten PrizePicks JSON:API response into a one-row-per-projection DataFrame.

    PrizePicks returns two top-level keys:
    - raw["data"]:     list of projection objects (line_score, stat_type, etc.)
    - raw["included"]: related objects; filter type == "new_player" for player metadata.
    """
    data = raw.get("data", [])
    included = raw.get("included", [])

    # Build player_id -> metadata lookup from included objects
    player_lookup: dict[str, dict] = {}
    for item in included:
        if item.get("type") == "new_player":
            attrs = item.get("attributes", {})
            player_lookup[item["id"]] = {
                "player_name": attrs.get("name", ""),
                "team": attrs.get("team_name") or attrs.get("team", ""),
                "position": attrs.get("position", ""),
            }

    rows: List[dict] = []
    for proj in data:
        if proj.get("type") != "projection":
            continue

        attrs = proj.get("attributes", {})

        # Resolve linked player
        player_id = (
            proj.get("relationships", {})
            .get("new_player", {})
            .get("data", {})
            .get("id", "")
        )
        player_info = player_lookup.get(player_id, {})

        rows.append({
            "projection_id": proj.get("id"),
            "player_name":   player_info.get("player_name", ""),
            "team":          player_info.get("team", ""),
            "position":      player_info.get("position", ""),
            "stat_type":     attrs.get("stat_type", ""),
            "line":          attrs.get("line_score"),
            "tier":          attrs.get("odds_type", "standard"),
            "start_time":    attrs.get("start_time"),
            "status":        attrs.get("status", ""),
            "is_promo":      attrs.get("is_promo", False),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.sort_values(["start_time", "player_name", "stat_type"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def resolve_stat_aliases(stat_args: List[str]) -> List[str]:
    """Map NBA short aliases (pts, pra, ...) to PrizePicks stat_type strings.

    Unknown strings are passed through unchanged so you can also pass full
    PrizePicks names directly (e.g. "Points", "Hits", "Goals").
    """
    return [STAT_ALIASES.get(s.strip().lower(), s.strip()) for s in stat_args]


def filter_by_stats(df: pd.DataFrame, stats: Iterable[str]) -> pd.DataFrame:
    """Keep only rows whose stat_type matches one of the provided values."""
    if df.empty:
        return df.copy()
    targets = {s.lower() for s in stats}
    mask = df["stat_type"].str.lower().isin(targets)
    return df.loc[mask].copy()


def filter_by_player(df: pd.DataFrame, player_query: str) -> pd.DataFrame:
    """Case-insensitive substring match on player_name."""
    if df.empty or not player_query:
        return df.copy()
    mask = df["player_name"].str.contains(player_query.strip(), case=False, na=False)
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Convenience functions for importing into other modules
# ---------------------------------------------------------------------------

def get_props(
    league: Union[str, int] = "NBA",
    stats: Optional[Iterable[str]] = None,
    player: Optional[str] = None,
    include_promos: bool = False,
) -> pd.DataFrame:
    """Fetch and filter current props from PrizePicks for any league.

    Args:
        league:         League name ('NBA', 'MLB', 'NHL', etc.) or raw league_id.
        stats:          Stat types to include. For NBA accepts short aliases
                        (pts, reb, ast, pra, pr, pa, 3pm). For other leagues use
                        the full PrizePicks stat_type string (e.g. 'Goals', 'Hits').
                        Defaults to all stat types for the league.
        player:         Optional player name filter (case-insensitive substring).
        include_promos: Whether to include promotional lines (default False).

    Returns:
        DataFrame with columns:
            player_name, team, position, stat_type, line, tier, start_time, status
    """
    raw = fetch_projections(league)
    df = parse_projections(raw)

    if df.empty:
        return df

    if not include_promos:
        df = df.loc[~df["is_promo"]].copy()

    if stats is not None:
        stat_list = resolve_stat_aliases(list(stats))
        df = filter_by_stats(df, stat_list)

    if player:
        df = filter_by_player(df, player)

    return df.reset_index(drop=True)


def get_nba_props(
    stats: Optional[Iterable[str]] = None,
    player: Optional[str] = None,
    include_promos: bool = False,
) -> pd.DataFrame:
    """Fetch and filter current NBA props (convenience wrapper for get_props).

    Defaults to the six main markets: pts, reb, ast, pra, pr, pa.
    Pass stats=None to get all NBA stat types.
    """
    default = NBA_DEFAULT_STATS if stats is None else None
    return get_props(
        league="NBA",
        stats=stats if stats is not None else default,
        player=player,
        include_promos=include_promos,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    league_choices = sorted(LEAGUES)
    parser = argparse.ArgumentParser(
        description="Fetch player prop lines from PrizePicks (no API key required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "NBA stat aliases:\n"
            "  pts  -> Points\n"
            "  reb  -> Rebounds\n"
            "  ast  -> Assists\n"
            "  3pm  -> 3-PT Made\n"
            "  pra  -> Pts+Rebs+Asts\n"
            "  pr   -> Pts+Rebs\n"
            "  pa   -> Pts+Asts\n"
            "\n"
            f"Available leagues: {', '.join(league_choices)}\n"
        ),
    )
    parser.add_argument(
        "--league",
        default="NBA",
        metavar="LEAGUE",
        help=f"Sport league. One of: {', '.join(league_choices)}. Default: NBA",
    )
    parser.add_argument(
        "--player",
        help="Filter to a player name (case-insensitive substring match).",
    )
    parser.add_argument(
        "--stats",
        default=None,
        help=(
            "Comma-separated stat types. NBA accepts aliases (pts,reb,ast,pra,pr,pa,3pm). "
            "Other leagues use full PrizePicks stat names. Defaults to all stats."
        ),
    )
    parser.add_argument(
        "--all-stats",
        action="store_true",
        help="Show every stat type returned (ignores --stats filter).",
    )
    parser.add_argument(
        "--include-promos",
        action="store_true",
        help="Include promotional lines (filtered out by default).",
    )
    parser.add_argument(
        "--list-stats",
        action="store_true",
        help="Print all unique stat types on the board for this league and exit.",
    )
    parser.add_argument(
        "--list-leagues",
        action="store_true",
        help="Print all supported league names and exit.",
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.list_leagues:
        print("Supported leagues:")
        for name, lid in sorted(LEAGUES.items()):
            print(f"  {name:<10} (id={lid})")
        return

    raw = fetch_projections(args.league)
    df = parse_projections(raw)

    if df.empty:
        print(f"No projections returned for {args.league}.")
        return

    if args.list_stats:
        print(f"Stat types on the board for {args.league}:")
        for stat in sorted(df["stat_type"].unique()):
            print(f"  {stat}")
        return

    if not args.include_promos:
        df = df.loc[~df["is_promo"]].copy()

    if not args.all_stats:
        if args.stats:
            stat_list = resolve_stat_aliases([s.strip() for s in args.stats.split(",")])
            df = filter_by_stats(df, stat_list)
        elif args.league.upper() == "NBA":
            df = filter_by_stats(df, NBA_DEFAULT_STATS)

    if args.player:
        df = filter_by_player(df, args.player)

    if df.empty:
        print("No matching props found.")
        return

    display_cols = [
        c for c in ["player_name", "team", "position", "stat_type", "line", "tier", "start_time", "status"]
        if c in df.columns
    ]
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
