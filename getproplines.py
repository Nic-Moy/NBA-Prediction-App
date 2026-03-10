"""
Fetch NBA player prop lines from The Odds API.

This file is intentionally standalone so you can use it from the terminal now,
and import the functions later into your model/backtesting scripts.

Key flow (quota-aware):
1) GET /v4/sports/basketball_nba/events          -> free (0 credits)
2) GET /v4/sports/basketball_nba/events/{id}/odds -> paid (credits depend on markets/regions)

Docs used:
- Events endpoint (free)
- Event odds endpoint (player props supported via event-level markets)
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional

import pandas as pd
import requests


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATE_FORMAT = "iso"

# Common NBA player prop markets (examples from The Odds API docs).
COMMON_PLAYER_PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
]


class OddsApiError(RuntimeError):
    """Raised when The Odds API request fails."""


def _get_api_key(explicit_key: Optional[str] = None) -> str:
    """Resolve API key from argument or environment."""
    api_key = explicit_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Odds API key. Pass --api-key or set environment variable ODDS_API_KEY."
        )
    return api_key


def _request_json(path: str, params: dict) -> tuple[object, requests.Response]:
    """Send request to The Odds API and return decoded JSON + response object."""
    url = f"{BASE_URL}{path}"
    response = requests.get(url, params=params, timeout=20)

    if not response.ok:
        detail = response.text.strip()
        raise OddsApiError(
            f"Odds API request failed ({response.status_code}) for {url}: {detail}"
        )

    return response.json(), response


def _usage_from_headers(response: requests.Response) -> dict:
    """
    Extract quota headers if present.

    The Odds API commonly returns headers like:
    - x-requests-remaining
    - x-requests-used
    - x-requests-last
    """
    headers = response.headers
    return {
        "requests_remaining": headers.get("x-requests-remaining"),
        "requests_used": headers.get("x-requests-used"),
        "requests_last": headers.get("x-requests-last"),
    }


def print_usage_info(response: requests.Response) -> None:
    """Print request usage info from response headers (if available)."""
    usage = _usage_from_headers(response)
    if any(v is not None for v in usage.values()):
        print(
            "Usage headers:",
            f"remaining={usage['requests_remaining']}",
            f"used={usage['requests_used']}",
            f"last={usage['requests_last']}",
        )


def list_nba_events(
    api_key: Optional[str] = None,
    *,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> tuple[pd.DataFrame, dict]:
    """
    Fetch upcoming NBA events (games).

    This endpoint is free (0 credits) according to The Odds API docs.
    """
    resolved_key = _get_api_key(api_key)
    payload, response = _request_json(
        f"/sports/{SPORT_KEY}/events",
        {"apiKey": resolved_key, "dateFormat": date_format},
    )

    events = pd.DataFrame(payload)
    if not events.empty:
        # Normalize date for easier reading
        if "commence_time" in events.columns:
            events["commence_time"] = pd.to_datetime(events["commence_time"], errors="coerce")
        sort_cols = [c for c in ["commence_time", "home_team", "away_team"] if c in events.columns]
        if sort_cols:
            events = events.sort_values(sort_cols).reset_index(drop=True)

    return events, _usage_from_headers(response)


def fetch_event_prop_odds(
    event_id: str,
    *,
    api_key: Optional[str] = None,
    markets: Iterable[str] = ("player_points",),
    regions: str = DEFAULT_REGIONS,
    odds_format: str = DEFAULT_ODDS_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    bookmakers: Optional[Iterable[str]] = None,
) -> tuple[dict, dict]:
    """
    Fetch event-level odds for one NBA game, including player props.

    Example market keys:
    - player_points
    - player_rebounds
    - player_assists
    - player_threes
    - player_points_rebounds_assists
    """
    resolved_key = _get_api_key(api_key)
    market_list = [m.strip() for m in markets if str(m).strip()]
    if not market_list:
        raise ValueError("At least one market key must be provided.")

    params = {
        "apiKey": resolved_key,
        "regions": regions,
        "markets": ",".join(market_list),
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        bookmaker_list = [b.strip() for b in bookmakers if str(b).strip()]
        if bookmaker_list:
            params["bookmakers"] = ",".join(bookmaker_list)

    payload, response = _request_json(
        f"/sports/{SPORT_KEY}/events/{event_id}/odds",
        params,
    )
    return payload, _usage_from_headers(response)


def normalize_prop_odds(event_payload: dict) -> pd.DataFrame:
    """
    Flatten event odds payload into one row per outcome.

    For player props, outcomes are typically:
    - name: 'Over' or 'Under'
    - description: player name
    - point: line value
    - price: odds
    """
    rows: List[dict] = []

    event_meta = {
        "event_id": event_payload.get("id"),
        "sport_key": event_payload.get("sport_key"),
        "sport_title": event_payload.get("sport_title"),
        "commence_time": event_payload.get("commence_time"),
        "home_team": event_payload.get("home_team"),
        "away_team": event_payload.get("away_team"),
    }

    for bookmaker in event_payload.get("bookmakers", []) or []:
        bookmaker_key = bookmaker.get("key")
        bookmaker_title = bookmaker.get("title")
        bookmaker_last_update = bookmaker.get("last_update")

        for market in bookmaker.get("markets", []) or []:
            market_key = market.get("key")
            market_last_update = market.get("last_update")

            for outcome in market.get("outcomes", []) or []:
                rows.append(
                    {
                        **event_meta,
                        "bookmaker_key": bookmaker_key,
                        "bookmaker_title": bookmaker_title,
                        "bookmaker_last_update": bookmaker_last_update,
                        "market_key": market_key,
                        "market_last_update": market_last_update,
                        # For player props this is usually Over/Under
                        "side": outcome.get("name"),
                        # For player props this is usually the player name
                        "player_name": outcome.get("description"),
                        "line": outcome.get("point"),
                        "price": outcome.get("price"),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["commence_time", "bookmaker_last_update", "market_last_update"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].fillna("")

    return df


def pair_over_under_rows(prop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one-row-per-outcome to one-row-per-line with over/under prices side by side.
    """
    if prop_df.empty:
        return prop_df.copy()

    base_cols = [
        "event_id",
        "commence_time",
        "home_team",
        "away_team",
        "bookmaker_key",
        "bookmaker_title",
        "market_key",
        "player_name",
        "line",
    ]

    available_base_cols = [c for c in base_cols if c in prop_df.columns]
    if not {"side", "price"}.issubset(prop_df.columns):
        return prop_df.copy()

    paired = (
        prop_df.pivot_table(
            index=available_base_cols,
            columns="side",
            values="price",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    rename_map = {}
    if "Over" in paired.columns:
        rename_map["Over"] = "over_price"
    if "Under" in paired.columns:
        rename_map["Under"] = "under_price"
    paired = paired.rename(columns=rename_map)

    sort_cols = [c for c in ["commence_time", "player_name", "bookmaker_title", "market_key"] if c in paired.columns]
    if sort_cols:
        paired = paired.sort_values(sort_cols)

    return paired.reset_index(drop=True)


def filter_player_rows(prop_df: pd.DataFrame, player_query: str) -> pd.DataFrame:
    """Case-insensitive contains match on player_name."""
    if prop_df.empty:
        return prop_df.copy()
    if "player_name" not in prop_df.columns:
        return prop_df.iloc[0:0].copy()

    mask = prop_df["player_name"].str.contains(player_query, case=False, na=False)
    return prop_df.loc[mask].copy()


def find_player_props_across_events(
    player_name: str,
    *,
    api_key: Optional[str] = None,
    markets: Iterable[str] = ("player_points",),
    regions: str = DEFAULT_REGIONS,
    bookmakers: Optional[Iterable[str]] = None,
    odds_format: str = DEFAULT_ODDS_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    print_quota: bool = True,
) -> pd.DataFrame:
    """
    Find a player's prop lines across all upcoming NBA events.

    Quota note:
    - events list call is free
    - each event odds call consumes credits (depends on markets/regions)
    """
    events_df, events_usage = list_nba_events(api_key=api_key, date_format=date_format)
    if print_quota:
        print("Fetched NBA events (free endpoint).", events_usage)

    if events_df.empty:
        return pd.DataFrame()

    all_rows: List[pd.DataFrame] = []
    resolved_key = _get_api_key(api_key)

    for _, event in events_df.iterrows():
        event_id = str(event["id"])
        try:
            payload, usage = fetch_event_prop_odds(
                event_id,
                api_key=resolved_key,
                markets=markets,
                regions=regions,
                odds_format=odds_format,
                date_format=date_format,
                bookmakers=bookmakers,
            )
            if print_quota:
                matchup = f"{event.get('away_team', 'Away')} @ {event.get('home_team', 'Home')}"
                print(f"Fetched props for {matchup} | event_id={event_id} | usage={usage}")
        except OddsApiError as exc:
            print(f"Skipping event {event_id}: {exc}")
            continue

        event_df = normalize_prop_odds(payload)
        if event_df.empty:
            continue

        player_df = filter_player_rows(event_df, player_name)
        if not player_df.empty:
            all_rows.append(player_df)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def _parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch NBA player prop lines from The Odds API."
    )
    parser.add_argument("--api-key", help="Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument(
        "--event-id",
        help="Fetch props for one specific NBA event ID (saves credits vs scanning all games if you already know it).",
    )
    parser.add_argument(
        "--player",
        help="Filter results to a player name (case-insensitive contains match).",
    )
    parser.add_argument(
        "--markets",
        default="player_points",
        help=(
            "Comma-separated market keys. Examples: "
            "player_points,player_rebounds,player_assists,player_threes,player_points_rebounds_assists"
        ),
    )
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help="Odds API regions param (default: us)")
    parser.add_argument(
        "--bookmakers",
        help="Optional comma-separated bookmakers to limit payload (ex: fanduel,draftkings,betmgm)",
    )
    parser.add_argument(
        "--odds-format",
        default=DEFAULT_ODDS_FORMAT,
        choices=["american", "decimal"],
        help="Odds format (default: american)",
    )
    parser.add_argument(
        "--date-format",
        default=DEFAULT_DATE_FORMAT,
        choices=["iso", "unix"],
        help="Date format (default: iso)",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Show paired over/under rows (one row per line/bookmaker/player).",
    )
    parser.add_argument(
        "--list-events",
        action="store_true",
        help="List upcoming NBA event IDs only (free endpoint).",
    )
    parser.add_argument(
        "--show-common-markets",
        action="store_true",
        help="Print a short list of common NBA player prop market keys and exit.",
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.show_common_markets:
        print("Common NBA player prop markets:")
        for market in COMMON_PLAYER_PROP_MARKETS:
            print(f"- {market}")
        return

    api_key = _get_api_key(args.api_key)
    markets = _parse_csv_arg(args.markets) or ["player_points"]
    bookmakers = _parse_csv_arg(args.bookmakers)

    if args.list_events:
        events_df, usage = list_nba_events(api_key=api_key, date_format=args.date_format)
        print("Upcoming NBA events (free endpoint)")
        print(f"Usage headers: {usage}")
        if events_df.empty:
            print("No events returned.")
            return

        display_cols = [c for c in ["id", "commence_time", "home_team", "away_team"] if c in events_df.columns]
        print(events_df[display_cols].to_string(index=False))
        return

    if args.event_id:
        payload, usage = fetch_event_prop_odds(
            args.event_id,
            api_key=api_key,
            markets=markets,
            regions=args.regions,
            odds_format=args.odds_format,
            date_format=args.date_format,
            bookmakers=bookmakers,
        )
        print(f"Usage headers: {usage}")
        prop_df = normalize_prop_odds(payload)
        if args.player:
            prop_df = filter_player_rows(prop_df, args.player)
    else:
        if not args.player:
            parser.error("Provide --player when scanning across events, or use --event-id.")
            return

        prop_df = find_player_props_across_events(
            args.player,
            api_key=api_key,
            markets=markets,
            regions=args.regions,
            bookmakers=bookmakers,
            odds_format=args.odds_format,
            date_format=args.date_format,
            print_quota=True,
        )

    if prop_df.empty:
        print("No prop lines found for the given query.")
        return

    output_df = pair_over_under_rows(prop_df) if args.paired else prop_df

    # Keep output readable in terminal.
    preferred_cols = [
        "commence_time",
        "away_team",
        "home_team",
        "bookmaker_title",
        "market_key",
        "player_name",
        "line",
        "over_price",
        "under_price",
        "side",
        "price",
    ]
    final_cols = [c for c in preferred_cols if c in output_df.columns]
    other_cols = [c for c in output_df.columns if c not in final_cols]
    output_df = output_df[final_cols + other_cols]

    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()
