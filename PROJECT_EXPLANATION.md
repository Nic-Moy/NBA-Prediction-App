# NBA Player Props — Project Documentation

This project predicts NBA player prop bets using historical game logs, machine learning, and real-time odds data. It is built in Python with a PostgreSQL cache, the `nba_api` library for stats, and The Odds API for live betting lines.

---

## Table of Contents
1. [Database](#1-database)
2. [Getting Player Info](#2-getting-player-info)
3. [ML Model](#3-ml-model)
4. [Getting Prop Lines](#4-getting-prop-lines)
5. [Main File](#5-main-file)

---

## 1. Database

**Files:** `database.py`, `setup.py`

### Overview
`database.py` is the PostgreSQL abstraction layer for the entire project. It uses SQLAlchemy to connect to a local Postgres database (`Betting_App`) and provides functions for caching NBA player data so the app avoids making redundant API calls every time it runs.

`setup.py` is the entry point for initially populating that database. It calls functions from `getplayerinfo.py` to pull data from the NBA API and write it into the database tables.

### Database Tables

| Table | Purpose |
|---|---|
| `players` | Stores player metadata: name, ID, team, position, active status |
| `teams` | Stores all 30 NBA teams: name, city, abbreviation |
| `player_game_logs` | Game-by-game stats for each player and season |

The `player_game_logs` table is indexed on `(player_id, season)` and `game_date` for fast lookups.

### Key Functions in `database.py`

- **`get_engine()`** — Returns a singleton SQLAlchemy database engine. Creates the connection on first call and reuses it afterward.
- **`ensure_tables()`** — Creates all four tables if they don't already exist. Safe to call repeatedly.
- **`cache_player(player_id, full_name, is_active)`** — Inserts or updates a single player record.
- **`cache_team(team_id, full_name, abbreviation, city, nickname)`** — Inserts or updates a team record.
- **`upsert_game_logs(raw_df, player_id, season)`** — Inserts game log rows for one player, skipping duplicates.
- **`upsert_game_logs_bulk(df, season)`** — Bulk upserts game logs for the entire league at once.
- **`load_cached_logs(player_id, season)`** — Retrieves a player's cached game logs as a pandas DataFrame.
- **`is_cache_fresh(player_id, season, max_age_hours=20)`** — Returns `True` if the data was fetched within the last 20 hours, preventing unnecessary re-fetching.
- **`get_all_player_ids()`** — Returns a list of all player IDs currently stored in the database.

### How `setup.py` Works
When run, `setup.py` asks you to choose between two loading modes:
- **Choice 1** — Loads all active NBA players, all 30 teams, and bulk game logs for the entire league (fast, ~seconds via the `LeagueGameLog` endpoint).
- **Choice 2** — Loads all players and teams, then prompts you for a single player name and loads only their game logs.

You must run `setup.py` before using `main.py`, as the model depends on cached data being in the database.

### Terminal Commands

```bash
# Initial setup — populate the database (run this first)
python setup.py

# Verify database schema (creates tables if missing, prints confirmation)
python database.py
```

---

## 2. Getting Player Info

**File:** `getplayerinfo.py`

### Overview
`getplayerinfo.py` is the NBA API wrapper for the project. It handles everything related to fetching player data from the `nba_api` library — resolving player names to IDs, pulling game logs, and writing that data into the PostgreSQL cache.

### Key Functions

- **`find_player_id(name)`** — Takes a player's name (e.g. `"Stephen Curry"`) and returns their NBA player ID by searching the `nba_api` static player list. Returns `None` if the player is not found.
- **`get_player_stats(player_id, season)`** — Fetches raw game logs for a player from the `PlayerGameLog` endpoint on the NBA stats API.
- **`clean_player_games(raw_df)`** — Normalizes the raw DataFrame: renames columns, converts minutes from `"MM:SS"` format to a decimal float (e.g. `"32:45"` → `32.75`), and sorts games by date ascending.
- **`load_all_players_to_db(active_only=True)`** — Fetches all NBA players from the static list and inserts them into the `players` table. Defaults to active players only.
- **`load_all_teams_to_db()`** — Fetches all 30 NBA teams and inserts them into the `teams` table.
- **`load_player_game_logs(player_name, season)`** — Fetches and caches game logs for a single named player. Includes retry logic to handle API rate limits.
- **`load_all_game_logs_bulk(season)`** — Fetches game logs for every player in the league in a single API call using the `LeagueGameLog` endpoint. Much faster than fetching player-by-player.

### CLI Arguments

| Flag | Description | Default |
|---|---|---|
| `--player` | Player full name (e.g. `"Stephen Curry"`) | Required |
| `--season` | NBA season string | `2025-26` |
| `--rows` | Number of recent games to display | `10` |

### Terminal Commands

```bash
# Fetch and display a player's recent game logs
python getplayerinfo.py --player "Stephen Curry"

# Specify a season and number of rows to show
python getplayerinfo.py --player "LeBron James" --season 2024-25 --rows 5
```

---

## 3. ML Model

**File:** `model.py`

### Overview
`model.py` contains all the feature engineering and machine learning logic. It takes a cleaned game log DataFrame, builds pre-game features that avoid data leakage, trains a Logistic Regression classifier using a walk-forward backtest approach, and evaluates whether the model beats the sports betting break-even threshold.

### Feature Engineering — `build_features(df)`

Seven features are built from the game log data. All rolling averages are **shifted by one game** so they only use information available before the game being predicted.

| Feature | Description |
|---|---|
| `is_home` | `1` if the player's team is playing at home, `0` for away |
| `rest_days` | Number of days since the previous game |
| `pts_avg_3` | Rolling 3-game average points (shifted, no leakage) |
| `pts_avg_5` | Rolling 5-game average points (shifted) |
| `reb_avg_3` | Rolling 3-game average rebounds (shifted) |
| `ast_avg_3` | Rolling 3-game average assists (shifted) |
| `min_avg_3` | Rolling 3-game average minutes played (shifted) |
| `line_proxy_pts` | 5-game average points used as a proxy for the betting line |
| `target_over_pts` | Label: `1` if actual points exceeded the line proxy, `0` otherwise |

### Walk-Forward Backtest — `walk_forward_backtest(df, lookback_games=15)`

The backtest simulates real-world prediction: for each game in the season (starting at game 16), it trains a fresh Logistic Regression model on the previous 15 games and predicts whether the player will go Over or Under the line proxy for the current game.

This approach prevents "future leakage" — the model never sees data from games it hasn't played yet.

### Evaluation — `evaluate_results(results_df)`

Results are measured against the **52.4% break-even threshold** — the win rate needed to be profitable at standard -110 odds. The evaluator reports:
- Overall prediction accuracy
- Accuracy on high-confidence picks (model probability >= 60%)
- A sample prediction table

### Terminal Commands

```bash
# Run the model interactively (prompts for player name and season)
python model.py
```

---

## 4. Getting Prop Lines

**File:** `getproplines.py`

### Overview
`getproplines.py` connects to The Odds API to fetch real NBA player prop betting lines. It pulls odds from multiple sportsbooks, flattens the nested JSON response into a clean DataFrame, and can optionally pair Over and Under lines into a single row for easy comparison.

This module is standalone — it does not interact with the database or the ML model.

### Key Functions

- **`list_nba_events(api_key)`** — Fetches all upcoming NBA games. This is a free endpoint that does not consume API credits.
- **`fetch_event_prop_odds(event_id, api_key, markets, ...)`** — Fetches prop odds for a specific game by event ID. Costs API credits.
- **`normalize_prop_odds(event_payload)`** — Flattens the raw API JSON into a flat DataFrame with one row per player/market/bookmaker/outcome.
- **`pair_over_under_rows(prop_df)`** — Merges Over and Under rows into a single row with `over_price` and `under_price` columns side by side.
- **`filter_player_rows(prop_df, player_query)`** — Filters the DataFrame to rows matching a player name (case-insensitive).
- **`find_player_props_across_events(player_name, api_key, ...)`** — Scans all upcoming NBA games and returns every prop line found for the given player.

### CLI Arguments

| Flag | Description | Default |
|---|---|---|
| `--player` | Player name to search for | Required (unless `--list-events`) |
| `--api-key` | Odds API key (or set `ODDS_API_KEY` env var) | — |
| `--event-id` | Query a specific game by event ID (saves credits) | — |
| `--markets` | Comma-separated prop markets to fetch | `player_points` |
| `--regions` | Odds regions | `us` |
| `--bookmakers` | Specific sportsbooks (e.g. `fanduel,draftkings`) | All |
| `--odds-format` | `american` or `decimal` | `american` |
| `--paired` | Show Over/Under merged into one row | `False` |
| `--list-events` | List upcoming NBA games only (free, no credits used) | `False` |
| `--show-common-markets` | Print all supported prop market keys | `False` |

### Supported Markets

| Market Key | Description |
|---|---|
| `player_points` | Points scored |
| `player_rebounds` | Total rebounds |
| `player_assists` | Assists |
| `player_threes` | Three-pointers made |
| `player_points_rebounds_assists` | Combo PRA prop |

### Terminal Commands

```bash
# List all upcoming NBA games (free, no API credits used)
python getproplines.py --list-events

# Fetch points prop lines for a player (paired over/under view)
python getproplines.py --player "LeBron James" --markets player_points --paired

# Fetch multiple prop markets for a player
python getproplines.py --player "Stephen Curry" --markets player_points,player_rebounds,player_assists

# Filter to specific sportsbooks
python getproplines.py --player "Nikola Jokic" --markets player_points --bookmakers fanduel,draftkings --paired

# Show all supported market keys
python getproplines.py --show-common-markets
```

> **Note:** Set your API key as an environment variable to avoid passing it every time:
> ```bash
> export ODDS_API_KEY="your_key_here"
> ```

---

## 5. Main File

**File:** `main.py`

### Overview
`main.py` is the primary entry point that ties the full pipeline together. It prompts for a player name and season, loads that player's game logs from the PostgreSQL cache, engineers features, runs the walk-forward backtest, and prints the evaluation results.

### How It Works — Step by Step

1. **Player lookup** — Prompts for a player name and resolves it to an NBA player ID using `find_player_id()`. Loops until a valid name is entered.
2. **Season selection** — Prompts for a season string. Defaults to the current season (`2025-26`) if left blank.
3. **Load cached logs** — Calls `load_cached_logs(player_id, season)` to retrieve the player's game logs from PostgreSQL. If no data is found, it reminds you to run `setup.py` first.
4. **Clean data** — Passes the raw DataFrame through `clean_player_games()` to normalize columns and convert minutes.
5. **Build features** — Calls `build_features()` to engineer all 7 pre-game features and create the Over/Under label.
6. **Print summaries** — Displays raw stat averages and a preview of the engineered features before the model runs.
7. **Run backtest** — Calls `walk_forward_backtest()` to simulate predictions across the season using 15-game rolling training windows.
8. **Evaluate results** — Calls `evaluate_results()` to print accuracy, high-confidence pick stats, and a sample prediction table.

### Dependencies

`main.py` relies on all three other core modules:

| Module | Used For |
|---|---|
| `getplayerinfo.py` | `find_player_id`, `clean_player_games` |
| `database.py` | `load_cached_logs` |
| `model.py` | `build_features`, `walk_forward_backtest`, `evaluate_results` |

> **Prerequisite:** The database must be populated before running `main.py`. Run `setup.py` first if you haven't already.

### Terminal Commands

```bash
# Run the full pipeline (prompts interactively for player and season)
python main.py
```
