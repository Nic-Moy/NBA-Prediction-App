# NBA Player Props — Project Documentation

This project predicts NBA player prop bets using historical game logs, machine learning, and real-time lines from PrizePicks. It is built in Python with a PostgreSQL cache, the `nba_api` library for stats, `curl_cffi` to bypass bot protection on PrizePicks, and Streamlit for a browsable props viewer.

---

## Table of Contents
1. [Database](#1-database)
2. [Getting Player Info](#2-getting-player-info)
3. [ML Model](#3-ml-model)
4. [Getting PrizePicks Lines](#4-getting-prizepicks-lines)
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
- **`ensure_tables()`** — Creates all tables if they don't already exist. Safe to call repeatedly.
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

You must run `setup.py` before using `main.py`, as the model depends on cached data being in the database. Run it again each morning on game days to pull in the latest game logs before making picks.

### Terminal Commands

```bash
# Initial setup — populate the database (run this first, and each morning on game days)
python setup.py

# Verify database schema (creates tables if missing)
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
`model.py` is the core of the prediction system. It takes a cleaned game log DataFrame, engineers pre-game features that strictly avoid data leakage, and runs two separate model approaches side by side: a **classification model** that predicts Over/Under direction, and a **regression model** that predicts the actual stat value. The regression model's output is what gets compared to the real PrizePicks line at pick time.

### The Problem with Historical Lines

To train a supervised model that predicts "over or under the real betting line," you'd normally need historical examples of what the line was set at for each past game. That data isn't available here. The regression approach sidesteps this entirely:

- **Train:** predict the actual points scored using only historical game logs
- **Evaluate:** measure how close the predictions were to real outcomes (no lines needed)
- **Pick:** at game time, compare the model's predicted value to today's real PrizePicks line

The real line only enters at inference time — not during training. This means the model can be trained on any historical game log data you have.

### Feature Engineering — `build_features(df)`

Seven pre-game features are built from the game log data. Every rolling average uses `.shift(1)` before computing — this shifts each stat value one row forward before rolling, so the average for game N is computed from games 1 through N-1 only. This is the key safeguard against **data leakage** (accidentally training on information that wouldn't have been available before the game).

| Feature | Description |
|---|---|
| `is_home` | `1` if the player's team is home, `0` for away (from matchup string) |
| `rest_days` | Days since the previous game, clipped at 0 |
| `pts_avg_3` | 3-game rolling average points — prior games only |
| `pts_avg_5` | 5-game rolling average points — prior games only |
| `reb_avg_3` | 3-game rolling average rebounds — prior games only |
| `ast_avg_3` | 3-game rolling average assists — prior games only |
| `min_avg_3` | 3-game rolling average minutes — prior games only |

Two additional columns are built for the classification model only:

| Column | Description |
|---|---|
| `line_proxy_pts` | 5-game rolling average used as a stand-in betting line |
| `target_over_pts` | Binary label: `1` if actual points exceeded the proxy line, `0` otherwise |

### Why Walk-Forward Backtesting?

A simple train/test split would train on early games and test on later ones — fine for an exam, but too optimistic for sports. Player performance drifts over a season (injuries, fatigue, role changes), and a model trained on October games doesn't reflect what the player looks like in March.

Walk-forward backtesting fixes this by simulating how the model would actually be used in real time:

1. Starting at game 16, train on the **previous 15 games only**
2. Predict game 16
3. Advance one game, retrain on the new 15-game window
4. Predict game 17
5. Repeat for every remaining game in the season

The model is retrained fresh for every single prediction. This means it always reflects the player's most recent form and never peeks at future data.

### Model 1: Classification — `walk_forward_backtest(df, lookback_games=15)`

Uses **Logistic Regression** to predict whether the player goes Over or Under their proxy line. The output is a probability (e.g. 67% Over), not just a binary pick.

Additional guards to prevent bad predictions:
- Skips games where the training window has fewer than 8 clean samples
- Skips games where both classes (Over and Under) haven't appeared in the training window — a model that has only seen "Over" can't meaningfully predict "Under"

**Evaluation — `evaluate_results(results_df)`**

Results are measured against the **52.4% break-even threshold** — the minimum win rate to profit at standard -110 odds (risk $110 to win $100). The evaluator reports:
- Overall accuracy vs the 52.4% threshold
- High-confidence picks only (model probability ≥ 60%) and their accuracy
- A table of the most recent predictions

### Model 2: Regression — `walk_forward_regression(df, stat_col="PTS", lookback_games=15)`

Uses **Ridge Regression** (linear regression with L2 regularization, `alpha=1.0`) to predict the actual number of points (or rebounds, or assists) a player will score. The same walk-forward structure applies — retrain on 15 games, predict the next.

Ridge is used instead of plain linear regression because it penalizes large coefficients, which reduces overfitting when the feature set is small and correlated (rolling averages are highly correlated with each other).

**Evaluation — `evaluate_regression_results(results_df, stat_col="PTS")`**

| Metric | What it means |
|---|---|
| **MAE** | Mean absolute error — on average, how many points off the prediction was |
| **Within ±2.5** | % of games where the prediction was within 2.5 pts of actual |
| **Within ±5.0** | % of games where the prediction was within 5 pts of actual |
| **Bias** | Average signed error — positive means the model tends to underpredict, negative means overpredict |

A last-10-games table shows actual vs predicted vs error for a sanity check.

### Next-Game Prediction — `predict_next_game(df, stat_col="PTS", lookback_games=15)`

This is what gets called at pick time. Rather than evaluating historical accuracy, it trains on the **most recent** lookback window and predicts the next unplayed game.

The next-game feature vector is built manually:
- `pts_avg_3/5`, `reb_avg_3`, `ast_avg_3`, `min_avg_3` — computed directly from the last N real games (no shift needed since there's no target to leak into)
- `rest_days` — computed from today's date minus the last game date. If the last cached game is more than 7 days ago (stale data), `rest_days` defaults to `2.0` instead of the real gap, because a 35-day gap is outside the model's training distribution and causes wild extrapolation. A stale data warning is printed.
- `is_home` — unknown without tomorrow's schedule, so defaults to `0.5` (neutral)

Returns `{"predicted": float, "margin_of_error": float}` where margin of error is ±1 standard deviation of training residuals — a rough confidence interval.

### Terminal Commands

```bash
# Run the standalone model script (fetches data live from nba_api)
python model.py
```

---

## 4. Getting PrizePicks Lines

**Files:** `getprizepicks.py`, `app.py`

### Overview
`getprizepicks.py` fetches current player prop lines from PrizePicks across any supported sport. No API key is required — PrizePicks exposes a public JSON endpoint. The module handles the PerimeterX bot protection PrizePicks uses by making requests through `curl_cffi`, which impersonates a real Chrome browser at the TLS fingerprint level (not just headers).

`app.py` is a Streamlit web app that wraps this module into an interactive browser-based table with filters for league, player, stat type, tier, and team.

### Tiers

PrizePicks posts multiple lines per player per stat in three tiers:

| Tier | Meaning |
|---|---|
| `goblin` | Easier line — set below the player's typical output. Lower payout multiplier. |
| `standard` | The main line — closest to what a traditional sportsbook would set. |
| `demon` | Harder line — set above the player's typical output. Higher payout multiplier. |

For model comparison purposes, always use `standard`. The model's predicted value is compared against the standard line to determine Over/Under.

### Supported Leagues

| Name | Sport |
|---|---|
| `NBA` | NBA Basketball |
| `MLB` | MLB Baseball |
| `NHL` | NHL Hockey |
| `NFL` | NFL Football |
| `WNBA` | WNBA Basketball |
| `CBB` | College Basketball |
| `PGA` | Golf |
| `SOCCER` | Soccer |
| `TENNIS` | Tennis |
| `MMA` | MMA |

### NBA Stat Aliases

Short aliases can be used anywhere a stat type is accepted:

| Alias | PrizePicks Stat Type |
|---|---|
| `pts` | Points |
| `reb` | Rebounds |
| `ast` | Assists |
| `3pm` | 3-PT Made |
| `pra` | Pts+Rebs+Asts |
| `pr` | Pts+Rebs |
| `pa` | Pts+Asts |

### Key Functions

- **`fetch_projections(league, per_page)`** — Hits the PrizePicks API and returns the raw JSON. League can be a name (`"NBA"`) or a raw integer ID.
- **`parse_projections(raw)`** — Flattens the JSON:API format into a DataFrame. The response has two arrays — `data` (projections) and `included` (player metadata) — that must be joined on player ID to get the player name alongside the line.
- **`filter_by_stats(df, stats)`** — Keeps only rows matching the given stat types.
- **`filter_by_player(df, player_query)`** — Case-insensitive substring match on player name.
- **`get_props(league, stats, player)`** — Main convenience function for importing into other modules. Fetches, parses, filters promos, and returns a clean DataFrame.
- **`get_nba_props(stats, player)`** — Wrapper around `get_props` for NBA specifically. Defaults to the six main markets (pts, reb, ast, pra, pr, pa) when no stats are specified.

### CLI Arguments

| Flag | Description | Default |
|---|---|---|
| `--league` | Sport league (NBA, MLB, NHL, etc.) | `NBA` |
| `--player` | Filter by player name (substring match) | — |
| `--stats` | Comma-separated stat types or aliases | All stats |
| `--all-stats` | Skip stat filter, show everything | `False` |
| `--include-promos` | Include promotional lines | `False` |
| `--list-stats` | Print all stat types on the board and exit | `False` |
| `--list-leagues` | Print all supported leagues and exit | `False` |

### Terminal Commands

```bash
# View all NBA props (default: pts, reb, ast, pra, pr, pa)
python getprizepicks.py

# Filter to a single player
python getprizepicks.py --player "Bam Adebayo"

# Specific stats for one player
python getprizepicks.py --player "Stephen Curry" --stats pts,3pm,pra

# Different league
python getprizepicks.py --league NHL --player "McDavid"
python getprizepicks.py --league MLB --list-stats

# See all stat types currently on the board
python getprizepicks.py --list-stats

# Launch the interactive Streamlit viewer
streamlit run app.py
```

---

## 5. Main File

**File:** `main.py`

### Overview
`main.py` is the primary entry point that runs the full prediction pipeline for a single player. It loads cached game logs from the database, engineers features, runs both the classification and regression backtests for historical validation, then generates a next-game prediction and compares it live against today's real PrizePicks line.

### How It Works — Step by Step

1. **Player lookup** — Prompts for a player name and resolves it to an NBA player ID using `find_player_id()`. Loops until a valid name is entered.
2. **Season selection** — Prompts for a season string. Defaults to the current season (`2025-26`) if left blank.
3. **Load cached logs** — Calls `load_cached_logs(player_id, season)` to retrieve the player's game logs from PostgreSQL. If no data is found, it reminds you to run `setup.py` first.
4. **Clean data** — Passes the raw DataFrame through `clean_player_games()` to normalize columns and convert minutes from `"MM:SS"` to a decimal float.
5. **Build features** — Calls `build_features()` to engineer all 7 pre-game features plus the proxy line and classification label.
6. **Print summaries** — Displays raw stat averages and a preview of the engineered features.
7. **Classification backtest** — Runs `walk_forward_backtest()` and prints accuracy vs the 52.4% break-even threshold.
8. **Regression backtest** — Runs `walk_forward_regression()` and prints MAE, within-N-pts accuracy, and bias over the full season history.
9. **Next-game prediction** — Calls `predict_next_game()` to get a predicted point total and margin of error for the player's next unplayed game.
10. **PrizePicks comparison** — Fetches today's standard line from PrizePicks using `get_nba_props()`, compares it to the model's prediction, and prints a pick recommendation with a confidence label.

### Example Output (final section)

```
============================================================
NEXT GAME PREDICTION
============================================================
Model prediction:  23.4 pts  (± 4.1)
PrizePicks line:   21.0 pts
Recommendation:    OVER  (+2.4 margin)
Confidence:        Moderate
```

If the player has no game today or the line hasn't been posted yet, the prediction still prints and the line comparison is skipped cleanly.

If the cached game logs are more than 7 days old, a warning is printed and the prediction uses a default rest value to avoid extrapolation errors — a reminder to refresh the database before making real picks.

### Dependencies

| Module | Used For |
|---|---|
| `getplayerinfo.py` | `find_player_id`, `clean_player_games` |
| `database.py` | `load_cached_logs` |
| `model.py` | `build_features`, both backtests, `predict_next_game` |
| `getprizepicks.py` | `get_nba_props` — fetches the real line at pick time |

> **Daily workflow:** Run `python setup.py` (choice 1) each morning on game days to pull fresh logs. Then run `python main.py` for each player you want a pick on.

### Terminal Commands

```bash
# Run the full prediction pipeline
python main.py
```
