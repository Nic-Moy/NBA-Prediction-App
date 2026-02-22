# `nba_api` Python Package — Endpoint Reference Guide

> **Package**: `nba_api` (by swar) — wraps the unofficial NBA.com Stats API  
> **Install**: `pip install nba_api`  
> **Requires**: Python 3.10+, `requests`, `numpy` (pandas optional but recommended)  
> **GitHub**: https://github.com/swar/nba_api

---

## How It Works

The library has three main modules:

1. **`nba_api.stats.static`** — Cached player/team lookup (no HTTP calls)
2. **`nba_api.stats.endpoints`** — The main stats endpoints (hits stats.nba.com)
3. **`nba_api.live.nba.endpoints`** — Live game data

Every endpoint returns data via `.get_data_frames()` (list of DataFrames), `.get_json()`, or `.get_dict()`. Most endpoints require either a `player_id` or `team_id`, which you get from the static module.

---

## Static Module (Player/Team ID Lookup)

```python
from nba_api.stats.static import players, teams

# Find a player
players.find_players_by_full_name("LeBron James")
# → [{'id': 2544, 'full_name': 'LeBron James', ...}]

# Get all active players
players.get_active_players()

# Find a team
teams.find_teams_by_full_name("Los Angeles Lakers")
# → [{'id': 1610612747, ...}]

# Get all 30 NBA teams
teams.get_teams()
```

---

## Endpoints Organized by Use Case

### 🏀 1. Individual Player Game Logs (YOUR PRIMARY ENDPOINT)

**This is the most important endpoint for your prop prediction project.** It gives you game-by-game stats for a single player in a season.

| Endpoint | Class | Key Params |
|---|---|---|
| `PlayerGameLog` | `playergamelog.PlayerGameLog` | `player_id`, `season` |

```python
from nba_api.stats.endpoints import playergamelog

log = playergamelog.PlayerGameLog(player_id=2544, season='2024-25')
df = log.get_data_frames()[0]
```

**Returns per game**: `GAME_DATE`, `MATCHUP`, `WL`, `MIN`, `FGM`, `FGA`, `FG_PCT`, `FG3M`, `FG3A`, `FG3_PCT`, `FTM`, `FTA`, `FT_PCT`, `OREB`, `DREB`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `PF`, `PTS`, `PLUS_MINUS`

**Why you need it**: This is your training data. Each row = one game. You build rolling averages, features, and targets (PTS, REB, AST for over/under classification) from this.

---

### 📊 2. Player Career & Season Averages

| Endpoint | Class | Use Case |
|---|---|---|
| `PlayerCareerStats` | `playercareerstats.PlayerCareerStats` | Season-by-season career totals/averages |
| `CommonPlayerInfo` | `commonplayerinfo.CommonPlayerInfo` | Bio info + headline stats (current PTS/REB/AST avg) |
| `PlayerProfileV2` | `playerprofilev2.PlayerProfileV2` | Comprehensive career profile with splits |

```python
from nba_api.stats.endpoints import playercareerstats

career = playercareerstats.PlayerCareerStats(player_id=203999)
season_totals = career.season_totals_regular_season.get_data_frame()
```

**Why you might need it**: Establishing baseline expectations for a player, comparing current season to career norms.

---

### 🛡️ 3. Team Defensive Stats (CRITICAL FOR YOUR MODEL)

These endpoints let you get opponent defensive ratings — a key feature for predicting if a player will go over/under.

| Endpoint | Class | Use Case |
|---|---|---|
| `LeagueDashTeamStats` | `leaguedashteamstats.LeagueDashTeamStats` | Team-level stats for all 30 teams (including defensive stats) |
| `TeamDashboardByGeneralSplits` | `teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits` | Team stats split by home/away, W/L, etc. |

```python
from nba_api.stats.endpoints import leaguedashteamstats

# Get all team stats — use MeasureType='Advanced' for OFF_RATING, DEF_RATING
team_stats = leaguedashteamstats.LeagueDashTeamStats(
    season='2024-25',
    measure_type_detailed_defense='Base',  # or 'Advanced'
    per_mode_detailed='PerGame'
)
df = team_stats.get_data_frames()[0]
```

**Returns**: `TEAM_ID`, `TEAM_NAME`, `GP`, `W`, `L`, `W_PCT`, `FGM`, `FGA`, `FG_PCT`, `FG3M`, `FG3A`, `FG3_PCT`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `PTS`, `PLUS_MINUS` + rank columns

**With `MeasureType='Advanced'`**: `OFF_RATING`, `DEF_RATING`, `NET_RATING`, `PACE`, `TS_PCT`, etc.

**Why you need it**: Join opponent `TEAM_ID` from the game log's `MATCHUP` column to get the opponent's defensive rating as a feature.

---

### 📋 4. League-Wide Player Stats

| Endpoint | Class | Use Case |
|---|---|---|
| `LeagueDashPlayerStats` | `leaguedashplayerstats.LeagueDashPlayerStats` | All players' season stats in one call |
| `LeagueLeaders` | `leagueleaders.LeagueLeaders` | Top players ranked by any stat category |
| `LeagueGameLog` | `leaguegamelog.LeagueGameLog` | Every player's (or team's) game log for a season |

```python
from nba_api.stats.endpoints import leaguedashplayerstats

all_players = leaguedashplayerstats.LeagueDashPlayerStats(
    season='2024-25',
    per_mode_detailed='PerGame'
)
df = all_players.get_data_frames()[0]
```

**Why you might need it**: `LeagueGameLog` with `PlayerOrTeam='P'` is great for bulk-downloading all player game logs in one call rather than looping through individual players. `LeagueLeaders` is useful for identifying top performers to display in your app.

---

### 📦 5. Box Score Endpoints (Per-Game Detail)

| Endpoint | Class | What It Adds |
|---|---|---|
| `BoxScoreTraditionalV2` | `boxscoretraditionalv2.BoxScoreTraditionalV2` | Standard box score for a specific game |
| `BoxScoreAdvancedV2` | `boxscoreadvancedv2.BoxScoreAdvancedV2` | Advanced stats (TS%, USG%, OFF_RATING per player) |
| `BoxScoreSummaryV2` | `boxscoresummaryv2.BoxScoreSummaryV2` | Game summary, officials, line score |
| `BoxScoreTraditionalV3` | (newer V3 version) | Updated format — use for games after April 2025 |

```python
from nba_api.stats.endpoints import boxscoretraditionalv2

box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id='0022400001')
player_stats = box.get_data_frames()[0]  # player-level
team_stats = box.get_data_frames()[1]    # team-level
```

**Note**: `game_id` comes from game logs (the `Game_ID` column). Format is `002XXYYYY` where XX = season type, YYYY = game number.

**Why you might need it**: If you want per-game advanced stats like usage rate or true shooting — these aren't in the basic game log.

---

### 🎯 6. Shot Chart Data

| Endpoint | Class | Use Case |
|---|---|---|
| `ShotChartDetail` | `shotchartdetail.ShotChartDetail` | Every shot attempt with x,y coordinates, zone, distance |

```python
from nba_api.stats.endpoints import shotchartdetail

shots = shotchartdetail.ShotChartDetail(
    player_id=2544,
    team_id=0,  # 0 for all teams
    season_nullable='2024-25',
    context_measure_simple='FGA'
)
df = shots.get_data_frames()[0]
```

**Returns**: `LOC_X`, `LOC_Y`, `SHOT_ZONE_BASIC`, `SHOT_ZONE_AREA`, `SHOT_DISTANCE`, `SHOT_MADE_FLAG`, `ACTION_TYPE`, etc.

**Why you might need it**: Not essential for your MVP, but cool for visualizations and could help with shooting efficiency features later.

---

### 🏟️ 7. Schedule & Scoreboard

| Endpoint | Class | Use Case |
|---|---|---|
| `ScoreboardV2` | `scoreboardv2.ScoreboardV2` | Today's games and scores |
| `LeagueGameFinder` | `leaguegamefinder.LeagueGameFinder` | Find games by team, date range, season |

```python
from nba_api.stats.endpoints import leaguegamefinder

games = leaguegamefinder.LeagueGameFinder(
    team_id_nullable=1610612747,  # Lakers
    season_nullable='2024-25'
)
df = games.get_data_frames()[0]
```

**Why you might need it**: `LeagueGameFinder` is useful for getting a team's schedule and results, which helps you figure out upcoming opponents for predictions.

---

### ⏱️ 8. Live Game Data

```python
from nba_api.live.nba.endpoints import scoreboard

games = scoreboard.ScoreBoard()
games.get_dict()
```

**Why you might need it**: For a future feature showing live scores in your app, not needed for the ML pipeline.

---

### 🧑‍🤝‍🧑 9. Team Roster & Info

| Endpoint | Class | Use Case |
|---|---|---|
| `CommonTeamRoster` | `commonteamroster.CommonTeamRoster` | Full roster with player IDs, positions, height, weight |
| `TeamInfoCommon` | `teaminfocommon.TeamInfoCommon` | Team record, conference, division |
| `TeamDetails` | `teamdetails.TeamDetails` | Franchise history, arena, coach, GM |

---

### 🔄 10. Player Dashboard & Splits

| Endpoint | Class | Use Case |
|---|---|---|
| `PlayerDashboardByGeneralSplits` | Various `playerdashboard*` | Home/Away, W/L, monthly, pre/post All-Star splits |
| `PlayerDashboardByOpponent` | | Stats broken down by opponent |
| `PlayerDashboardByGameSplits` | | By half, quarter |

**Why you might need it**: Home/away splits and opponent-specific performance could be strong features for your classification model.

---

## Recommended Endpoints for Your Project

Based on your prop prediction MVP, here's the priority order:

| Priority | Endpoint | Why |
|---|---|---|
| **P0 — Must Have** | `PlayerGameLog` | Training data — game-by-game PTS, REB, AST, MIN |
| **P0 — Must Have** | `LeagueDashTeamStats` (Advanced) | Opponent DEF_RATING as a feature |
| **P1 — Important** | `LeagueGameFinder` | Map matchups to opponent team IDs |
| **P1 — Important** | `LeagueLeaders` | Display top performers in your app UI |
| **P2 — Nice to Have** | `BoxScoreAdvancedV2` | Per-game USG%, TS% as additional features |
| **P2 — Nice to Have** | `CommonPlayerInfo` | Player bio/position for your app |
| **P3 — Later** | `ShotChartDetail` | Visualizations, shot profile features |
| **P3 — Later** | Live scoreboard | Real-time scores in your app |

---

## Important Tips

1. **Rate limiting**: Add `time.sleep(0.6)` between API calls. NBA.com will throttle/block you otherwise.

2. **Season format**: Always `'2024-25'` style (not `'2025'`).

3. **V2 vs V3 endpoints**: Some V2 box score endpoints stopped returning data after April 2025. Use V3 versions for recent games (`BoxScoreTraditionalV3`, `BoxScoreSummaryV3`).

4. **LeagueID**: Recent versions of `nba_api` default to `'00'` (NBA). If you get empty results, explicitly pass `league_id_nullable='00'`.

5. **Bulk data strategy**: Instead of calling `PlayerGameLog` for 400+ players individually, use `LeagueGameLog(player_or_team_abbreviation='P')` to get ALL player game logs for a season in one call.
