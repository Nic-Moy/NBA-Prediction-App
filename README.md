# NBA-Prediction-App

A project for learning ML with my favorite sport. Pulls NBA game logs, builds
features, and projects next-game player stats — then compares the model's forecast
against live PrizePicks lines for an Over/Under read.

## PROPSDESK web app

`PROPSDESK` is the front end: a FastAPI + HTMX + Tailwind terminal that lists live
PrizePicks props and runs the model on demand.

### Run

```bash
# install deps (use your venv)
env/bin/pip install -r requirements.txt

# start the server
env/bin/python -m uvicorn server:app --reload --port 8000
```

Then open http://localhost:8000.

- **Props grid** — live PrizePicks lines, filterable by league, player, stat, and tier
  (5-minute cache; hit Refresh to bypass it).
- **Model Console** — enter a player + opponent + stat (PTS / REB / AST) to get the
  next-game forecast (± margin of error) and the Over/Under call vs the posted line.

Projections require cached game logs in the database. Load them first with:

```bash
env/bin/python setup.py
```

## Notes

- `ODDS_API_KEY` (used by `getproplines.py`) is read from the environment, or from
  the gitignored `.streamlit/secrets.toml` if you still run that path. Keep that key
  somewhere safe — it is not committed.
- Model logic lives in `model.py`; scraping in `getprizepicks.py` / `getproplines.py`;
  data access in `database.py` / `getplayerinfo.py`.
