"""Run the daily refresh for the NBA props database.

Used by the macOS LaunchAgent
``com.nic.betting-app.daily-update`` to refresh the database each morning.
It calls setup.py, and the resulting cached data is read by main.py and
server.py for terminal and web predictions.

Provides:
- current_season() -> returns the appropriate NBA season label, such as
  ``2025-26``.
- main() -> runs the full league-wide daily refresh without user prompts.
"""

from datetime import date
from setup import load_all_data_to_db

def current_season(today: date | None = None) -> str:
    """Return the NBA season label for a date, such as ``2025-26``.

    NBA seasons begin in October, so January through September belong to the
    season that began in the preceding calendar year. Called by ``main()``.
    """
    today = today or date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def main() -> None:
    """Run the non-interactive league-wide refresh for the current season.

    Uses ``userchoice=1`` so setup.py loads all player game logs rather than
    prompting for one player. Called by the LaunchAgent and when this file is
    run directly for a manual test.
    """
    season = current_season()
    print(f"Starting daily NBA database refresh for {season}")
    load_all_data_to_db(userchoice=1, season=season)
    print("Daily refresh finished successfully.")

if __name__ == "__main__":
    main()
