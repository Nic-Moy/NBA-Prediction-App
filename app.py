"""
PrizePicks Props Viewer

Run with:
    streamlit run app.py
"""

import datetime
import streamlit as st
import pandas as pd
from getprizepicks import fetch_projections, parse_projections, LEAGUES, NBA_DEFAULT_STATS

st.set_page_config(page_title="PrizePicks Props", layout="wide")
st.title("PrizePicks Props Viewer")

# ---------------------------------------------------------------------------
# Sidebar — league selector at top
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

league = st.sidebar.selectbox("League", sorted(LEAGUES), index=sorted(LEAGUES).index("NBA"))

# ---------------------------------------------------------------------------
# Fetch — cached per league for 5 minutes
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_props(selected_league: str) -> pd.DataFrame:
    raw = fetch_projections(selected_league)
    df = parse_projections(raw)
    return df[~df["is_promo"]].copy() if not df.empty else df


import pandas as pd  # noqa: E402 (needed after cache_data decorator)

col_left, col_right = st.columns([6, 1])
with col_right:
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()

df = load_props(league)

if df.empty:
    st.warning(f"No props returned for {league}.")
    st.stop()

fetched_at = datetime.datetime.now().strftime("%-I:%M %p")
st.caption(f"{league} — fetched at {fetched_at}. Auto-refreshes every 5 min.")

# ---------------------------------------------------------------------------
# Sidebar — remaining filters (repopulate based on the selected league)
# ---------------------------------------------------------------------------

player_query = st.sidebar.text_input("Player name", placeholder="e.g. LeBron")

all_stats = sorted(df["stat_type"].unique())
# Default to NBA main markets for NBA, all stats for other leagues
if league == "NBA":
    default_stats = [s for s in NBA_DEFAULT_STATS if s in all_stats]
else:
    default_stats = all_stats
selected_stats = st.sidebar.multiselect("Stat type", all_stats, default=default_stats)

all_tiers = sorted(df["tier"].unique())
selected_tiers = st.sidebar.multiselect("Tier", all_tiers, default=all_tiers)

all_teams = sorted(df["team"].dropna().unique())
selected_teams = st.sidebar.multiselect("Team", all_teams, default=[])

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

filtered = df.copy()

if player_query:
    filtered = filtered[filtered["player_name"].str.contains(player_query, case=False, na=False)]

if selected_stats:
    filtered = filtered[filtered["stat_type"].isin(selected_stats)]

if selected_tiers:
    filtered = filtered[filtered["tier"].isin(selected_tiers)]

if selected_teams:
    filtered = filtered[filtered["team"].isin(selected_teams)]

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

label = f"'{player_query}'" if player_query else league
st.subheader(f"{len(filtered)} props — {label}")

display_cols = ["player_name", "team", "position", "stat_type", "line", "tier", "start_time", "status"]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[display_cols].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    column_config={
        "player_name": st.column_config.TextColumn("Player"),
        "team":        st.column_config.TextColumn("Team"),
        "position":    st.column_config.TextColumn("Pos"),
        "stat_type":   st.column_config.TextColumn("Stat"),
        "line":        st.column_config.NumberColumn("Line", format="%.1f"),
        "tier":        st.column_config.TextColumn("Tier"),
        "start_time":  st.column_config.DatetimeColumn("Game Time", format="h:mm a"),
        "status":      st.column_config.TextColumn("Status"),
    },
)
