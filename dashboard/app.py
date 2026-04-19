"""Main Streamlit app — Song Lyrics Thematic Analyzer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Song Lyrics Analyzer",
    layout="wide",
    page_icon="🎵",
)

from config import DEMO_MODE
from src.database.db_manager import DBManager

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🎵 Song Lyrics Analyzer")
st.sidebar.markdown("NLP-powered thematic analysis of popular music.")

GENRES  = ["pop", "rock", "hip-hop", "country", "r&b", "electronic"]
DECADES = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]

selected_genres  = st.sidebar.multiselect("Genre",  GENRES,  default=GENRES)
selected_decades = st.sidebar.multiselect("Decade", DECADES, default=DECADES)
st.session_state["selected_genres"]  = selected_genres
st.session_state["selected_decades"] = selected_decades

if DEMO_MODE:
    st.sidebar.warning("Running in Demo Mode with sample data")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    db = DBManager()
    return db.get_dashboard_data()

try:
    data = load_data()
    st.session_state["data"] = data
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

songs_df     = data["songs_df"]
topics_df    = data["topics_df"]
sentiments_df = data["sentiments_df"]

# Filter
if not songs_df.empty:
    filtered = songs_df[
        songs_df["genre"].isin(selected_genres) &
        songs_df["decade"].isin(selected_decades)
    ]
else:
    filtered = songs_df

n_songs   = len(filtered)
n_genres  = filtered["genre"].nunique() if not filtered.empty else 0
n_decades = filtered["decade"].nunique() if not filtered.empty else 0
n_topics  = len(topics_df) if not topics_df.empty else 0

st.sidebar.markdown("---")
st.sidebar.metric("Total Songs", n_songs)
st.sidebar.metric("Genres", n_genres)
st.sidebar.metric("Decades", n_decades)
st.sidebar.metric("Topics Found", n_topics)

# ── Home page ──────────────────────────────────────────────────────────────────
st.title("🎵 Song Lyrics Thematic Analyzer")
st.markdown("Explore themes, sentiment, and emotions across 6 decades of popular music.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Songs", n_songs)
c2.metric("Genres", n_genres)
c3.metric("Decades", n_decades)
c4.metric("Topics", n_topics)

st.markdown("---")

if not filtered.empty:
    col1, col2 = st.columns(2)
    with col1:
        decade_counts = filtered.groupby("decade").size().reset_index(name="count")
        fig1 = px.bar(decade_counts, x="decade", y="count", title="Songs per Decade",
                      color="decade", color_discrete_sequence=px.colors.qualitative.Vivid)
        fig1.update_layout(template="plotly_dark", showlegend=False, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        genre_counts = filtered.groupby("genre").size().reset_index(name="count").sort_values("count")
        fig2 = px.bar(genre_counts, x="count", y="genre", orientation="h",
                      title="Songs per Genre", color="genre",
                      color_discrete_sequence=px.colors.qualitative.Vivid)
        fig2.update_layout(template="plotly_dark", showlegend=False, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No songs match the current filters. Adjust genre/decade selection.")
