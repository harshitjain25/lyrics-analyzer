"""Page 2 — Sentiment & Emotion Trends."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sentiment Trends", layout="wide", page_icon="📈")
st.title("📈 Sentiment & Emotion Trends")

from src.database.db_manager import DBManager
from dashboard.components.charts import apply_theme, PALETTE, EMOTION_COLORS

@st.cache_data(ttl=3600)
def load():
    return DBManager().get_dashboard_data()

data = load()
songs_df      = data["songs_df"]
sentiments_df = data["sentiments_df"]

GENRES  = ["pop","rock","hip-hop","country","r&b","electronic"]
DECADES = ["1960s","1970s","1980s","1990s","2000s","2010s","2020s"]
EMOTIONS = ["joy","sadness","anger","fear","surprise","disgust","neutral"]

sel_genres  = st.session_state.get("selected_genres",  GENRES)
sel_decades = st.session_state.get("selected_decades", DECADES)

if songs_df.empty or sentiments_df.empty:
    st.warning("No sentiment data. Run the analysis pipeline first or enable DEMO_MODE.")
    st.stop()

# Expand emotion_scores JSON column if present
if "emotion_scores" in sentiments_df.columns and EMOTIONS[0] not in sentiments_df.columns:
    import json
    def expand(row):
        try:
            es = row["emotion_scores"]
            if isinstance(es, str): es = json.loads(es)
            return pd.Series({e: es.get(e, 0) for e in EMOTIONS})
        except:
            return pd.Series({e: 0.0 for e in EMOTIONS})
    sentiments_df = pd.concat([sentiments_df, sentiments_df.apply(expand, axis=1)], axis=1)

merged = songs_df.merge(sentiments_df, left_on="id", right_on="song_id", how="left")
merged = merged[merged["genre"].isin(sel_genres) & merged["decade"].isin(sel_decades)]

# ── Row 1: Sentiment score by decade, one line per genre ──────────────────────
st.subheader("Average Sentiment Score by Decade")
decade_genre = merged.groupby(["decade","genre"])["sentiment_score"].mean().reset_index()
decade_genre = decade_genre[decade_genre["genre"].isin(sel_genres)]
fig1 = px.line(decade_genre, x="decade", y="sentiment_score", color="genre",
               markers=True, color_discrete_sequence=PALETTE,
               category_orders={"decade": DECADES})
fig1.update_layout(yaxis_title="Avg Sentiment Score", xaxis_title="Decade")
apply_theme(fig1)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")
col1, col2 = st.columns(2)

# ── Row 2 left: Emotion radar ──────────────────────────────────────────────────
with col1:
    st.subheader("Emotion Radar by Genre")
    g1 = st.selectbox("Genre A", sel_genres, index=0)
    g2 = st.selectbox("Genre B", sel_genres, index=min(1, len(sel_genres)-1))
    fig2 = go.Figure()
    for genre, color in [(g1, PALETTE[0]), (g2, PALETTE[2])]:
        gdf = merged[merged["genre"] == genre]
        vals = [gdf[e].mean() if e in gdf.columns else 0 for e in EMOTIONS]
        fig2.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=EMOTIONS + [EMOTIONS[0]],
            fill="toself", name=genre, line_color=color, opacity=0.7,
        ))
    fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Emotion Radar")
    apply_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2 right: Stacked area — emotion proportions over decades ───────────────
with col2:
    st.subheader("Emotion Mix Over Decades")
    sel_g = st.selectbox("Filter by Genre", ["All"] + sel_genres)
    area_df = merged if sel_g == "All" else merged[merged["genre"] == sel_g]
    show_emotions = ["joy","sadness","anger","fear"]
    em_decade = area_df.groupby("decade")[show_emotions].mean().reset_index()
    em_melt = em_decade.melt(id_vars="decade", var_name="emotion", value_name="score")
    em_melt = em_melt[em_melt["decade"].isin(DECADES)]
    colors = [EMOTION_COLORS.get(e,"#AAA") for e in show_emotions]
    fig3 = px.area(em_melt, x="decade", y="score", color="emotion",
                   color_discrete_map=EMOTION_COLORS,
                   category_orders={"decade": DECADES})
    apply_theme(fig3)
    st.plotly_chart(fig3, use_container_width=True)
