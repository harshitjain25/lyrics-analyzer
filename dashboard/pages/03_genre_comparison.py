"""Page 3 — Genre Comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Genre Comparison", layout="wide", page_icon="🎸")
st.title("🎸 Genre Comparison")

from src.database.db_manager import DBManager
from dashboard.components.charts import apply_theme, PALETTE, EMOTION_COLORS

@st.cache_data(ttl=3600)
def load():
    return DBManager().get_dashboard_data()

data = load()
songs_df      = data["songs_df"]
sentiments_df = data["sentiments_df"]
song_topics_df = data["song_topics_df"]
topics_df     = data["topics_df"]

GENRES  = ["pop","rock","hip-hop","country","r&b","electronic"]
DECADES = ["1960s","1970s","1980s","1990s","2000s","2010s","2020s"]
EMOTIONS = ["joy","sadness","anger","fear","surprise","disgust","neutral"]

if songs_df.empty or sentiments_df.empty:
    st.warning("No data. Run the pipeline or enable DEMO_MODE.")
    st.stop()

# Expand emotion_scores
if "emotion_scores" in sentiments_df.columns and EMOTIONS[0] not in sentiments_df.columns:
    def expand(row):
        try:
            es = row["emotion_scores"]
            if isinstance(es, str): es = json.loads(es)
            return pd.Series({e: es.get(e, 0) for e in EMOTIONS})
        except:
            return pd.Series({e: 0.0 for e in EMOTIONS})
    sentiments_df = pd.concat([sentiments_df, sentiments_df.apply(expand, axis=1)], axis=1)

merged = songs_df.merge(sentiments_df, left_on="id", right_on="song_id", how="left")

# ── Chart 1: Grouped bar — avg sentiment by genre per decade ──────────────────
st.subheader("Average Sentiment Score by Genre × Decade")
gd = merged.groupby(["genre","decade"])["sentiment_score"].mean().reset_index()
fig1 = px.bar(gd, x="decade", y="sentiment_score", color="genre", barmode="group",
              color_discrete_sequence=PALETTE,
              category_orders={"decade": DECADES})
apply_theme(fig1)
st.plotly_chart(fig1, use_container_width=True)

col1, col2 = st.columns(2)

# ── Chart 2: Diverging bar — genre ranked by overall sentiment ────────────────
with col1:
    st.subheader("Genres Ranked by Sentiment")
    genre_sent = merged.groupby("genre")["sentiment_score"].mean().reset_index()
    genre_sent = genre_sent.sort_values("sentiment_score")
    genre_sent["color"] = genre_sent["sentiment_score"].apply(
        lambda x: "#E63946" if x < 0.6 else "#2DC653"
    )
    fig2 = go.Figure(go.Bar(
        x=genre_sent["sentiment_score"] - 0.5,
        y=genre_sent["genre"],
        orientation="h",
        marker_color=genre_sent["color"],
    ))
    fig2.update_layout(xaxis_title="Relative Sentiment (centered at 0.5)", title="Most → Least Positive")
    apply_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

# ── Chart 3: Emotion heatmap genres × emotions ────────────────────────────────
with col2:
    st.subheader("Emotion Profile by Genre")
    avail = [e for e in EMOTIONS if e in merged.columns]
    em_genre = merged.groupby("genre")[avail].mean().reset_index()
    em_melt = em_genre.melt(id_vars="genre", var_name="emotion", value_name="score")
    fig3 = px.imshow(
        em_genre.set_index("genre")[avail],
        color_continuous_scale="RdYlGn", aspect="auto",
        title="Avg Emotion Score per Genre"
    )
    apply_theme(fig3)
    st.plotly_chart(fig3, use_container_width=True)

# ── Chart 4: Topic overlap matrix ─────────────────────────────────────────────
st.subheader("Topic Overlap Across Genres")
if not song_topics_df.empty and not topics_df.empty:
    mt = songs_df.merge(song_topics_df, left_on="id", right_on="song_id", how="left")
    mt = mt.merge(topics_df[["topic_id","label"]], on="topic_id", how="left")
    topic_genre = mt.groupby(["label","genre"]).size().reset_index(name="count")
    fig4 = px.scatter(topic_genre, x="genre", y="label", size="count",
                      color="count", color_continuous_scale="Purples",
                      title="Topic × Genre Song Count (bubble size = # songs)")
    apply_theme(fig4)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Topic data not available.")
