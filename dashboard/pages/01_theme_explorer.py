"""Page 1 — Theme Explorer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Theme Explorer", layout="wide", page_icon="🗂️")
st.title("🗂️ Theme Explorer")

from src.database.db_manager import DBManager
from dashboard.components.charts import apply_theme, heatmap, PALETTE

@st.cache_data(ttl=3600)
def load():
    return DBManager().get_dashboard_data()

data = load()
songs_df      = data["songs_df"]
topics_df     = data["topics_df"]
song_topics_df = data["song_topics_df"]

GENRES  = ["pop","rock","hip-hop","country","r&b","electronic"]
DECADES = ["1960s","1970s","1980s","1990s","2000s","2010s","2020s"]

sel_genres  = st.session_state.get("selected_genres",  GENRES)
sel_decades = st.session_state.get("selected_decades", DECADES)

# Merge songs with topics
if not songs_df.empty and not song_topics_df.empty and not topics_df.empty:
    merged = songs_df.merge(song_topics_df, left_on="id", right_on="song_id", how="left")
    merged = merged.merge(topics_df[["topic_id","label"]], on="topic_id", how="left")
    merged["label"] = merged["label"].fillna("Unknown")
    filtered = merged[merged["genre"].isin(sel_genres) & merged["decade"].isin(sel_decades)]
else:
    st.warning("No topic data found. Run the analysis pipeline first or enable DEMO_MODE.")
    st.stop()

# ── Row 1: Stacked bar (decade × topic, normalised) ───────────────────────────
st.subheader("Theme Distribution by Decade")
if not filtered.empty:
    decade_topic = filtered.groupby(["decade","label"]).size().reset_index(name="count")
    decade_total = filtered.groupby("decade").size().reset_index(name="total")
    decade_topic = decade_topic.merge(decade_total, on="decade")
    decade_topic["pct"] = (decade_topic["count"] / decade_topic["total"] * 100).round(1)
    fig = px.bar(decade_topic, x="decade", y="pct", color="label",
                 title="% of Songs per Theme by Decade",
                 color_discrete_sequence=PALETTE, barmode="stack",
                 category_orders={"decade": DECADES})
    fig.update_layout(yaxis_title="% of Songs", xaxis_title="Decade")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col1, col2 = st.columns(2)

# ── Row 2 left: Genre × Topic heatmap ─────────────────────────────────────────
with col1:
    st.subheader("Genre × Theme Heatmap")
    if not filtered.empty:
        gt = filtered.groupby(["genre","label"]).size().reset_index(name="count")
        gtotal = filtered.groupby("genre").size().reset_index(name="total")
        gt = gt.merge(gtotal, on="genre")
        gt["pct"] = (gt["count"] / gt["total"] * 100).round(1)
        pivot = gt.pivot_table(index="genre", columns="label", values="pct", aggfunc="sum").fillna(0)
        fig2 = px.imshow(pivot, title="% Genre Songs per Theme",
                         color_continuous_scale="Purples", aspect="auto")
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

# ── Row 2 right: Topic keyword bar ────────────────────────────────────────────
with col2:
    st.subheader("Topic Keywords")
    topic_labels = sorted(topics_df["label"].dropna().unique().tolist()) if not topics_df.empty else []
    if topic_labels:
        sel_topic = st.selectbox("Select a theme", topic_labels)
        row = topics_df[topics_df["label"] == sel_topic].iloc[0]
        keywords = row.get("keywords", [])
        if isinstance(keywords, str):
            import ast
            try: keywords = ast.literal_eval(keywords)
            except: keywords = keywords.split(",")
        if keywords:
            kw_df = pd.DataFrame({"keyword": keywords, "score": range(len(keywords), 0, -1)})
            fig3 = px.bar(kw_df, x="score", y="keyword", orientation="h",
                          title=f"Top Keywords: {sel_topic}",
                          color="score", color_continuous_scale="Purples")
            apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No keywords available for this topic.")
    else:
        st.info("No topics found.")
