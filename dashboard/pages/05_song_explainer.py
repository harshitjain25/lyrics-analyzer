"""Page 5 — AI Song Explainer."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Song Explainer", layout="wide", page_icon="🤖")
st.title("🤖 AI Song Explainer")

from config import GROQ_API_KEY, MOCK_MODE
from src.database.db_manager import DBManager
from src.llm.song_explainer import SongExplainer
from dashboard.components.charts import apply_theme, EMOTION_COLORS

@st.cache_data(ttl=3600)
def load():
    return DBManager().get_dashboard_data()

data = load()
songs_df      = data["songs_df"]
sentiments_df = data["sentiments_df"]
song_topics_df = data["song_topics_df"]
topics_df     = data["topics_df"]
aspects_df    = data["aspects_df"]

if songs_df.empty:
    st.warning("No songs found. Run the pipeline or enable DEMO_MODE.")
    st.stop()

# Build dropdown options
songs_df["display"] = songs_df["title"] + " — " + songs_df["artist"] + " (" + songs_df["year"].astype(str) + ")"
options = songs_df["display"].tolist()

# Pre-select from search page if set
preselect_id = st.session_state.get("explain_song_id", None)
default_idx = 0
if preselect_id:
    matches = songs_df[songs_df["id"] == preselect_id].index.tolist()
    if matches: default_idx = matches[0]

sel = st.selectbox("Select a song", options, index=default_idx)
row = songs_df[songs_df["display"] == sel].iloc[0]
sid = row["id"]

# Lookup data
sent_row = {}
if not sentiments_df.empty:
    s = sentiments_df[sentiments_df.get("song_id", sentiments_df.get("id","")) == sid] if "song_id" in sentiments_df.columns else sentiments_df[sentiments_df["id"] == sid]
    if not s.empty: sent_row = s.iloc[0].to_dict()

topic_label, topic_desc = "—", ""
if not song_topics_df.empty and not topics_df.empty:
    st_row = song_topics_df[song_topics_df["song_id"] == sid]
    if not st_row.empty:
        tid = st_row.iloc[0]["topic_id"]
        t_row = topics_df[topics_df["topic_id"] == tid]
        if not t_row.empty:
            topic_label = t_row.iloc[0].get("label","—")
            topic_desc  = t_row.iloc[0].get("description","")

asp_rows = aspects_df[aspects_df["song_id"] == sid].to_dict("records") if not aspects_df.empty else []

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([4, 6])

with left:
    st.markdown(f"### {row['title']}")
    st.markdown(f"**Artist:** {row['artist']}  |  **Year:** {row['year']}  |  **Genre:** {row['genre']}")

    st.info(f"**Theme:** {topic_label}\n\n{topic_desc}")

    sentiment = sent_row.get("overall_sentiment", "—")
    badge = {"positive":"🟢 Positive","negative":"🔴 Negative","neutral":"🟡 Neutral"}.get(sentiment, sentiment)
    st.markdown(f"**Sentiment:** {badge}")

    es = sent_row.get("emotion_scores", {})
    if isinstance(es, str):
        try: es = json.loads(es)
        except: es = {}
    if es:
        em_df = pd.DataFrame({"emotion": list(es.keys()), "score": list(es.values())})
        fig = px.bar(em_df, x="score", y="emotion", orientation="h",
                     color="emotion", color_discrete_map=EMOTION_COLORS, title="Emotion Scores")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    if asp_rows:
        asp_df = pd.DataFrame(asp_rows)
        fig2 = px.bar(asp_df, x="score", y="aspect", color="sentiment", orientation="h",
                      color_discrete_map={"positive":"#2DC653","negative":"#E63946","neutral":"#ADB5BD"},
                      title="Aspect Sentiments")
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

with right:
    lyrics = str(row.get("clean_lyrics",""))
    with st.expander("📜 Lyrics", expanded=False):
        st.text(lyrics)

    # Read GROQ key at runtime — config.py may be imported before st.secrets is ready
    try:
        _groq_key = GROQ_API_KEY or st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        _groq_key = GROQ_API_KEY
    if not _groq_key and not MOCK_MODE:
        st.info("Add GROQ_API_KEY to your .env file to enable AI explanations.")
    else:
        if st.button("✨ Explain This Song", type="primary"):
            explainer = SongExplainer(api_key=_groq_key or None)
            with st.spinner("Generating analysis…"):
                stream = explainer.explain(
                    title=row["title"],
                    artist=row["artist"],
                    year=row["year"],
                    genre=row["genre"],
                    lyrics=lyrics,
                    topic_label=topic_label,
                    topic_description=topic_desc,
                    sentiment=sentiment,
                    emotion_scores=es,
                    aspect_sentiments=asp_rows,
                )
                st.write_stream(stream)
