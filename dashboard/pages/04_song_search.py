"""Page 4 — Semantic Song Search."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Song Search", layout="wide", page_icon="🔍")
st.title("🔍 Semantic Song Search")

from src.database.db_manager import DBManager
from dashboard.components.charts import apply_theme, EMOTION_COLORS

EMBEDDINGS_PATH = Path(__file__).parent.parent.parent / "data" / "embeddings" / "embeddings.npy"
IDS_PATH        = Path(__file__).parent.parent.parent / "data" / "embeddings" / "song_ids.json"

@st.cache_data(ttl=3600)
def load():
    return DBManager().get_dashboard_data()

@st.cache_resource
def load_embeddings():
    if not EMBEDDINGS_PATH.exists() or not IDS_PATH.exists():
        return None, None
    emb = np.load(str(EMBEDDINGS_PATH))
    with open(IDS_PATH) as f:
        ids = json.load(f)
    return emb, ids

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

data = load()
songs_df      = data["songs_df"]
sentiments_df = data["sentiments_df"]
song_topics_df = data["song_topics_df"]
topics_df     = data["topics_df"]
aspects_df    = data["aspects_df"]

emb, song_ids = load_embeddings()

if emb is None:
    st.warning("Embeddings not found. Run `python scripts/run_analysis.py` first, or enable DEMO_MODE.")
    st.info("In DEMO_MODE, set `DEMO_MODE=true` in your `.env` file.")
    st.stop()

# Build lookup dicts
songs_lookup = songs_df.set_index("id").to_dict("index") if not songs_df.empty else {}

sent_lookup: dict = {}
if not sentiments_df.empty:
    for _, row in sentiments_df.iterrows():
        sid = row.get("song_id", row.get("id",""))
        sent_lookup[sid] = row.to_dict()

topic_lookup: dict = {}
if not song_topics_df.empty and not topics_df.empty:
    tlabel = topics_df.set_index("topic_id")["label"].to_dict()
    for _, row in song_topics_df.iterrows():
        topic_lookup[row["song_id"]] = tlabel.get(row["topic_id"], "Unknown")

query = st.text_input("Describe a theme or mood (e.g. heartbreak and longing, party anthem)")

if query:
    with st.spinner("Searching…"):
        model = load_model()
        q_emb = model.encode([query])
        # Cosine similarity
        norm_emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        norm_q   = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        sims = (norm_emb @ norm_q.T).squeeze()
        top10 = np.argsort(sims)[::-1][:10]

    st.markdown(f"**Top 10 results for:** _{query}_")
    for rank, idx in enumerate(top10):
        sid = song_ids[idx]
        song = songs_lookup.get(sid, {})
        if not song: continue
        sent = sent_lookup.get(sid, {})
        topic_label = topic_lookup.get(sid, "—")
        sentiment   = sent.get("overall_sentiment", "—")
        badge_color = {"positive":"🟢","negative":"🔴","neutral":"🟡"}.get(sentiment,"⚪")

        with st.expander(f"{rank+1}. **{song.get('title','?')}** — {song.get('artist','?')} ({song.get('year','?')}) · {song.get('genre','?')} · {badge_color} {sentiment}"):
            st.markdown(f"**Theme:** {topic_label}")
            st.markdown(f"**Lyrics excerpt:**")
            st.text(str(song.get("clean_lyrics",""))[:400] + "…")

            # Emotion chart
            es = sent.get("emotion_scores", {})
            if isinstance(es, str):
                try: es = json.loads(es)
                except: es = {}
            if es:
                em_df = pd.DataFrame({"emotion": list(es.keys()), "score": list(es.values())})
                fig = px.bar(em_df, x="score", y="emotion", orientation="h",
                             color="emotion", color_discrete_map=EMOTION_COLORS)
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            # Aspect breakdown
            if not aspects_df.empty:
                asp = aspects_df[aspects_df["song_id"] == sid]
                if not asp.empty:
                    fig2 = px.bar(asp, x="score", y="aspect", color="sentiment", orientation="h",
                                  color_discrete_map={"positive":"#2DC653","negative":"#E63946","neutral":"#ADB5BD"})
                    apply_theme(fig2)
                    st.plotly_chart(fig2, use_container_width=True)

            if st.button("🤖 Explain this song", key=f"explain_{sid}"):
                st.session_state["explain_song_id"] = sid
                st.switch_page("pages/05_song_explainer.py")
