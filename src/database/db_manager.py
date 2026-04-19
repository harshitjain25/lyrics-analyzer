"""Supabase database manager for the Song Lyrics Analyzer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import DEMO_MODE, SUPABASE_KEY, SUPABASE_URL, logger

_SAMPLE_CSV = Path(__file__).parent.parent.parent / "data" / "processed" / "sample_100.csv"


class DBManager:
    """Thin wrapper around Supabase for all read/write operations."""

    def __init__(self) -> None:
        """Initialise the Supabase client (or skip in DEMO_MODE)."""
        self._client = None
        # Auto-fallback to DEMO_MODE when credentials are absent
        self._demo = DEMO_MODE or not SUPABASE_URL or not SUPABASE_KEY
        if not self._demo:
            try:
                from supabase import create_client
                self._client = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("Supabase client initialised")
            except Exception as exc:
                logger.warning(f"Supabase unavailable, falling back to DEMO_MODE: {exc}")
                self._demo = True

    # ── Writes ─────────────────────────────────────────────────────────────────

    def insert_songs(self, df: pd.DataFrame) -> None:
        """Upsert songs DataFrame into the songs table."""
        if self._demo:
            logger.info("DEMO_MODE: skipping insert_songs")
            return
        records = df.to_dict(orient="records")
        try:
            self._client.table("songs").upsert(records, on_conflict="id").execute()
            logger.info(f"Upserted {len(records)} songs")
        except Exception as exc:
            logger.error(f"insert_songs failed: {exc}")
            raise

    def insert_topics(self, topics: list[dict[str, Any]]) -> None:
        """Insert topic label records."""
        if self._demo:
            logger.info("DEMO_MODE: skipping insert_topics")
            return
        try:
            self._client.table("topics").upsert(topics).execute()
            logger.info(f"Upserted {len(topics)} topics")
        except Exception as exc:
            logger.error(f"insert_topics failed: {exc}")
            raise

    def insert_song_topics(self, assignments: list[dict[str, Any]]) -> None:
        """Insert song↔topic assignment records."""
        if self._demo:
            logger.info("DEMO_MODE: skipping insert_song_topics")
            return
        try:
            self._client.table("song_topics").upsert(assignments).execute()
            logger.info(f"Upserted {len(assignments)} song_topic rows")
        except Exception as exc:
            logger.error(f"insert_song_topics failed: {exc}")
            raise

    def insert_sentiments(self, sentiments: list[dict[str, Any]]) -> None:
        """Upsert sentiment records (one row per song)."""
        if self._demo:
            logger.info("DEMO_MODE: skipping insert_sentiments")
            return
        try:
            self._client.table("sentiments").upsert(sentiments, on_conflict="song_id").execute()
            logger.info(f"Upserted {len(sentiments)} sentiment rows")
        except Exception as exc:
            logger.error(f"insert_sentiments failed: {exc}")
            raise

    def insert_aspect_sentiments(self, aspects: list[dict[str, Any]]) -> None:
        """Insert aspect-sentiment records."""
        if self._demo:
            logger.info("DEMO_MODE: skipping insert_aspect_sentiments")
            return
        try:
            self._client.table("aspect_sentiments").upsert(aspects).execute()
            logger.info(f"Upserted {len(aspects)} aspect_sentiment rows")
        except Exception as exc:
            logger.error(f"insert_aspect_sentiments failed: {exc}")
            raise

    # ── Reads ──────────────────────────────────────────────────────────────────

    def get_songs(self, genre: str | None = None, decade: str | None = None) -> pd.DataFrame:
        """Fetch songs, optionally filtered by genre and/or decade."""
        if self._demo:
            return self._load_sample(genre=genre, decade=decade)
        try:
            q = self._client.table("songs").select("*")
            if genre:
                q = q.eq("genre", genre)
            if decade:
                q = q.eq("decade", decade)
            result = q.execute()
            return pd.DataFrame(result.data)
        except Exception as exc:
            logger.error(f"get_songs failed: {exc}")
            raise

    def get_dashboard_data(self) -> dict[str, pd.DataFrame]:
        """Return all tables needed by the dashboard as a dict of DataFrames."""
        if self._demo:
            return self._demo_dashboard_data()
        try:
            songs_df = pd.DataFrame(self._client.table("songs").select("*").execute().data)
            topics_df = pd.DataFrame(self._client.table("topics").select("*").execute().data)
            song_topics_df = pd.DataFrame(self._client.table("song_topics").select("*").execute().data)
            sentiments_df = pd.DataFrame(self._client.table("sentiments").select("*").execute().data)
            aspects_df = pd.DataFrame(self._client.table("aspect_sentiments").select("*").execute().data)
            return {
                "songs_df": songs_df,
                "topics_df": topics_df,
                "song_topics_df": song_topics_df,
                "sentiments_df": sentiments_df,
                "aspects_df": aspects_df,
            }
        except Exception as exc:
            logger.error(f"get_dashboard_data failed: {exc}")
            raise

    # ── Demo helpers ───────────────────────────────────────────────────────────

    def _load_sample(self, genre: str | None = None, decade: str | None = None) -> pd.DataFrame:
        """Load sample CSV and optionally filter."""
        df = pd.read_csv(_SAMPLE_CSV)
        if genre:
            df = df[df["genre"] == genre]
        if decade:
            df = df[df["decade"] == decade]
        return df.reset_index(drop=True)

    def _demo_dashboard_data(self) -> dict[str, pd.DataFrame]:
        """Build dashboard DataFrames from sample_100.csv."""
        df = pd.read_csv(_SAMPLE_CSV)

        songs_df = df[["id", "title", "artist", "genre", "year", "decade", "clean_lyrics", "word_count"]].copy()

        # Build topics from unique topic_id/topic_label pairs
        topics_df = (
            df[["topic_id", "topic_label"]]
            .drop_duplicates("topic_id")
            .rename(columns={"topic_label": "label"})
            .assign(keywords=lambda x: x["label"].apply(lambda l: l.split(" ")), description="", cultural_note="")
            .reset_index(drop=True)
        )

        song_topics_df = df[["id", "topic_id"]].rename(columns={"id": "song_id"}).assign(probability=0.85)

        emotion_cols = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
        sentiments_df = df[["id", "overall_sentiment", "sentiment_score", "emotion_label"] + emotion_cols].copy()
        sentiments_df = sentiments_df.rename(columns={"id": "song_id"})
        sentiments_df["emotion_scores"] = sentiments_df[emotion_cols].apply(
            lambda row: row.to_dict(), axis=1
        )
        sentiments_df = sentiments_df.drop(columns=emotion_cols)

        # Build aspect_sentiments: unpack from a simple synthetic representation
        aspect_rows = []
        aspects = ["love and relationships", "identity and self-expression", "celebration and party", "loss and grief"]
        for _, row in df.iterrows():
            aspect = aspects[int(row["topic_id"]) % len(aspects)]
            aspect_rows.append({
                "song_id": row["id"],
                "aspect": aspect,
                "sentiment": row["overall_sentiment"],
                "score": float(row["sentiment_score"]),
            })
        aspects_df = pd.DataFrame(aspect_rows)

        return {
            "songs_df": songs_df,
            "topics_df": topics_df,
            "song_topics_df": song_topics_df,
            "sentiments_df": sentiments_df,
            "aspects_df": aspects_df,
        }
