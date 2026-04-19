-- Song Lyrics Analyzer — Database Migration 001
-- Run this SQL in your Supabase project's SQL editor.

-- ── songs ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS songs (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    artist      TEXT NOT NULL,
    genre       TEXT NOT NULL,
    year        INTEGER NOT NULL,
    decade      TEXT NOT NULL,
    clean_lyrics TEXT NOT NULL,
    word_count  INTEGER NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_songs_genre  ON songs (genre);
CREATE INDEX IF NOT EXISTS idx_songs_decade ON songs (decade);

-- ── topics ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS topics (
    id           SERIAL PRIMARY KEY,
    topic_id     INTEGER NOT NULL,
    label        TEXT NOT NULL,
    keywords     TEXT[] NOT NULL DEFAULT '{}',
    description  TEXT NOT NULL DEFAULT '',
    cultural_note TEXT NOT NULL DEFAULT '',
    created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ── song_topics ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS song_topics (
    id          SERIAL PRIMARY KEY,
    song_id     TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    topic_id    INTEGER NOT NULL,
    probability DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_song_topics_song_id ON song_topics (song_id);

-- ── sentiments ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sentiments (
    song_id            TEXT PRIMARY KEY REFERENCES songs(id) ON DELETE CASCADE,
    overall_sentiment  TEXT NOT NULL,
    sentiment_score    DOUBLE PRECISION NOT NULL,
    emotion_label      TEXT NOT NULL,
    emotion_scores     JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sentiments_song_id ON sentiments (song_id);

-- ── aspect_sentiments ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS aspect_sentiments (
    id         SERIAL PRIMARY KEY,
    song_id    TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    aspect     TEXT NOT NULL,
    sentiment  TEXT NOT NULL,
    score      DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_aspect_sentiments_song_id ON aspect_sentiments (song_id);
