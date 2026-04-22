"""Cleans and normalises raw song data into a structured DataFrame."""

from __future__ import annotations

import re
import uuid
from typing import Any

import pandas as pd

from config import DECADE_YEAR_RANGES, logger

_SECTION_HEADER_RE = re.compile(r"^\[.*?\]$", re.MULTILINE)


class LyricsPreprocessor:
    """Transforms a list of raw song dicts into a clean pandas DataFrame."""

    def preprocess(self, raw_songs: list[dict[str, Any]]) -> pd.DataFrame:
        """Run the full preprocessing pipeline on raw song data.

        Steps:
        1. Dedup on (title, artist)
        2. Language detection — keep English only
        3. Strip [Section Header] lines
        4. Normalise whitespace
        5. Filter by word count (50–2000)
        6. Add id, word_count, clean_lyrics, decade columns

        Returns a DataFrame with columns:
        id, title, artist, genre, year, decade, clean_lyrics, word_count
        """
        df = pd.DataFrame(raw_songs)
        initial = len(df)
        logger.info(f"Preprocessing: {initial} raw songs")

        # Step 1 — deduplicate
        df = df.drop_duplicates(subset=["title", "artist"], keep="first").reset_index(drop=True)
        logger.info(f"After dedup: {len(df)} songs (removed {initial - len(df)})")

        # Step 2 — language detection
        df = self._filter_english(df)
        logger.info(f"After lang filter: {len(df)} songs")

        # Step 3+4 — strip section headers and normalise whitespace
        df["clean_lyrics"] = df["lyrics"].apply(self._clean_text)

        # Step 5 — word count filter
        df["word_count"] = df["clean_lyrics"].apply(lambda t: len(t.split()))
        before_wc = len(df)
        df = df[(df["word_count"] >= 50) & (df["word_count"] <= 2000)].reset_index(drop=True)
        logger.info(f"After word count filter (50–2000): {len(df)} songs (removed {before_wc - len(df)})")

        # Step 5b — drop songs with missing/invalid year (prevents "0s" decade bucket)
        before_year = len(df)
        df["year"] = df["year"].apply(lambda y: int(y) if y and int(y) >= 1960 else 0)
        df = df[df["year"] >= 1960].reset_index(drop=True)
        logger.info(f"After year filter (>=1960): {len(df)} songs (removed {before_year - len(df)})")

        # Step 6 — add id and decade
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df["decade"] = df["year"].apply(self._year_to_decade)

        result = df[["id", "title", "artist", "genre", "year", "decade", "clean_lyrics", "word_count"]].copy()
        logger.info(f"Preprocessing complete: {len(result)} songs ready")
        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _filter_english(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows whose lyrics are not detected as English."""
        try:
            from langdetect import LangDetectException, detect
        except ImportError:
            logger.warning("langdetect not installed, skipping language filter")
            return df

        keep = []
        for _, row in df.iterrows():
            try:
                lang = detect(str(row.get("lyrics", ""))[:500])
                keep.append(lang == "en")
            except LangDetectException:
                keep.append(False)

        before = len(df)
        result = df[keep].reset_index(drop=True)
        logger.info(f"Language filter removed {before - len(result)} non-English songs")
        return result

    def _clean_text(self, lyrics: str) -> str:
        """Remove section headers and normalise whitespace."""
        text = _SECTION_HEADER_RE.sub("", str(lyrics))
        lines = [line.strip() for line in text.splitlines()]
        lines = [l for l in lines if l]
        return "\n".join(lines)

    def _year_to_decade(self, year: int) -> str:
        """Map a year integer to a decade string like '1980s'."""
        for decade, (start, end) in DECADE_YEAR_RANGES.items():
            if start <= int(year) <= end:
                return decade
        decade_start = (int(year) // 10) * 10
        return f"{decade_start}s"
