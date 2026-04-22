"""Collects song lyrics from the Genius API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config import DECADE_YEAR_RANGES, MOCK_MODE, RAW_DIR, logger


class GeniusCollector:
    """Fetches lyrics from Genius for a specified genre × decade corpus."""

    def __init__(self, api_key: str) -> None:
        """Initialise lyricsgenius client (skipped in MOCK_MODE)."""
        self._api_key = api_key
        self._genius = None
        if not MOCK_MODE:
            import lyricsgenius
            self._genius = lyricsgenius.Genius(
                api_key,
                skip_non_songs=True,
                excluded_terms=["(Remix)", "(Live)"],
                remove_section_headers=False,
                timeout=10,
            )
            self._genius.verbose = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def collect_corpus(
        self,
        genres: list[str],
        decades: list[str],
        songs_per_cell: int = 140,
    ) -> list[dict[str, Any]]:
        """Collect a lyrics corpus across all genre × decade cells.

        Returns a flat list of song dicts with keys:
        title, artist, lyrics, genre, year, decade, album, genius_url.
        """
        if MOCK_MODE:
            return self._mock_songs()

        all_songs: list[dict[str, Any]] = []
        for genre in genres:
            for decade in decades:
                cell_songs = self._collect_cell(genre, decade, songs_per_cell)
                all_songs.extend(cell_songs)
        return all_songs

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _collect_cell(
        self, genre: str, decade: str, songs_per_cell: int
    ) -> list[dict[str, Any]]:
        """Collect songs for a single genre × decade cell with checkpoint support."""
        checkpoint_path = RAW_DIR / f"{genre}_{decade}.json"
        if checkpoint_path.exists():
            logger.info(f"Checkpoint found for {genre} {decade}, skipping API calls")
            with open(checkpoint_path) as f:
                return json.load(f)

        start_year, end_year = DECADE_YEAR_RANGES[decade]
        seen: set[tuple[str, str]] = set()
        songs: list[dict[str, Any]] = []

        for year in range(start_year, end_year + 1):
            if len(songs) >= songs_per_cell:
                break
            query = f"{genre} {year}"
            try:
                results = self._search_with_backoff(query, per_page=min(10, songs_per_cell - len(songs)))
            except Exception as exc:
                logger.warning(f"Search failed for '{query}': {exc}")
                continue

            for hit in results:
                key = (hit["title"].lower(), hit["artist"].lower())
                if key in seen:
                    continue
                seen.add(key)
                songs.append(hit)
                if len(songs) >= songs_per_cell:
                    break

        logger.info(f"Collecting {genre} {decade}: {len(songs)} songs")

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(songs, f, ensure_ascii=False, indent=2)

        return songs

    def _search_with_backoff(self, query: str, per_page: int = 10) -> list[dict[str, Any]]:
        """Search Genius with exponential backoff (max 3 retries)."""
        for attempt in range(3):
            try:
                result = self._genius.search_songs(query, per_page=per_page)
                hits = result.get("hits", []) if isinstance(result, dict) else []
                songs = []
                for h in hits:
                    song_data = h.get("result") or {}
                    pa = song_data.get("primary_artist") or {}
                    artist_name = pa.get("name", "") if isinstance(pa, dict) else ""
                    title = song_data.get("title", "")
                    rdc = song_data.get("release_date_components") or {}
                    year = int(rdc.get("year") or 0) if isinstance(rdc, dict) else 0
                    album_data = song_data.get("album") or {}
                    album_name = album_data.get("name", "") if isinstance(album_data, dict) else ""
                    lyrics = self._fetch_lyrics(song_data.get("id"))
                    if not lyrics:
                        continue
                    songs.append({
                        "title": title,
                        "artist": artist_name,
                        "lyrics": lyrics,
                        "genre": query.split()[0],
                        "year": year,
                        "decade": "",
                        "album": album_name,
                        "genius_url": song_data.get("url", ""),
                    })
                return songs
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(f"Genius API error (attempt {attempt+1}/3): {exc} — retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"Genius search failed after 3 retries for query: {query}")

    def _fetch_lyrics(self, song_id: int | None) -> str:
        """Fetch lyrics for a specific Genius song ID."""
        if not song_id:
            return ""
        try:
            song = self._genius.search_song(song_id=song_id)
            return song.lyrics if song else ""
        except Exception:
            return ""

    # ── Mock support ───────────────────────────────────────────────────────────

    def _mock_songs(self) -> list[dict[str, Any]]:
        """Return 20 realistic fake songs for testing without any API calls."""
        mock = []
        titles = [
            ("Neon Lights", "Synthwave Heroes"),
            ("Broken Road", "Country Roads Duo"),
            ("City Rain", "The Urban Poets"),
            ("Golden Days", "Retro Revival"),
            ("Heart on Fire", "Pop Sensation"),
            ("Deep Blue", "Jazz Collective"),
            ("Thunder Roll", "Southern Rock Band"),
            ("Sweet Sixteen", "R&B Icons"),
            ("Midnight Call", "Electronic Dreams"),
            ("Fade Away", "Indie Voices"),
            ("Rebel Yell", "Punk Uprising"),
            ("Summer Vibes", "Beach Crew"),
            ("Lost in Time", "Classic Rock"),
            ("New Dawn", "Soul Express"),
            ("Digital World", "Tech Pop"),
            ("Love is Blind", "Pop Duo"),
            ("Freedom Song", "Folk Revival"),
            ("Street Lights", "Hip Hop Collective"),
            ("Eternal Flame", "Ballad Queens"),
            ("Rising Sun", "World Fusion"),
        ]
        for i, (title, artist) in enumerate(titles):
            mock.append({
                "title": title,
                "artist": artist,
                "lyrics": f"Verse 1\nThis is the story of {title.lower()}\nA journey through the night and day\n\nChorus\nOh {title.lower()}, you take my breath away\nEvery single moment, I want to stay\n\nVerse 2\nThe memories we made along the way\nWill last forever and a day",
                "genre": ["pop", "rock", "hip-hop", "country", "r&b", "electronic"][i % 6],
                "year": 1965 + i * 3,
                "decade": "",
                "album": f"Album {i+1}",
                "genius_url": f"https://genius.com/mock-{i+1}",
            })
        return mock
