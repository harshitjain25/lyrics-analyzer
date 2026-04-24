"""Boost song count via artist-based Genius searches.

Searches by curated artist names (not 'genre year') to get unique songs.
Checkpoints per artist in data/raw/boost/ — safe to interrupt and resume.
Run after run_collection.py; upserts into Supabase on top of existing songs.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GENIUS_API_KEY, DECADE_YEAR_RANGES, logger
from src.collection.preprocessor import LyricsPreprocessor
from src.database.db_manager import DBManager

# ── Artist roster: {genre: [(artist, decade), ...]} ──────────────────────────
# ~20 songs fetched per artist → target ~2000+ additional raw songs

ARTIST_ROSTER: dict[str, list[tuple[str, str]]] = {
    "pop": [
        ("The Beatles", "1960s"),
        ("The Supremes", "1960s"),
        ("Roy Orbison", "1960s"),
        ("Simon & Garfunkel", "1960s"),
        ("Elton John", "1970s"),
        ("ABBA", "1970s"),
        ("Carpenters", "1970s"),
        ("Donna Summer", "1970s"),
        ("Michael Jackson", "1980s"),
        ("Madonna", "1980s"),
        ("Whitney Houston", "1980s"),
        ("Prince", "1980s"),
        ("Mariah Carey", "1990s"),
        ("Celine Dion", "1990s"),
        ("Backstreet Boys", "1990s"),
        ("Britney Spears", "2000s"),
        ("Beyonce", "2000s"),
        ("Justin Timberlake", "2000s"),
        ("Lady Gaga", "2010s"),
        ("Taylor Swift", "2010s"),
        ("Adele", "2010s"),
        ("Bruno Mars", "2010s"),
        ("Katy Perry", "2010s"),
        ("Ed Sheeran", "2010s"),
        ("Dua Lipa", "2020s"),
        ("Olivia Rodrigo", "2020s"),
        ("Harry Styles", "2020s"),
        ("The Weeknd", "2020s"),
    ],
    "rock": [
        ("The Rolling Stones", "1960s"),
        ("The Who", "1960s"),
        ("The Kinks", "1960s"),
        ("Jimi Hendrix", "1960s"),
        ("Led Zeppelin", "1970s"),
        ("Fleetwood Mac", "1970s"),
        ("Queen", "1970s"),
        ("Aerosmith", "1970s"),
        ("Eagles", "1970s"),
        ("Bon Jovi", "1980s"),
        ("Guns N Roses", "1980s"),
        ("The Cure", "1980s"),
        ("R.E.M.", "1980s"),
        ("U2", "1980s"),
        ("Nirvana", "1990s"),
        ("Pearl Jam", "1990s"),
        ("Green Day", "1990s"),
        ("Oasis", "1990s"),
        ("Red Hot Chili Peppers", "1990s"),
        ("Radiohead", "1990s"),
        ("Linkin Park", "2000s"),
        ("Coldplay", "2000s"),
        ("The Killers", "2000s"),
        ("Fall Out Boy", "2000s"),
        ("Arctic Monkeys", "2010s"),
        ("Imagine Dragons", "2010s"),
        ("Twenty One Pilots", "2010s"),
        ("Foo Fighters", "2010s"),
    ],
    "hip-hop": [
        ("Run DMC", "1980s"),
        ("LL Cool J", "1980s"),
        ("Public Enemy", "1980s"),
        ("Eric B and Rakim", "1980s"),
        ("Tupac Shakur", "1990s"),
        ("The Notorious BIG", "1990s"),
        ("Jay-Z", "1990s"),
        ("Nas", "1990s"),
        ("Wu-Tang Clan", "1990s"),
        ("Snoop Dogg", "1990s"),
        ("Eminem", "2000s"),
        ("Kanye West", "2000s"),
        ("Lil Wayne", "2000s"),
        ("50 Cent", "2000s"),
        ("T.I.", "2000s"),
        ("Kendrick Lamar", "2010s"),
        ("Drake", "2010s"),
        ("J. Cole", "2010s"),
        ("Nicki Minaj", "2010s"),
        ("Chance the Rapper", "2010s"),
        ("Travis Scott", "2020s"),
        ("Cardi B", "2020s"),
        ("Roddy Ricch", "2020s"),
        ("Tyler the Creator", "2020s"),
    ],
    "country": [
        ("Johnny Cash", "1960s"),
        ("Patsy Cline", "1960s"),
        ("Merle Haggard", "1960s"),
        ("Tammy Wynette", "1960s"),
        ("Dolly Parton", "1970s"),
        ("Willie Nelson", "1970s"),
        ("Waylon Jennings", "1970s"),
        ("Kenny Rogers", "1980s"),
        ("Alabama", "1980s"),
        ("Reba McEntire", "1980s"),
        ("Garth Brooks", "1990s"),
        ("Shania Twain", "1990s"),
        ("Alan Jackson", "1990s"),
        ("Tim McGraw", "2000s"),
        ("Kenny Chesney", "2000s"),
        ("Carrie Underwood", "2000s"),
        ("Blake Shelton", "2010s"),
        ("Luke Bryan", "2010s"),
        ("Miranda Lambert", "2010s"),
        ("Morgan Wallen", "2020s"),
        ("Luke Combs", "2020s"),
        ("Kane Brown", "2020s"),
    ],
    "r&b": [
        ("Marvin Gaye", "1960s"),
        ("Aretha Franklin", "1960s"),
        ("Stevie Wonder", "1960s"),
        ("James Brown", "1960s"),
        ("Al Green", "1970s"),
        ("Curtis Mayfield", "1970s"),
        ("Earth Wind and Fire", "1970s"),
        ("Barry White", "1970s"),
        ("Janet Jackson", "1980s"),
        ("Luther Vandross", "1980s"),
        ("Lionel Richie", "1980s"),
        ("Boyz II Men", "1990s"),
        ("TLC", "1990s"),
        ("Mary J Blige", "1990s"),
        ("Usher", "2000s"),
        ("Alicia Keys", "2000s"),
        ("Ne-Yo", "2000s"),
        ("Ciara", "2000s"),
        ("Frank Ocean", "2010s"),
        ("Miguel", "2010s"),
        ("SZA", "2020s"),
        ("Summer Walker", "2020s"),
        ("HER", "2020s"),
    ],
    "electronic": [
        ("Kraftwerk", "1970s"),
        ("Giorgio Moroder", "1970s"),
        ("Depeche Mode", "1980s"),
        ("New Order", "1980s"),
        ("Pet Shop Boys", "1980s"),
        ("Erasure", "1980s"),
        ("Daft Punk", "1990s"),
        ("The Prodigy", "1990s"),
        ("Fatboy Slim", "1990s"),
        ("The Chemical Brothers", "1990s"),
        ("Basement Jaxx", "2000s"),
        ("David Guetta", "2000s"),
        ("Justice", "2000s"),
        ("Calvin Harris", "2010s"),
        ("Skrillex", "2010s"),
        ("Disclosure", "2010s"),
        ("Flume", "2010s"),
        ("Kygo", "2010s"),
        ("Fred again", "2020s"),
        ("Four Tet", "2020s"),
    ],
}

BOOST_DIR = Path("data/raw/boost")
SONGS_PER_ARTIST = 20


def _year_from_song(song_data: dict) -> int:
    rdc = song_data.get("release_date_components") or {}
    year = int(rdc.get("year") or 0) if isinstance(rdc, dict) else 0
    if not year:
        rd = (song_data.get("release_date_for_display") or
              song_data.get("release_date") or "")
        m = re.search(r"(19[6-9]\d|20[0-2]\d)", str(rd))
        if m:
            year = int(m.group(1))
    return year


def _decade_from_year(year: int) -> str:
    for decade, (start, end) in DECADE_YEAR_RANGES.items():
        if start <= year <= end:
            return decade
    return ""


def collect_artist(genius, artist_name: str, genre: str, fallback_decade: str,
                   seen_keys: set[tuple[str, str]]) -> list[dict]:
    """Search Genius for an artist's songs and return enriched song dicts."""
    checkpoint = BOOST_DIR / f"{genre}_{artist_name.replace(' ', '_')}.json"
    if checkpoint.exists():
        logger.info(f"  [cached] {artist_name}")
        with open(checkpoint) as f:
            return json.load(f)

    logger.info(f"  Searching: {artist_name} ({genre}, {fallback_decade})")
    songs: list[dict] = []
    try:
        for attempt in range(3):
            try:
                result = genius.search_songs(artist_name, per_page=SONGS_PER_ARTIST)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(f"    API error attempt {attempt+1}: {exc} — wait {wait}s")
                time.sleep(wait)
        else:
            logger.warning(f"    Skipping {artist_name} after 3 failures")
            return []

        hits = result.get("hits", []) if isinstance(result, dict) else []
        for h in hits:
            song_data = h.get("result") or {}
            pa = song_data.get("primary_artist") or {}
            title = song_data.get("title", "")
            fetched_artist = pa.get("name", "") if isinstance(pa, dict) else ""

            key = (title.lower().strip(), fetched_artist.lower().strip())
            if key in seen_keys:
                continue
            seen_keys.add(key)

            year = _year_from_song(song_data)
            decade = _decade_from_year(year) if year >= 1960 else fallback_decade

            # Fetch lyrics
            song_id = song_data.get("id")
            if not song_id:
                continue
            try:
                song_obj = genius.search_song(song_id=song_id)
                lyrics = song_obj.lyrics if song_obj else ""
            except Exception:
                lyrics = ""
            if not lyrics:
                continue

            album_data = song_data.get("album") or {}
            album_name = album_data.get("name", "") if isinstance(album_data, dict) else ""

            songs.append({
                "title": title,
                "artist": fetched_artist,
                "lyrics": lyrics,
                "genre": genre,
                "year": year,
                "decade": decade,
                "album": album_name,
                "genius_url": song_data.get("url", ""),
            })

    except Exception as exc:
        logger.warning(f"    Error collecting {artist_name}: {exc}")

    with open(checkpoint, "w") as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)
    logger.info(f"    Got {len(songs)} songs from {artist_name}")
    return songs


def main() -> None:
    import lyricsgenius

    BOOST_DIR.mkdir(parents=True, exist_ok=True)

    genius = lyricsgenius.Genius(
        GENIUS_API_KEY,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)", "(Demo)"],
        remove_section_headers=False,
        timeout=15,
    )
    genius.verbose = False

    # Load existing song keys from already-collected raw checkpoints to avoid re-inserting
    seen_keys: set[tuple[str, str]] = set()
    for f in Path("data/raw").glob("*.json"):
        try:
            with open(f) as fp:
                for s in json.load(fp):
                    t = s.get("title", "").lower().strip()
                    a = s.get("artist", "").lower().strip()
                    if t and a:
                        seen_keys.add((t, a))
        except Exception:
            pass
    logger.info(f"Loaded {len(seen_keys)} existing song keys to skip duplicates")

    all_new_songs: list[dict] = []
    total_artists = sum(len(v) for v in ARTIST_ROSTER.values())
    done = 0

    for genre, artists in ARTIST_ROSTER.items():
        logger.info(f"\n=== Genre: {genre.upper()} ===")
        for artist_name, fallback_decade in artists:
            new_songs = collect_artist(genius, artist_name, genre, fallback_decade, seen_keys)
            all_new_songs.extend(new_songs)
            done += 1
            logger.info(f"Progress: {done}/{total_artists} artists | {len(all_new_songs)} new songs so far")

    logger.info(f"\nBoost collection complete: {len(all_new_songs)} raw songs")

    if not all_new_songs:
        logger.warning("No new songs collected — nothing to insert")
        return

    # Preprocess and upsert
    preprocessor = LyricsPreprocessor()
    df = preprocessor.preprocess(all_new_songs)
    logger.info(f"After preprocessing: {len(df)} clean songs")

    db = DBManager()
    db.insert_songs(df)

    print("\n" + "=" * 50)
    print("BOOST SUMMARY")
    print("=" * 50)
    print(f"{'Raw songs collected:':<35} {len(all_new_songs)}")
    print(f"{'Songs after preprocessing:':<35} {len(df)}")
    if len(df) > 0:
        print(f"\nGenre breakdown:")
        for genre, count in df["genre"].value_counts().items():
            print(f"  {genre:<25} {count}")
        print(f"\nDecade breakdown:")
        for decade, count in df["decade"].value_counts().sort_index().items():
            print(f"  {decade:<25} {count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
