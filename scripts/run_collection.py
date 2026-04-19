"""Pipeline script: collect lyrics → preprocess → insert to Supabase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GENIUS_API_KEY, GENRES, DECADES, TARGET_SONGS_PER_CELL, logger
from src.collection.genius_collector import GeniusCollector
from src.collection.preprocessor import LyricsPreprocessor
from src.database.db_manager import DBManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect song lyrics and store in Supabase")
    p.add_argument("--genres", nargs="+", default=GENRES, help="Genres to collect")
    p.add_argument("--decades", nargs="+", default=DECADES, help="Decades to collect")
    p.add_argument("--songs-per-cell", type=int, default=TARGET_SONGS_PER_CELL)
    p.add_argument("--mock", action="store_true", help="Use MOCK_MODE (no API calls)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mock:
        import os
        os.environ["MOCK_MODE"] = "true"

    logger.info("=== Starting collection pipeline ===")
    logger.info(f"Genres: {args.genres}")
    logger.info(f"Decades: {args.decades}")
    logger.info(f"Songs per cell: {args.songs_per_cell}")

    # Collection
    collector = GeniusCollector(api_key=GENIUS_API_KEY)
    raw_songs = collector.collect_corpus(
        genres=args.genres,
        decades=args.decades,
        songs_per_cell=args.songs_per_cell,
    )
    logger.info(f"Collected {len(raw_songs)} raw songs")

    # Preprocessing
    preprocessor = LyricsPreprocessor()
    df = preprocessor.preprocess(raw_songs)
    logger.info(f"Preprocessed to {len(df)} clean songs")

    # Database insert
    db = DBManager()
    db.insert_songs(df)

    # Stats table
    print("\n" + "=" * 50)
    print("COLLECTION SUMMARY")
    print("=" * 50)
    print(f"{'Raw songs collected:':<30} {len(raw_songs)}")
    print(f"{'Songs after preprocessing:':<30} {len(df)}")
    if len(df) > 0:
        print(f"{'Genres covered:':<30} {df['genre'].nunique()}")
        print(f"{'Decades covered:':<30} {df['decade'].nunique()}")
        print(f"{'Avg word count:':<30} {df['word_count'].mean():.0f}")
        print("\nGenre breakdown:")
        for genre, count in df["genre"].value_counts().items():
            print(f"  {genre:<20} {count}")
        print("\nDecade breakdown:")
        for decade, count in df["decade"].value_counts().sort_index().items():
            print(f"  {decade:<20} {count}")
    print("=" * 50)
    logger.info("=== Collection pipeline complete ===")


if __name__ == "__main__":
    main()
