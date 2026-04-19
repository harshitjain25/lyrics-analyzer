"""Pipeline script: load lyrics from Supabase → run all NLP analyses → save results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import logger
from src.database.db_manager import DBManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full NLP analysis pipeline")
    p.add_argument("--skip-topics", action="store_true")
    p.add_argument("--skip-sentiment", action="store_true")
    p.add_argument("--skip-emotion", action="store_true")
    p.add_argument("--skip-aspects", action="store_true")
    p.add_argument("--skip-llm-labels", action="store_true")
    p.add_argument("--mock", action="store_true", help="Use MOCK_MODE")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mock:
        import os
        os.environ["MOCK_MODE"] = "true"

    logger.info("=== Starting analysis pipeline ===")

    db = DBManager()
    songs_df = db.get_songs()
    if songs_df.empty:
        logger.error("No songs found in the database. Run run_collection.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(songs_df)} songs")
    lyrics_list = songs_df["clean_lyrics"].tolist()
    ids = songs_df["id"].tolist()

    completed: list[str] = []

    # ── Topic modeling ─────────────────────────────────────────────────────────
    topic_assignments: dict = {}
    topic_keywords: dict = {}
    if not args.skip_topics:
        from src.analysis.topic_modeling import TopicModeler
        logger.info("Running BERTopic …")
        modeler = TopicModeler()
        topic_assignments = modeler.fit(lyrics_list, ids)
        topic_keywords = modeler.get_topic_keywords()
        assignment_records = [
            {"song_id": sid, "topic_id": v["topic_id"], "probability": v["probability"]}
            for sid, v in topic_assignments.items()
        ]
        db.insert_song_topics(assignment_records)
        completed.append("topics")
        logger.info(f"Topics: {len(topic_keywords)} topics discovered")

    # ── LLM topic labels ───────────────────────────────────────────────────────
    if not args.skip_llm_labels and topic_keywords:
        from src.llm.topic_labeler import TopicLabeler
        logger.info("Labeling topics with Groq …")
        labeler = TopicLabeler()
        labels = labeler.label_topics(topic_keywords)
        topic_records = [
            {
                "topic_id": tid,
                "label": info.get("label", f"Topic {tid}"),
                "keywords": topic_keywords.get(tid, []),
                "description": info.get("description", ""),
                "cultural_note": info.get("cultural_note", ""),
            }
            for tid, info in labels.items()
        ]
        db.insert_topics(topic_records)
        completed.append("llm-labels")

    # ── Sentiment analysis ─────────────────────────────────────────────────────
    sentiment_results = []
    if not args.skip_sentiment:
        from src.analysis.sentiment_analysis import SentimentAnalyzer
        logger.info("Running sentiment analysis …")
        analyzer = SentimentAnalyzer()
        sentiment_results = analyzer.analyze_batch(lyrics_list)
        completed.append("sentiment")

    # ── Emotion analysis ───────────────────────────────────────────────────────
    emotion_results = []
    if not args.skip_emotion:
        from src.analysis.emotion_analysis import EmotionAnalyzer
        logger.info("Running emotion analysis …")
        analyzer = EmotionAnalyzer()
        emotion_results = analyzer.analyze_batch(lyrics_list)
        completed.append("emotion")

    # ── Merge and insert sentiments ────────────────────────────────────────────
    if sentiment_results or emotion_results:
        sentiment_records = []
        for i, sid in enumerate(ids):
            rec: dict = {"song_id": sid}
            if i < len(sentiment_results):
                rec.update(sentiment_results[i])
            else:
                rec.update({"overall_sentiment": "neutral", "sentiment_score": 0.5})
            if i < len(emotion_results):
                rec["emotion_label"] = emotion_results[i]["emotion_label"]
                rec["emotion_scores"] = emotion_results[i]["emotion_scores"]
            else:
                rec["emotion_label"] = "neutral"
                rec["emotion_scores"] = {}
            sentiment_records.append(rec)
        db.insert_sentiments(sentiment_records)

    # ── Aspect sentiment ───────────────────────────────────────────────────────
    if not args.skip_aspects:
        from src.analysis.aspect_sentiment import AspectSentimentAnalyzer
        logger.info("Running aspect sentiment analysis …")
        analyzer = AspectSentimentAnalyzer()
        song_dicts = songs_df.to_dict(orient="records")
        aspect_results = analyzer.analyze_batch(song_dicts)
        aspect_records = [
            {"song_id": sid, **aspect}
            for sid, aspects in zip(ids, aspect_results)
            for aspect in aspects
        ]
        db.insert_aspect_sentiments(aspect_records)
        completed.append("aspects")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"{'Songs analyzed:':<30} {len(songs_df)}")
    print(f"{'Completed steps:':<30} {', '.join(completed) or 'none'}")
    print(f"{'Topics discovered:':<30} {len(topic_keywords)}")
    print("=" * 50)
    logger.info("=== Analysis pipeline complete ===")


if __name__ == "__main__":
    main()
