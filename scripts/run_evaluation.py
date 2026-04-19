"""Evaluation script: BERTopic coherence + sentiment accuracy/F1."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import EMBEDDINGS_DIR, PROCESSED_DIR, logger

_GOLD_CSV = PROCESSED_DIR / "sentiment_gold.csv"
_IDS_JSON = EMBEDDINGS_DIR / "song_ids.json"
_MODEL_PATH = EMBEDDINGS_DIR / "bertopic_model"


def evaluate_topic_coherence() -> float | None:
    """Compute BERTopic CV coherence score using gensim."""
    try:
        import json
        import numpy as np
        from bertopic import BERTopic
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel

        if not _MODEL_PATH.exists():
            logger.warning(f"BERTopic model not found at {_MODEL_PATH}")
            return None

        model = BERTopic.load(str(_MODEL_PATH))
        topics = model.get_topics()
        topic_words = [
            [word for word, _ in words[:10]]
            for tid, words in topics.items()
            if tid != -1 and words
        ]

        if not _IDS_JSON.exists():
            logger.warning("song_ids.json not found, cannot compute coherence")
            return None

        from src.database.db_manager import DBManager
        db = DBManager()
        songs_df = db.get_songs()
        texts = [lyrics.split() for lyrics in songs_df["clean_lyrics"].tolist()]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        cm = CoherenceModel(
            topics=topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        score = cm.get_coherence()
        return score
    except Exception as exc:
        logger.error(f"Coherence evaluation failed: {exc}")
        return None


def evaluate_sentiment() -> dict | None:
    """Compute accuracy and macro F1 against the gold sentiment labels."""
    if not _GOLD_CSV.exists():
        logger.warning(f"Gold labels not found at {_GOLD_CSV}")
        return None
    try:
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        gold_df = pd.read_csv(_GOLD_CSV)

        from src.database.db_manager import DBManager
        db = DBManager()
        data = db.get_dashboard_data()
        sentiments_df = data["sentiments_df"]

        merged = gold_df.merge(sentiments_df, left_on="id", right_on="song_id", how="inner")
        if merged.empty:
            logger.warning("No overlapping IDs between gold labels and predictions")
            return None

        y_true = merged["true_sentiment"].tolist()
        y_pred = merged["overall_sentiment"].tolist()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0)
        return {"accuracy": acc, "macro_f1": f1, "report": report, "n_samples": len(merged)}
    except Exception as exc:
        logger.error(f"Sentiment evaluation failed: {exc}")
        return None


def main() -> None:
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print("\n[1] BERTopic Coherence (CV)")
    score = evaluate_topic_coherence()
    if score is not None:
        print(f"  Coherence Score (CV): {score:.4f}")
        if score >= 0.55:
            print("  ✓ Good coherence (≥0.55)")
        elif score >= 0.40:
            print("  ~ Acceptable coherence (0.40–0.55)")
        else:
            print("  ✗ Low coherence (<0.40) — consider adjusting HDBSCAN parameters")
    else:
        print("  Skipped (model or data not found)")

    print("\n[2] Sentiment Classification")
    result = evaluate_sentiment()
    if result:
        print(f"  Samples evaluated:  {result['n_samples']}")
        print(f"  Accuracy:           {result['accuracy']:.4f}")
        print(f"  Macro F1:           {result['macro_f1']:.4f}")
        print("\n  Per-class report:")
        for line in result["report"].splitlines():
            print(f"    {line}")
    else:
        print("  Skipped (gold labels or predictions not found)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
