"""Sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment-latest."""

from __future__ import annotations

import random
from typing import Any

from tqdm import tqdm

from config import MOCK_MODE, SENTIMENT_MODEL, logger

_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
_MAX_TOKENS = 512


class SentimentAnalyzer:
    """Batch sentiment analyser with 512-token chunking for long lyrics."""

    def __init__(self) -> None:
        """Load the sentiment pipeline (skipped in MOCK_MODE)."""
        self._pipe = None
        if not MOCK_MODE:
            from transformers import pipeline
            self._pipe = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL,
                truncation=True,
                max_length=_MAX_TOKENS,
            )
            logger.info(f"Sentiment pipeline loaded: {SENTIMENT_MODEL}")

    def analyze_batch(
        self, lyrics_list: list[str], batch_size: int = 32
    ) -> list[dict[str, Any]]:
        """Analyse sentiment for a list of lyrics strings.

        Each input is split into 512-token chunks; scores are averaged across
        chunks before taking the argmax label.

        Returns a list of dicts: {overall_sentiment, sentiment_score}.
        """
        if MOCK_MODE:
            return self._mock_batch(len(lyrics_list))

        results = []
        for lyrics in tqdm(lyrics_list, desc="Sentiment analysis"):
            chunks = self._chunk_text(lyrics)
            chunk_scores: dict[str, list[float]] = {"negative": [], "neutral": [], "positive": []}
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i: i + batch_size]
                try:
                    preds = self._pipe(batch, truncation=True, max_length=_MAX_TOKENS)
                    for pred in preds:
                        label = _LABEL_MAP.get(pred["label"], pred["label"].lower())
                        chunk_scores[label].append(pred["score"])
                except Exception as exc:
                    logger.warning(f"Sentiment inference error: {exc}")
                    continue

            avg_scores = {
                k: (sum(v) / len(v) if v else 0.0) for k, v in chunk_scores.items()
            }
            best_label = max(avg_scores, key=lambda k: avg_scores[k])
            results.append({"overall_sentiment": best_label, "sentiment_score": round(avg_scores[best_label], 4)})

        return results

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, words_per_chunk: int = 300) -> list[str]:
        """Split text into word-based chunks that fit within token limits."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunks.append(" ".join(words[i: i + words_per_chunk]))
        return chunks or [""]

    def _mock_batch(self, n: int) -> list[dict[str, Any]]:
        labels = ["positive", "negative", "neutral"]
        return [
            {"overall_sentiment": random.choice(labels), "sentiment_score": round(random.uniform(0.5, 0.95), 4)}
            for _ in range(n)
        ]
