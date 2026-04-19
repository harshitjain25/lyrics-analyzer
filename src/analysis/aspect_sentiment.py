"""Aspect-level sentiment analysis via zero-shot classification."""

from __future__ import annotations

import random
from typing import Any

from config import ASPECT_MODEL, ASPECTS, MOCK_MODE, logger


class AspectSentimentAnalyzer:
    """Classifies each sentence into a lyrical aspect, then measures sentiment."""

    def __init__(self) -> None:
        """Load zero-shot and sentiment pipelines (skipped in MOCK_MODE)."""
        self._zero_shot = None
        self._sentiment = None
        if not MOCK_MODE:
            from transformers import pipeline
            self._zero_shot = pipeline("zero-shot-classification", model=ASPECT_MODEL)
            self._sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                truncation=True,
                max_length=512,
            )
            logger.info(f"Aspect + sentiment pipelines loaded ({ASPECT_MODEL})")

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, lyrics: str) -> list[dict[str, Any]]:
        """Classify sentences by aspect and run sentiment per aspect group.

        Returns [{aspect, sentiment, score, example_sentence}] for aspects
        with ≥3 qualifying sentences.
        """
        if MOCK_MODE:
            return self._mock_aspects()

        try:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(lyrics)
        except Exception:
            sentences = [s.strip() for s in lyrics.split("\n") if s.strip()]

        if not sentences:
            return []

        # Classify each sentence
        aspect_groups: dict[str, list[str]] = {a: [] for a in ASPECTS}
        try:
            predictions = self._zero_shot(sentences, ASPECTS, multi_label=False)
            if not isinstance(predictions, list):
                predictions = [predictions]
            for sent, pred in zip(sentences, predictions):
                top_label = pred["labels"][0]
                top_score = pred["scores"][0]
                if top_score >= 0.4:
                    aspect_groups[top_label].append(sent)
        except Exception as exc:
            logger.warning(f"Zero-shot classification failed: {exc}")
            return []

        # Run sentiment on groups with ≥3 sentences
        results = []
        for aspect, sents in aspect_groups.items():
            if len(sents) < 3:
                continue
            combined = " ".join(sents[:10])
            try:
                pred = self._sentiment(combined[:512])[0]
                label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
                sentiment = label_map.get(pred["label"], pred["label"].lower())
                results.append({
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "score": round(pred["score"], 4),
                    "example_sentence": sents[0],
                })
            except Exception as exc:
                logger.warning(f"Sentiment for aspect '{aspect}' failed: {exc}")

        return results

    def analyze_batch(
        self, songs: list[dict[str, Any]], batch_size: int = 16
    ) -> list[list[dict[str, Any]]]:
        """Analyse aspects for a list of song dicts (must have 'clean_lyrics' key).

        Returns a list of per-song aspect results.
        """
        if MOCK_MODE:
            return [self._mock_aspects() for _ in songs]

        results = []
        for i, song in enumerate(songs):
            lyrics = song.get("clean_lyrics", "")
            try:
                results.append(self.analyze(lyrics))
            except Exception as exc:
                logger.warning(f"Aspect analysis failed for song {i}: {exc}")
                results.append([])
        return results

    # ── Mock helpers ───────────────────────────────────────────────────────────

    def _mock_aspects(self) -> list[dict[str, Any]]:
        chosen = random.sample(ASPECTS, k=random.randint(2, 3))
        return [
            {
                "aspect": a,
                "sentiment": random.choice(["positive", "negative", "neutral"]),
                "score": round(random.uniform(0.5, 0.92), 4),
                "example_sentence": f"Example sentence about {a}.",
            }
            for a in chosen
        ]
