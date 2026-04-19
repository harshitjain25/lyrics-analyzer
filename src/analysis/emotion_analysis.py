"""Emotion analysis using j-hartmann/emotion-english-distilroberta-base."""

from __future__ import annotations

import random
from typing import Any

from tqdm import tqdm

from config import EMOTION_MODEL, MOCK_MODE, logger

_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
_MAX_TOKENS = 512


class EmotionAnalyzer:
    """Detects all 7 emotions with per-chunk averaging for long lyrics."""

    def __init__(self) -> None:
        """Load the emotion classification pipeline (skipped in MOCK_MODE)."""
        self._pipe = None
        if not MOCK_MODE:
            from transformers import pipeline
            self._pipe = pipeline(
                "text-classification",
                model=EMOTION_MODEL,
                top_k=None,
                truncation=True,
                max_length=_MAX_TOKENS,
            )
            logger.info(f"Emotion pipeline loaded: {EMOTION_MODEL}")

    def analyze_batch(
        self, lyrics_list: list[str], batch_size: int = 32
    ) -> list[dict[str, Any]]:
        """Analyse emotions for a list of lyrics strings.

        Returns a list of dicts: {emotion_label, emotion_scores: {emotion: float}}.
        """
        if MOCK_MODE:
            return self._mock_batch(len(lyrics_list))

        results = []
        for lyrics in tqdm(lyrics_list, desc="Emotion analysis"):
            chunks = self._chunk_text(lyrics)
            accumulated: dict[str, list[float]] = {e: [] for e in _EMOTIONS}

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i: i + batch_size]
                try:
                    batch_preds = self._pipe(batch, truncation=True, max_length=_MAX_TOKENS)
                    for preds in batch_preds:
                        for item in preds:
                            label = item["label"].lower()
                            if label in accumulated:
                                accumulated[label].append(item["score"])
                except Exception as exc:
                    logger.warning(f"Emotion inference error: {exc}")
                    continue

            avg_scores = {e: round(sum(v) / len(v), 4) if v else 0.0 for e, v in accumulated.items()}
            dominant = max(avg_scores, key=lambda k: avg_scores[k])
            results.append({"emotion_label": dominant, "emotion_scores": avg_scores})

        return results

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, words_per_chunk: int = 300) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunks.append(" ".join(words[i: i + words_per_chunk]))
        return chunks or [""]

    def _mock_batch(self, n: int) -> list[dict[str, Any]]:
        results = []
        for _ in range(n):
            raw = {e: random.uniform(0.01, 0.4) for e in _EMOTIONS}
            total = sum(raw.values())
            scores = {e: round(v / total, 4) for e, v in raw.items()}
            dominant = max(scores, key=lambda k: scores[k])
            results.append({"emotion_label": dominant, "emotion_scores": scores})
        return results
