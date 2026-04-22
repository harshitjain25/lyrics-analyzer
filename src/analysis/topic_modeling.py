"""BERTopic-based topic modeler for song lyrics."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from config import EMBEDDINGS_DIR, MOCK_MODE, logger

_MODEL_PATH = EMBEDDINGS_DIR / "bertopic_model"
_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
_IDS_PATH = EMBEDDINGS_DIR / "song_ids.json"


class TopicModeler:
    """Fits BERTopic on a corpus of lyrics and persists artefacts to disk."""

    def __init__(self) -> None:
        """Initialise BERTopic with all sub-models (skipped in MOCK_MODE)."""
        self._model = None
        self._embeddings: np.ndarray | None = None
        if not MOCK_MODE:
            self._build_model()

    def _build_model(self) -> None:
        """Construct the BERTopic pipeline."""
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from umap import UMAP

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=30, metric="euclidean", prediction_data=True)

        self._model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True,
        )
        logger.info("BERTopic model initialised")

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, lyrics_list: list[str], ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fit BERTopic on the corpus and return per-song topic assignments.

        Returns a dict mapping song_id → {topic_id, probability}.
        Side effects: saves model, embeddings.npy, and song_ids.json.
        """
        if MOCK_MODE:
            return self._mock_assignments(ids)

        logger.info(f"Fitting BERTopic on {len(lyrics_list)} songs")
        topics, probs = self._model.fit_transform(lyrics_list)

        # Persist artefacts
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self._model.save(str(_MODEL_PATH), serialization="safetensors", save_ctfidf=True)
        logger.info(f"BERTopic model saved to {_MODEL_PATH}")

        # Save embeddings computed during fit_transform
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embeddings = st_model.encode(lyrics_list, show_progress_bar=True)
            np.save(str(_EMBEDDINGS_PATH), self._embeddings)
            logger.info(f"Embeddings saved to {_EMBEDDINGS_PATH}")
        except Exception as exc:
            logger.warning(f"Could not save embeddings: {exc}")

        with open(_IDS_PATH, "w") as f:
            json.dump(ids, f)

        probs_list = probs.tolist() if hasattr(probs, "tolist") else list(probs)
        return {
            song_id: {"topic_id": int(t), "probability": float(p)}
            for song_id, t, p in zip(ids, topics, probs_list)
        }

    def get_topic_keywords(self) -> dict[int, list[str]]:
        """Return the top-10 keywords for each discovered topic.

        Returns {topic_id: [keyword1, …, keyword10]}.
        """
        if MOCK_MODE or self._model is None:
            return {i: [f"keyword_{i}_{j}" for j in range(10)] for i in range(8)}

        topic_info = self._model.get_topics()
        return {
            topic_id: [word for word, _ in words[:10]]
            for topic_id, words in topic_info.items()
            if topic_id != -1
        }

    def load_model(self, path: str | Path) -> None:
        """Load a previously saved BERTopic model from disk."""
        if MOCK_MODE:
            return
        from bertopic import BERTopic
        self._model = BERTopic.load(str(path))
        logger.info(f"BERTopic model loaded from {path}")

    # ── Mock helpers ───────────────────────────────────────────────────────────

    def _mock_assignments(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return random topic assignments without any model inference."""
        return {
            song_id: {"topic_id": random.randint(0, 7), "probability": round(random.uniform(0.4, 0.95), 3)}
            for song_id in ids
        }
