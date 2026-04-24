"""
Reassign topics using cosine similarity between song embeddings and
predefined thematic label embeddings. Runs locally in ~30 seconds.
Replaces the coarse BERTopic clusters with 8 meaningful topic buckets.
"""
from __future__ import annotations
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import logger
from src.database.db_manager import DBManager

EMBEDDINGS_PATH = Path("data/embeddings/embeddings.npy")
IDS_PATH        = Path("data/embeddings/song_ids.json")

TOPICS = [
    {"id": 0, "label": "Love and Romance",
     "keywords": ["love", "heart", "baby", "kiss", "forever", "together", "darling", "hold"],
     "description": "Songs about romantic love, desire, and connection."},
    {"id": 1, "label": "Heartbreak and Loss",
     "keywords": ["goodbye", "tears", "alone", "broken", "miss", "pain", "cry", "lost"],
     "description": "Songs dealing with breakups, grief, and emotional pain."},
    {"id": 2, "label": "Party and Celebration",
     "keywords": ["dance", "party", "tonight", "feel good", "move", "jump", "celebrate", "fun"],
     "description": "Upbeat songs about dancing, partying, and having fun."},
    {"id": 3, "label": "Identity and Self-Expression",
     "keywords": ["myself", "who I am", "free", "believe", "proud", "real", "rise", "power"],
     "description": "Songs about personal identity, empowerment, and authenticity."},
    {"id": 4, "label": "Social Issues and Protest",
     "keywords": ["fight", "justice", "change", "war", "freedom", "system", "people", "streets"],
     "description": "Songs addressing social commentary, politics, and protest."},
    {"id": 5, "label": "Faith and Spirituality",
     "keywords": ["god", "soul", "heaven", "pray", "spirit", "blessed", "light", "grace"],
     "description": "Songs exploring religion, faith, and spiritual themes."},
    {"id": 6, "label": "Money Power and Success",
     "keywords": ["money", "hustle", "grind", "rich", "success", "top", "king", "boss"],
     "description": "Songs about ambition, wealth, and achieving success."},
    {"id": 7, "label": "Nostalgia and Growing Up",
     "keywords": ["remember", "yesterday", "home", "childhood", "old days", "time", "young", "past"],
     "description": "Songs about memories, nostalgia, and the passage of time."},
]


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between every row of a and every row of b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def main() -> None:
    from sentence_transformers import SentenceTransformer

    logger.info("Loading existing song embeddings …")
    embeddings = np.load(str(EMBEDDINGS_PATH))
    with open(IDS_PATH) as f:
        song_ids = json.load(f)
    logger.info(f"Loaded {len(song_ids)} song embeddings  shape={embeddings.shape}")

    logger.info("Encoding topic labels …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_texts = [t["label"] + ": " + ", ".join(t["keywords"]) for t in TOPICS]
    topic_embeddings = model.encode(topic_texts, show_progress_bar=False)

    logger.info("Computing cosine similarity …")
    sim = cosine_similarity_matrix(embeddings, topic_embeddings)  # (N, 8)
    assigned_ids   = sim.argmax(axis=1).tolist()
    assigned_probs = sim.max(axis=1).tolist()

    # ── Wipe old topic data and insert new ────────────────────────────────────
    db = DBManager()

    logger.info("Wiping old song_topics and topics …")
    db._client.table("song_topics").delete().gte("id", 0).execute()
    db._client.table("topics").delete().gte("id", 0).execute()

    logger.info("Inserting 8 topics …")
    topic_rows = [
        {
            "topic_id": t["id"],
            "label": t["label"],
            "keywords": t["keywords"],
            "description": t["description"],
            "cultural_note": "",
        }
        for t in TOPICS
    ]
    db._client.table("topics").upsert(topic_rows).execute()

    logger.info("Inserting song_topic assignments …")
    assignments = [
        {"song_id": sid, "topic_id": tid, "probability": round(float(prob), 4)}
        for sid, tid, prob in zip(song_ids, assigned_ids, assigned_probs)
    ]
    # Batch insert in chunks of 500
    for i in range(0, len(assignments), 500):
        db._client.table("song_topics").upsert(assignments[i:i+500]).execute()

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(assigned_ids)
    print("\n" + "=" * 50)
    print("TOPIC DISTRIBUTION")
    print("=" * 50)
    for t in TOPICS:
        bar = "█" * (counts[t["id"]] // 20)
        print(f"  {t['label']:<35} {counts[t['id']]:>4}  {bar}")
    print("=" * 50)
    logger.info("Topic remapping complete — refresh your dashboard.")


if __name__ == "__main__":
    main()
