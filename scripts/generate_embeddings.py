"""Fast helper: compute embeddings.npy + song_ids.json from Supabase data."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDINGS_DIR, logger
from src.database.db_manager import DBManager

logger.info("Loading songs from Supabase...")
db = DBManager()
songs_df = db.get_songs()
logger.info(f"Loaded {len(songs_df)} songs")

logger.info("Loading sentence-transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

logger.info("Encoding lyrics...")
embeddings = model.encode(songs_df["clean_lyrics"].tolist(), show_progress_bar=True)

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
np.save(str(EMBEDDINGS_DIR / "embeddings.npy"), embeddings)
with open(EMBEDDINGS_DIR / "song_ids.json", "w") as f:
    json.dump(songs_df["id"].tolist(), f)

logger.info(f"Saved embeddings.npy ({embeddings.shape}) and song_ids.json")
