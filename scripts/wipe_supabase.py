"""Wipe all data from Supabase tables — use before a fresh collection run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import logger
from src.database.db_manager import DBManager

db = DBManager()
if db._demo:
    logger.error("In DEMO_MODE — not wiping anything")
    sys.exit(1)

# Order matters: delete children before parents (FK constraints)
tables = ["aspect_sentiments", "sentiments", "song_topics", "topics", "songs"]
for t in tables:
    try:
        db._client.table(t).delete().neq("id" if t != "sentiments" else "song_id", "__impossible__").execute()
        logger.info(f"Wiped table: {t}")
    except Exception as e:
        logger.warning(f"Could not wipe {t}: {e}")

logger.info("All tables wiped. Safe to run run_collection.py fresh.")
