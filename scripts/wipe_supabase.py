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
# Use appropriate filter per table — integer PK vs text PK
plans = [
    ("aspect_sentiments", "id",      "int"),
    ("sentiments",        "song_id", "text"),
    ("song_topics",       "id",      "int"),
    ("topics",            "id",      "int"),
    ("songs",             "id",      "text"),
]
for table, col, kind in plans:
    try:
        q = db._client.table(table).delete()
        q = q.gte(col, 0) if kind == "int" else q.neq(col, "__impossible__")
        q.execute()
        logger.info(f"Wiped table: {table}")
    except Exception as e:
        logger.warning(f"Could not wipe {table}: {e}")

logger.info("All tables wiped. Safe to run run_collection.py fresh.")
