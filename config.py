"""Central configuration and constants for the Song Lyrics Analyzer."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Directory roots ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ── API keys ───────────────────────────────────────────────────────────────────
GENIUS_API_KEY: str = os.getenv("GENIUS_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

# ── Feature flags ──────────────────────────────────────────────────────────────
DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")
MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")

# ── Domain constants ───────────────────────────────────────────────────────────
DECADES: list[str] = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
GENRES: list[str] = ["pop", "rock", "hip-hop", "country", "r&b", "electronic"]
TARGET_SONGS_PER_CELL: int = 140

DECADE_YEAR_RANGES: dict[str, tuple[int, int]] = {
    "1960s": (1960, 1969),
    "1970s": (1970, 1979),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2029),
}

# ── Model identifiers ──────────────────────────────────────────────────────────
GROQ_MODEL: str = "llama-3.3-70b-versatile"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
ASPECT_MODEL: str = "facebook/bart-large-mnli"

# ── Aspect labels ──────────────────────────────────────────────────────────────
ASPECTS: list[str] = [
    "love and relationships",
    "identity and self-expression",
    "social issues and politics",
    "celebration and party",
    "loss and grief",
    "money and success",
    "nature and environment",
    "spirituality and faith",
]

# ── Logger setup ───────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)
logger.add(
    DATA_DIR / "analyzer.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}",
)
