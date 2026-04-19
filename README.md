# Song Lyrics Thematic Analyzer

NLP-powered analysis of lyrical themes, sentiment, and emotions across 6 decades of popular music.

```
┌─────────────────────────────────────────────────────────┐
│                  Song Lyrics Analyzer                   │
├──────────────┬──────────────────────────┬───────────────┤
│ Genius API   │  BERTopic + HuggingFace  │  Groq LLM     │
│ (collection) │  (NLP analysis)          │  (labeling)   │
└──────┬───────┴────────────┬─────────────┴───────┬───────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────┐    ┌──────────────────┐   ┌─────────────┐
│  Supabase   │◄───│  Pipeline scripts│   │  Streamlit  │
│ (PostgreSQL)│    │  run_collection  │   │  Dashboard  │
└─────────────┘    │  run_analysis    │   └─────────────┘
                   └──────────────────┘
```

## Tech Stack
- **LLM:** Groq API — llama-3.3-70b-versatile
- **Topics:** BERTopic + all-MiniLM-L6-v2
- **Sentiment:** cardiffnlp/twitter-roberta-base-sentiment-latest
- **Emotion:** j-hartmann/emotion-english-distilroberta-base
- **Aspect:** facebook/bart-large-mnli (zero-shot)
- **Lyrics:** Genius API via lyricsgenius
- **Database:** Supabase (PostgreSQL)
- **Dashboard:** Streamlit Community Cloud

## Prerequisites
- Python 3.10+
- Free accounts: [Genius](https://genius.com/api-clients), [Groq](https://console.groq.com), [Supabase](https://supabase.com), [Streamlit Community Cloud](https://streamlit.io/cloud)

## Local Setup

```bash
git clone <your-repo>
cd song_lyrics_analyzer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

## Database Migration

1. Open your Supabase project → SQL Editor
2. Paste and run the contents of `migrations/001_init.sql`

## Run the Pipeline

```bash
# 1. Collect lyrics from Genius
python scripts/run_collection.py

# 2. Run NLP analysis (BERTopic, sentiment, emotion, aspects)
python scripts/run_analysis.py

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

## Demo Mode (no API keys needed)

```bash
DEMO_MODE=true streamlit run dashboard/app.py
```

Or set `DEMO_MODE=true` in your `.env` file. Uses `data/processed/sample_100.csv`.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, set **Main file path:** `dashboard/app.py`
4. Under **Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key"
   SUPABASE_URL = "your_url"
   SUPABASE_KEY = "your_key"
   DEMO_MODE = "true"
   ```
5. Deploy

## Environment Variables

| Variable | Description |
|---|---|
| `GENIUS_API_KEY` | Genius API token (collection only) |
| `GROQ_API_KEY` | Groq API key (AI explanations) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/publishable key |
| `DEMO_MODE` | `true` = use sample data, skip DB |
| `MOCK_MODE` | `true` = skip all API/model calls |

## Mock Mode (development, no downloads)

```bash
MOCK_MODE=true python scripts/run_collection.py
MOCK_MODE=true python scripts/run_analysis.py
```

## Known Limitations
- BERTopic requires ~500+ songs for meaningful topics
- First run downloads ~2GB of HuggingFace models
- Genius API rate-limits at ~5 req/s
- Zero-shot aspect classification is slow on CPU
