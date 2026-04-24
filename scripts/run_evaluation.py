"""Evaluation: topic confidence scores + sentiment accuracy on gold standard."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from config import logger

# ── Gold standard: 30 songs with known sentiment ──────────────────────────────
# Format: (title_fragment, artist_fragment, expected_sentiment)
# expected_sentiment: "positive", "negative", "neutral"
GOLD_STANDARDS = [
    # Clearly positive
    ("Happy",                   "Pharrell",          "positive"),
    ("Dancing Queen",           "ABBA",              "positive"),
    ("Don't Stop Me Now",       "Queen",             "positive"),
    ("I Will Survive",          "Gloria Gaynor",     "positive"),
    ("Superstition",            "Stevie Wonder",     "positive"),
    ("Respect",                 "Aretha Franklin",   "positive"),
    ("Born to Run",             "Springsteen",       "positive"),
    ("Good Vibrations",         "Beach Boys",        "positive"),
    ("Dancing in the Street",   "Marvin Gaye",       "positive"),
    ("September",               "Earth Wind",        "positive"),
    # Clearly negative
    ("Yesterday",               "Beatles",           "negative"),
    ("Hurt",                    "Johnny Cash",       "negative"),
    ("The Sound of Silence",    "Simon",             "negative"),
    ("Bohemian Rhapsody",       "Queen",             "negative"),
    ("Black",                   "Pearl Jam",         "negative"),
    ("Losing My Religion",      "R.E.M.",            "negative"),
    ("One",                     "U2",                "negative"),
    ("Mad World",               "Tears for Fears",   "negative"),
    ("The Night They Drove",    "The Band",          "negative"),
    ("Knocking on Heaven",      "Bob Dylan",         "negative"),
    # Neutral / mixed
    ("Hotel California",        "Eagles",            "neutral"),
    ("Smells Like Teen Spirit", "Nirvana",           "neutral"),
    ("Born in the U.S.A.",      "Springsteen",       "neutral"),
    ("Fight the Power",         "Public Enemy",      "neutral"),
    ("Changes",                 "David Bowie",       "neutral"),
    ("The Times They Are",      "Bob Dylan",         "neutral"),
    ("Purple Rain",             "Prince",            "neutral"),
    ("Thriller",                "Michael Jackson",   "neutral"),
    ("Gimme Shelter",           "Rolling Stones",    "neutral"),
    ("What's Going On",         "Marvin Gaye",       "neutral"),
]


def evaluate_topics(db) -> None:
    """Evaluate topic quality via per-topic confidence and distribution."""
    print("\n[1] TOPIC QUALITY EVALUATION")
    print("-" * 50)

    data = db.get_dashboard_data()
    topics_df    = data["topics_df"]
    st_df        = data["song_topics_df"]

    if st_df.empty or topics_df.empty:
        print("  No topic data found.")
        return

    merged = st_df.merge(topics_df[["topic_id", "label"]], on="topic_id", how="left")

    print(f"  Songs with topic assignments: {len(merged)}")
    print(f"  Number of topics: {topics_df['topic_id'].nunique()}")
    print()
    print(f"  {'Topic':<40} {'Songs':>6}  {'Avg Confidence':>15}  {'Min':>6}  {'Max':>6}")
    print(f"  {'-'*40} {'-'*6}  {'-'*15}  {'-'*6}  {'-'*6}")

    overall_conf = []
    for _, topic in topics_df.sort_values("topic_id").iterrows():
        subset = merged[merged["topic_id"] == topic["topic_id"]]
        if subset.empty:
            continue
        probs = subset["probability"].astype(float)
        overall_conf.extend(probs.tolist())
        print(f"  {topic['label']:<40} {len(subset):>6}  {probs.mean():>14.3f}  {probs.min():>6.3f}  {probs.max():>6.3f}")

    print(f"\n  Overall avg confidence: {np.mean(overall_conf):.3f}")
    if np.mean(overall_conf) >= 0.6:
        print("  ✓ Good topic confidence (≥0.60)")
    else:
        print("  ~ Moderate topic confidence (<0.60)")


def evaluate_sentiment(db) -> None:
    """Match gold standard songs against model predictions and compute accuracy."""
    print("\n[2] SENTIMENT ACCURACY (Gold Standard — 30 songs)")
    print("-" * 50)

    data = db.get_dashboard_data()
    songs_df      = data["songs_df"]
    sentiments_df = data["sentiments_df"]

    if songs_df.empty or sentiments_df.empty:
        print("  No sentiment data found.")
        return

    merged = songs_df.merge(sentiments_df, left_on="id", right_on="song_id", how="inner")

    results = []
    not_found = []

    for title_frag, artist_frag, expected in GOLD_STANDARDS:
        mask = (
            merged["title"].str.contains(title_frag, case=False, na=False) &
            merged["artist"].str.contains(artist_frag, case=False, na=False)
        )
        matches = merged[mask]
        if matches.empty:
            not_found.append(f"{title_frag} — {artist_frag}")
            continue
        row = matches.iloc[0]
        predicted = row["overall_sentiment"].lower().strip()
        # Normalise: model may output "positive"/"negative"/"neutral"
        correct = predicted == expected
        results.append({
            "title":     row["title"],
            "artist":    row["artist"],
            "expected":  expected,
            "predicted": predicted,
            "correct":   correct,
        })

    if not results:
        print("  No gold standard songs found in corpus.")
        print(f"  Missing: {not_found}")
        return

    res_df  = pd.DataFrame(results)
    correct = res_df["correct"].sum()
    total   = len(res_df)
    acc     = correct / total

    print(f"  Gold songs matched in corpus: {total}/30")
    print(f"  Correct predictions:          {correct}/{total}")
    print(f"  Accuracy:                     {acc:.2%}")
    print()

    # Per-class breakdown
    for label in ["positive", "negative", "neutral"]:
        subset = res_df[res_df["expected"] == label]
        if subset.empty:
            continue
        sub_acc = subset["correct"].sum() / len(subset)
        print(f"  {label.capitalize():<10}  {subset['correct'].sum()}/{len(subset)}  ({sub_acc:.0%})")

    print()
    print(f"  {'Title':<35} {'Expected':<12} {'Predicted':<12} {'OK'}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*4}")
    for _, r in res_df.iterrows():
        ok = "✓" if r["correct"] else "✗"
        print(f"  {r['title'][:34]:<35} {r['expected']:<12} {r['predicted']:<12} {ok}")

    if not_found:
        print(f"\n  Not in corpus ({len(not_found)}): {', '.join(not_found[:5])}{'...' if len(not_found)>5 else ''}")

    # Overall assessment
    print()
    if acc >= 0.75:
        print("  ✓ Strong sentiment accuracy (≥75%)")
    elif acc >= 0.55:
        print("  ~ Acceptable sentiment accuracy (55–75%)")
    else:
        print("  ✗ Low sentiment accuracy (<55%) — model may need fine-tuning")


def main() -> None:
    from src.database.db_manager import DBManager
    db = DBManager()

    print("\n" + "=" * 60)
    print("EVALUATION REPORT — Song Lyrics Thematic Analyzer")
    print("=" * 60)

    evaluate_topics(db)
    evaluate_sentiment(db)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
