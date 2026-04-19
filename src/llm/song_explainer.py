"""Groq-powered streaming song explainer."""

from __future__ import annotations

from typing import Any, Generator

from config import GROQ_API_KEY, GROQ_MODEL, MOCK_MODE, logger

_MOCK_ANALYSIS = """
**Thematic Heart of the Song**

This track sits at the intersection of personal longing and universal experience. The lyrics weave a narrative about searching for meaning in an uncertain world — a theme that resonates across generations. The writer uses concrete imagery drawn from everyday life to ground what might otherwise be abstract emotional territory, making the song feel both intimate and expansive at once.

**Emotional Journey**

The song opens with a mood of quiet melancholy, the verses carrying a restrained sadness that slowly builds in intensity. By the chorus, the dominant emotion shifts toward a kind of defiant hope — a refusal to surrender to despair. This emotional arc, from vulnerability through resilience, gives the track its memorable cathartic quality. The bridge introduces a brief moment of anger before resolving back into acceptance, mirroring the real psychological journey of processing difficult emotions.

**Cultural Context**

Released during a period of significant cultural transformation, this song captured a generational mood with remarkable precision. The genre was undergoing its own evolution at the time, and this track stands as a representative artifact of that moment — blending established traditions with emerging sonic and lyrical sensibilities. It reflected anxieties and aspirations that were distinctly of its era, even as its emotional core transcends the specific historical moment.

**Lyrical Style & Distinctiveness**

What sets this song apart is the writer's command of compression — the ability to pack an entire emotional world into a few well-chosen lines. The use of second-person address draws the listener directly into the narrative, collapsing the distance between singer and audience. Metaphors are consistently drawn from the physical world (light, weather, journeys) rather than abstract concepts, giving the lyrics a tactile quality that rewards close listening. The refrain functions as both a musical anchor and a philosophical statement, gathering new meaning with each repetition.
"""


class SongExplainer:
    """Streams a 4-paragraph song analysis from Groq."""

    def __init__(self) -> None:
        """Initialise Groq client (skipped in MOCK_MODE)."""
        self._client = None
        if not MOCK_MODE:
            try:
                import groq
                self._client = groq.Groq(api_key=GROQ_API_KEY)
                logger.info("Groq client initialised for song explanation")
            except Exception as exc:
                logger.error(f"Failed to initialise Groq client: {exc}")
                raise

    def explain(
        self,
        title: str,
        artist: str,
        year: int | str,
        genre: str,
        lyrics: str,
        topic_label: str,
        topic_description: str,
        sentiment: str,
        emotion_scores: dict[str, float],
        aspect_sentiments: list[dict[str, Any]],
    ) -> Generator[str, None, None]:
        """Stream a 4-paragraph song analysis.

        Yields text chunks suitable for st.write_stream().
        """
        if MOCK_MODE:
            yield _MOCK_ANALYSIS
            return

        # Format emotion scores as a ranked list
        ranked_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        emotion_str = ", ".join(f"{e} ({score:.2f})" for e, score in ranked_emotions[:4])

        # Format aspect sentiments
        if aspect_sentiments:
            aspect_str = "; ".join(
                f"{a['aspect']} ({a['sentiment']}, {a['score']:.2f})"
                for a in aspect_sentiments[:4]
            )
        else:
            aspect_str = "none detected"

        system = (
            "You are an insightful music critic with deep knowledge of popular "
            "music history, cultural context, and lyrical analysis."
        )

        user_prompt = f"""Analyze this song:
Title: {title} | Artist: {artist} | Year: {year} | Genre: {genre}

Discovered Theme: {topic_label} — {topic_description}
Overall Sentiment: {sentiment}
Dominant Emotions: {emotion_str}
Lyrical Aspects Detected: {aspect_str}

Lyrics (excerpt, first 500 chars): {str(lyrics)[:500]}...

Write a 4-paragraph analysis covering:
1. What this song is fundamentally about thematically
2. The emotional journey and how sentiment shifts through the lyrics
3. How this song reflects the cultural moment of {year} in {genre} music
4. What makes this song's lyrical style notable or distinctive

Write in an engaging, accessible style suitable for music enthusiasts."""

        try:
            stream = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=0.7,
                max_tokens=800,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error(f"Groq streaming error in explain(): {exc}")
            yield f"\n\n*Analysis unavailable: {exc}*"
