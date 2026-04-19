"""Groq-powered topic labeler — assigns human-readable names to BERTopic clusters."""

from __future__ import annotations

import json
from typing import Any

from config import GROQ_API_KEY, GROQ_MODEL, MOCK_MODE, logger


class TopicLabeler:
    """Uses Groq (Llama 3.3 70B) to generate labels for BERTopic topics."""

    def __init__(self) -> None:
        """Initialise the Groq client (skipped in MOCK_MODE)."""
        self._client = None
        if not MOCK_MODE:
            try:
                import groq
                self._client = groq.Groq(api_key=GROQ_API_KEY)
                logger.info("Groq client initialised for topic labeling")
            except Exception as exc:
                logger.error(f"Failed to initialise Groq client: {exc}")
                raise

    def label_topics(self, topics_dict: dict[int, list[str]]) -> dict[int, dict[str, str]]:
        """Generate labels, descriptions, and cultural notes for all topics in one call.

        Args:
            topics_dict: {topic_id: [keyword1, keyword2, …]} mapping.

        Returns:
            {topic_id: {label, description, cultural_note}}.
        """
        if MOCK_MODE:
            return self._mock_labels(topics_dict)

        topics_text = "\n".join(
            f"Topic {tid}: {', '.join(keywords)}"
            for tid, keywords in sorted(topics_dict.items())
        )
        n = len(topics_dict)

        user_prompt = (
            f"Here are {n} lyrical themes discovered from a corpus of 5000 songs "
            f"spanning 6 decades and 6 genres. Each theme is represented by its top keywords. "
            f"For each topic, provide: a short label (3-5 words max), a 2-sentence description "
            f"of what this theme represents, and a brief cultural note about when/where this "
            f"theme tends to appear in music history.\n\n"
            f"{topics_text}\n\n"
            f'Return ONLY valid JSON in this exact format, no markdown, no preamble:\n'
            f'{{"0": {{"label": "...", "description": "...", "cultural_note": "..."}}, ...}}'
        )

        result = self._call_groq(user_prompt)
        if result is None:
            return self._fallback_labels(topics_dict)
        return {int(k): v for k, v in result.items()}

    # ── Private helpers ────────────────────────────────────────────────────────

    def _call_groq(self, user_prompt: str, strict: bool = False) -> dict[str, Any] | None:
        """Send a prompt to Groq and parse the JSON response."""
        system = (
            "You are an expert musicologist and cultural critic specializing "
            "in popular music from the 1960s to 2020s."
        )
        if strict:
            user_prompt += "\n\nIMPORTANT: Your entire response must be valid JSON only. No text before or after."

        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError:
            if not strict:
                logger.warning("JSON parse error from Groq — retrying with stricter prompt")
                return self._call_groq(user_prompt, strict=True)
            logger.error("Could not parse JSON from Groq response after retry")
            return None
        except Exception as exc:
            logger.warning(f"Groq API error in label_topics: {exc}")
            return None

    def _fallback_labels(self, topics_dict: dict[int, list[str]]) -> dict[int, dict[str, str]]:
        """Return minimal fallback labels when the API call fails."""
        return {
            tid: {"label": f"Topic {tid}", "description": "Theme cluster", "cultural_note": ""}
            for tid in topics_dict
        }

    def _mock_labels(self, topics_dict: dict[int, list[str]]) -> dict[int, dict[str, str]]:
        mock_data = [
            ("Heartbreak & Loss", "Songs exploring emotional pain after relationships end.", "Dominant in 1970s soft rock and 1990s pop ballads."),
            ("Party & Celebration", "Anthems about dancing, fun, and living in the moment.", "Peaked during the disco era and again in 2010s EDM."),
            ("Social Justice", "Lyrics addressing inequality, protest, and systemic change.", "Central to 1960s folk, 1980s hip-hop, and 2010s R&B."),
            ("Love & Romance", "Themes of falling in love, longing, and affection.", "The most enduring theme across all decades and genres."),
            ("Identity & Self", "Songs about personal growth, authenticity, and self-expression.", "Rose prominently in 1990s grunge and 2010s pop."),
            ("Money & Success", "Materialism, ambition, and the pursuit of wealth.", "Central to hip-hop from the 1990s onward."),
            ("Nostalgia & Memory", "Reflections on the past, hometown, and lost times.", "Prevalent in country music and 1980s rock ballads."),
            ("Faith & Spirituality", "Religious themes, hope, and transcendence.", "Rooted in gospel and weaves through soul and country."),
        ]
        return {
            tid: {
                "label": mock_data[tid % len(mock_data)][0],
                "description": mock_data[tid % len(mock_data)][1],
                "cultural_note": mock_data[tid % len(mock_data)][2],
            }
            for tid in topics_dict
        }
