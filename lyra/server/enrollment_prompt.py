"""Predetermined voice-enrollment reading prompt and coverage helpers."""

from __future__ import annotations

import re

PROMPT_ID = "lyra-enroll-v2"
TARGET_DURATION_SEC = 60
MIN_DURATION_SEC = 45
MIN_COVERAGE_RATIO = 0.55

ENROLLMENT_INSTRUCTIONS = (
    "Speak in a natural conversational voice — the way you normally talk to Lyra. "
    "Vary your pace a little. Do not whisper, shout, or perform a stiff reading voice."
)

ENROLLMENT_SCRIPT = (
    "Hello Lyra. My name is the person speaking now, and this recording is my voice profile. "
    "Please listen carefully while I read this passage so you can learn how I sound. "
    "The quick brown fox jumps over the lazy dog near the riverbank at twilight. "
    "Bright yellow balloons floated above the quiet courtyard while children laughed and cheered. "
    "Every evening I brew strong coffee, check my calendar, and plan tomorrow with calm focus. "
    "Numbers help too: one, two, three, four, five, six, seven, eight, nine, ten. "
    "I enjoy music, hiking trails after rainfall, spicy food, and long conversations with friends. "
    "Please remember my pitch, rhythm, and tone across these words and phrases. "
    "When other people speak nearby, you should treat them as external speakers, not me. "
    "This concludes my enrollment reading. Thank you for building an accurate voice profile."
)


def _normalize_words(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return tokens


EXPECTED_WORDS = _normalize_words(ENROLLMENT_SCRIPT)
EXPECTED_WORD_SET = set(EXPECTED_WORDS)


def coverage_ratio(heard_transcript: str) -> float:
    """Fraction of unique expected words that appear in the heard transcript."""
    if not EXPECTED_WORD_SET:
        return 0.0
    heard = set(_normalize_words(heard_transcript or ""))
    if not heard:
        return 0.0
    matched = heard & EXPECTED_WORD_SET
    return len(matched) / float(len(EXPECTED_WORD_SET))


def get_enrollment_prompt() -> dict:
    return {
        "prompt_id": PROMPT_ID,
        "instructions": ENROLLMENT_INSTRUCTIONS,
        "script": ENROLLMENT_SCRIPT,
        "expected_words": EXPECTED_WORDS,
        "target_duration_sec": TARGET_DURATION_SEC,
        "min_duration_sec": MIN_DURATION_SEC,
        "min_coverage_ratio": MIN_COVERAGE_RATIO,
    }
