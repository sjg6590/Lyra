"""Ollama-backed ambient transcript cleanup (Wispr-style formatting, local)."""

from __future__ import annotations

import re
from typing import Any

from lyra.server.ollama_client import OllamaClient, strip_think_blocks

_CLEANUP_SYSTEM = (
    "You clean up speech-to-text transcripts for an ambient assistant. "
    "Rewrite the user's ASR text as a clean spoken transcript: fix obvious ASR "
    "glitches, strip filler words (um, uh, like as filler), and add light "
    "punctuation/casing. Do NOT invent facts, names, or content that was not said. "
    "Reply with ONLY the cleaned transcript text."
)


def heuristic_cleanup(text: str) -> str:
    """Lightweight offline cleanup when Ollama is unavailable."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    # Drop standalone fillers.
    cleaned = re.sub(r"\b(um+|uh+|erm+|hmm+)\b[,.]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = cleaned + "."
    return cleaned


class TranscriptCleaner:
    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        enabled: bool = True,
        timeout_seconds: float = 8.0,
        num_predict: int = 64,
    ):
        self.ollama_client = ollama_client
        self.enabled = bool(enabled)
        self.timeout_seconds = float(timeout_seconds)
        self.num_predict = int(num_predict)

    def clean(self, text: str) -> str:
        raw = " ".join((text or "").split())
        if not raw:
            return ""
        if not self.enabled:
            return raw

        if self.ollama_client is None:
            return heuristic_cleanup(raw)

        # Temporarily tighten generation budget for ambient latency.
        prev_timeout = self.ollama_client.timeout_seconds
        prev_predict = self.ollama_client.num_predict
        prev_temp = self.ollama_client.temperature
        try:
            self.ollama_client.timeout_seconds = self.timeout_seconds
            self.ollama_client.num_predict = self.num_predict
            self.ollama_client.temperature = 0.2
            reply = self.ollama_client.chat(
                [
                    {"role": "system", "content": _CLEANUP_SYSTEM},
                    {"role": "user", "content": raw},
                ]
            )
            cleaned = strip_think_blocks(reply).strip()
            # Guard against empty or wildly longer hallucinations.
            if not cleaned or len(cleaned) > max(40, int(len(raw) * 2.5)):
                return heuristic_cleanup(raw)
            return cleaned
        except Exception as e:
            print(f"[Cleanup] Ollama cleanup failed ({e}); using heuristic.")
            return heuristic_cleanup(raw)
        finally:
            self.ollama_client.timeout_seconds = prev_timeout
            self.ollama_client.num_predict = prev_predict
            self.ollama_client.temperature = prev_temp

    def status(self) -> dict[str, Any]:
        reachable = False
        if self.ollama_client is not None:
            try:
                reachable = self.ollama_client.is_reachable()
            except Exception:
                reachable = False
        return {
            "enabled": self.enabled,
            "ollama_reachable": reachable,
        }
