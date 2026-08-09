"""Server-side speech-to-text via faster-whisper (mockable for tests)."""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

# Optional override: Callable[[np.ndarray, int], str]
_transcribe_override: Callable[[np.ndarray, int], str] | None = None

# Classic Whisper hallucinations on short / trailing silence.
_HALLUCINATION_PHRASES = (
    "end of recording",
    "end of the recording",
    "thanks for watching",
    "thank you for watching",
    "thank you for listening",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "see you next time",
    "see you in the next video",
    "don't forget to subscribe",
    "thanks for listening",
    "amara.org",
    "subtitles by",
    "captioned by",
)


def set_transcribe_override(fn: Callable[[np.ndarray, int], str] | None) -> None:
    """Inject a deterministic ASR function for unit tests (avoids model download)."""
    global _transcribe_override
    _transcribe_override = fn


def scrub_asr_hallucinations(text: str) -> str:
    """
    Strip common Whisper hallucination phrases (often appear in brackets or alone).
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""

    # Remove bracketed / parenthetical hallucination snippets.
    def _bracket_repl(match: re.Match[str]) -> str:
        inner = match.group(1).strip().lower()
        for phrase in _HALLUCINATION_PHRASES:
            if phrase in inner or inner in phrase:
                return " "
        return match.group(0)

    cleaned = re.sub(r"[\[\(\{]([^\]\)\}]{0,80})[\]\)\}]", _bracket_repl, cleaned)

    # Drop whole-utterance or trailing hallucination sentences.
    lowered = cleaned.lower()
    for phrase in _HALLUCINATION_PHRASES:
        if lowered == phrase or lowered == f"{phrase}.":
            return ""
        # Trailing "... end of recording" / "End of recording."
        pattern = rf"(?:^|[\s.,;:!?]){re.escape(phrase)}(?:[.!?…]*\s*)?$"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        lowered = cleaned.lower()

    cleaned = " ".join(cleaned.split()).strip(" ,;:-")
    return cleaned


class WhisperASR:
    """Lazy-loaded faster-whisper wrapper."""

    def __init__(
        self,
        model_size: str = "small.en",
        enabled: bool = True,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.enabled = bool(enabled)
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self.enabled and (_transcribe_override is not None or self._ensure_model() is not None)

    def _ensure_model(self) -> Any:
        if not self.enabled:
            return None
        if _transcribe_override is not None:
            return True
        if self._model is not None:
            return self._model
        if self._load_error:
            return None
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            print(f"[ASR] Loaded faster-whisper model '{self.model_size}' ({self.device}/{self.compute_type}).")
            return self._model
        except Exception as e:
            self._load_error = str(e)
            print(f"[ASR] Failed to load faster-whisper: {e}")
            return None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Return transcript text for mono float32 PCM, or empty string."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / 32768.0

        if _transcribe_override is not None:
            raw = (_transcribe_override(samples, sample_rate) or "").strip()
            return scrub_asr_hallucinations(raw)

        model = self._ensure_model()
        if model is None:
            return ""

        try:
            segments, _info = model.transcribe(
                samples,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                without_timestamps=True,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
            )
            parts = [seg.text.strip() for seg in segments if getattr(seg, "text", None)]
            joined = " ".join(p for p in parts if p).strip()
            return scrub_asr_hallucinations(joined)
        except Exception as e:
            print(f"[ASR] Transcription failed: {e}")
            return ""

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "engine": "faster_whisper",
            "model": self.model_size,
            "available": self.available,
            "load_error": self._load_error,
        }
