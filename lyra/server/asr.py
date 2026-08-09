"""Server-side speech-to-text via faster-whisper (mockable for tests)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# Optional override: Callable[[np.ndarray, int], str]
_transcribe_override: Callable[[np.ndarray, int], str] | None = None


def set_transcribe_override(fn: Callable[[np.ndarray, int], str] | None) -> None:
    """Inject a deterministic ASR function for unit tests (avoids model download)."""
    global _transcribe_override
    _transcribe_override = fn


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
            return (_transcribe_override(samples, sample_rate) or "").strip()

        model = self._ensure_model()
        if model is None:
            return ""

        try:
            segments, _info = model.transcribe(
                samples,
                language="en",
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
            )
            parts = [seg.text.strip() for seg in segments if getattr(seg, "text", None)]
            return " ".join(p for p in parts if p).strip()
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
