"""Accumulate ambient PCM into VAD-bounded utterances for server-side ASR."""

from __future__ import annotations

from typing import Any

import numpy as np


class UtteranceBuffer:
    """
    Collects speech-framed PCM and emits a completed utterance when silence
    hangover elapses or max duration is reached.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_hangover_sec: float = 0.6,
        max_utterance_sec: float = 15.0,
        min_utterance_sec: float = 0.35,
    ):
        self.sample_rate = int(sample_rate)
        self.silence_hangover_samples = int(sample_rate * silence_hangover_sec)
        self.max_utterance_samples = int(sample_rate * max_utterance_sec)
        self.min_utterance_samples = int(sample_rate * min_utterance_sec)
        self._chunks: list[np.ndarray] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._in_utterance = False

    def reset(self) -> None:
        self._chunks.clear()
        self._speech_samples = 0
        self._silence_samples = 0
        self._in_utterance = False

    def _concat(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).astype(np.float32)

    def push(self, audio: np.ndarray, is_speech: bool) -> np.ndarray | None:
        """
        Append a PCM chunk. Returns a completed utterance array when ready,
        otherwise None.
        """
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None

        if is_speech:
            if not self._in_utterance:
                self._in_utterance = True
                self._chunks = []
                self._speech_samples = 0
                self._silence_samples = 0
            self._chunks.append(samples)
            self._speech_samples += samples.size
            self._silence_samples = 0
            if self._speech_samples >= self.max_utterance_samples:
                return self._finalize()
            return None

        # Non-speech
        if not self._in_utterance:
            return None

        self._chunks.append(samples)
        self._silence_samples += samples.size
        if self._silence_samples >= self.silence_hangover_samples:
            return self._finalize()
        return None

    def _finalize(self) -> np.ndarray | None:
        utterance = self._concat()
        speech_len = self._speech_samples
        self.reset()
        if speech_len < self.min_utterance_samples:
            return None
        # Trim trailing silence beyond a small pad.
        pad = int(self.sample_rate * 0.15)
        keep = min(len(utterance), speech_len + pad)
        return utterance[:keep]

    def flush(self) -> np.ndarray | None:
        """Force-emit current utterance if any (e.g. on disconnect)."""
        if not self._in_utterance:
            return None
        return self._finalize()

    def status(self) -> dict[str, Any]:
        return {
            "in_utterance": self._in_utterance,
            "speech_samples": self._speech_samples,
            "silence_samples": self._silence_samples,
        }
