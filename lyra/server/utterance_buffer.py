"""Accumulate ambient PCM into VAD-bounded utterances for server-side ASR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UtteranceEvent:
    """Audio emitted by the buffer for ASR (provisional or final)."""

    audio: np.ndarray
    is_final: bool


class UtteranceBuffer:
    """
    Collects speech-framed PCM and emits:
    - a provisional utterance after a short pause (fast UI update)
    - a final utterance after silence hangover (or max duration)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_hangover_sec: float = 1.2,
        provisional_silence_sec: float = 0.35,
        max_utterance_sec: float = 15.0,
        min_utterance_sec: float = 0.35,
        trailing_pad_sec: float = 0.45,
    ):
        self.sample_rate = int(sample_rate)
        self.silence_hangover_samples = int(sample_rate * silence_hangover_sec)
        self.provisional_silence_samples = int(sample_rate * provisional_silence_sec)
        self.max_utterance_samples = int(sample_rate * max_utterance_sec)
        self.min_utterance_samples = int(sample_rate * min_utterance_sec)
        self.trailing_pad_samples = int(sample_rate * trailing_pad_sec)
        self._chunks: list[np.ndarray] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._in_utterance = False
        self._provisional_emitted = False

    def reset(self) -> None:
        self._chunks.clear()
        self._speech_samples = 0
        self._silence_samples = 0
        self._in_utterance = False
        self._provisional_emitted = False

    def _concat(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).astype(np.float32)

    def _snapshot(self) -> np.ndarray | None:
        if self._speech_samples < self.min_utterance_samples:
            return None
        utterance = self._concat()
        pad = self.trailing_pad_samples
        keep = min(len(utterance), self._speech_samples + pad)
        return utterance[:keep]

    def push(self, audio: np.ndarray, is_speech: bool) -> UtteranceEvent | None:
        """
        Append a PCM chunk. Returns a provisional or final UtteranceEvent when
        ready, otherwise None.
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
                self._provisional_emitted = False
            self._chunks.append(samples)
            self._speech_samples += samples.size
            self._silence_samples = 0
            # New speech after a provisional means the prior pause was mid-sentence.
            self._provisional_emitted = False
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

        # Fast path: emit a provisional after a short pause so the rolling
        # transcript updates before the full hangover completes.
        if (
            not self._provisional_emitted
            and self._silence_samples >= self.provisional_silence_samples
        ):
            snap = self._snapshot()
            if snap is not None and snap.size > 0:
                self._provisional_emitted = True
                return UtteranceEvent(audio=snap, is_final=False)
        return None

    def _finalize(self) -> UtteranceEvent | None:
        snap = self._snapshot()
        self.reset()
        if snap is None or snap.size == 0:
            return None
        return UtteranceEvent(audio=snap, is_final=True)

    def flush(self) -> UtteranceEvent | None:
        """Force-emit current utterance if any (e.g. on disconnect)."""
        if not self._in_utterance:
            return None
        return self._finalize()

    def status(self) -> dict[str, Any]:
        return {
            "in_utterance": self._in_utterance,
            "speech_samples": self._speech_samples,
            "silence_samples": self._silence_samples,
            "provisional_emitted": self._provisional_emitted,
        }
