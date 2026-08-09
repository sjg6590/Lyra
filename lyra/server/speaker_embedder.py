"""ONNX ECAPA speaker embedding backend (WeSpeaker via speakeronnx)."""

from __future__ import annotations

from typing import Callable

import numpy as np

MODEL_ID = "wespeaker-ecapa512"
EMBED_DIM = 192
SAMPLE_RATE = 16000

# Optional override for tests: Callable[[np.ndarray, int], np.ndarray]
_embed_fn_override: Callable[[np.ndarray, int], np.ndarray] | None = None
_embedder = None


def set_embed_fn_override(fn: Callable[[np.ndarray, int], np.ndarray] | None) -> None:
    """Inject a deterministic embedder for unit tests (avoids model download)."""
    global _embed_fn_override
    _embed_fn_override = fn


def reset_embedder() -> None:
    """Clear cached ONNX session (used in tests)."""
    global _embedder
    _embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from speakeronnx import SpeakerEmbedder

        _embedder = SpeakerEmbedder(model=MODEL_ID)
    return _embedder


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(out))
    if norm > 0:
        return out / norm
    return out


def embed_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract an L2-normalized speaker embedding from mono PCM.

    Returns shape (EMBED_DIM,).
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    if np.max(np.abs(samples)) > 1.0:
        samples = samples / 32768.0

    if _embed_fn_override is not None:
        return _l2_normalize(_embed_fn_override(samples, sample_rate))

    if sample_rate != SAMPLE_RATE and sample_rate > 0:
        # Lightweight linear resample to model rate when needed.
        duration = len(samples) / float(sample_rate)
        target_len = max(1, int(duration * SAMPLE_RATE))
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.float32)

    embedder = _get_embedder()
    embedding = embedder.embed(samples)
    return _l2_normalize(embedding)
