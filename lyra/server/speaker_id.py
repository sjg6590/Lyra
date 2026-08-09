import json
import os
import time
from collections import deque
from typing import Any

import numpy as np

from lyra.server import speaker_embedder


class TargetSpeakerExtractor:
    """
    Target Speaker Extraction & Voice Biometric Diarization Module.
    Differentiates User ('Me') from Third-Party External Speakers ('Not Me')
    using WeSpeaker ECAPA-TDNN ONNX embeddings with a multi-prototype ensemble.
    """

    FEATURE_DIM = speaker_embedder.EMBED_DIM
    MODEL_ID = speaker_embedder.MODEL_ID
    MAX_PROTOTYPES = 40
    WINDOW_SEC = 1.0
    WINDOW_STEP_SEC = 0.5
    SPEECH_RMS_THRESHOLD = 0.01

    def __init__(self, profile_path: str = "user_voice_profile.json", similarity_threshold: float = 0.40):
        self.profile_path = profile_path
        self.similarity_threshold = similarity_threshold
        self.enrolled_profile: np.ndarray | None = None
        self.enrolled_prototypes: list[np.ndarray] = []
        self.enrolled_metadata: dict[str, Any] = {}

        # Continuous streaming audio ring buffer (up to 1.0s = 16000 samples at 16kHz)
        self.audio_ring_buffer: deque = deque(maxlen=16000)

        # Exponential Moving Average (EMA) of similarity scores for streaming stability
        self.ema_similarity: float | None = None
        self.ema_alpha: float = 0.3

        self.load_enrolled_profile()

    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract an L2-normalized ECAPA speaker embedding."""
        return speaker_embedder.embed_audio(audio_data, sample_rate=sample_rate)

    def _window_is_speech(self, samples: np.ndarray) -> bool:
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
        return rms >= self.SPEECH_RMS_THRESHOLD

    def _select_prototype_indices(self, candidate_indices: list[int]) -> list[int]:
        if len(candidate_indices) <= self.MAX_PROTOTYPES:
            return candidate_indices
        # Evenly subsample speech windows across the enrollment take.
        positions = np.linspace(0, len(candidate_indices) - 1, self.MAX_PROTOTYPES)
        picked = sorted({candidate_indices[int(round(p))] for p in positions})
        return picked

    def enroll_user(
        self,
        audio_data: np.ndarray,
        user_name: str = "User",
        sample_rate: int = 16000,
        prompt_id: str | None = None,
        coverage_ratio: float | None = None,
    ) -> bool:
        """
        Enrolls user voice profile by extracting speech-gated ECAPA window embeddings
        and a global mean prototype vector.
        """
        samples = audio_data.astype(np.float32)
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / 32768.0

        window_size = int(sample_rate * self.WINDOW_SEC)
        window_step = int(sample_rate * self.WINDOW_STEP_SEC)
        duration_sec = float(len(samples) / sample_rate) if sample_rate else 0.0

        speech_embeddings: list[np.ndarray] = []
        candidate_starts: list[int] = []

        if len(samples) >= window_size:
            for start in range(0, len(samples) - window_size + 1, window_step):
                sub_audio = samples[start : start + window_size]
                if self._window_is_speech(sub_audio):
                    candidate_starts.append(start)
            selected_starts = self._select_prototype_indices(candidate_starts)
            for start in selected_starts:
                sub_audio = samples[start : start + window_size]
                speech_embeddings.append(self.extract_features(sub_audio, sample_rate))
        elif self._window_is_speech(samples):
            speech_embeddings.append(self.extract_features(samples, sample_rate))

        if not speech_embeddings:
            # Fallback: embed whatever we have so enrollment does not hard-fail on quiet mics.
            speech_embeddings.append(self.extract_features(samples, sample_rate))

        stacked = np.stack(speech_embeddings, axis=0)
        global_embedding = stacked.mean(axis=0)
        norm = float(np.linalg.norm(global_embedding))
        if norm > 0:
            global_embedding = global_embedding / norm
        global_embedding = global_embedding.astype(np.float32)

        prototypes = [emb.astype(np.float32).tolist() for emb in speech_embeddings]

        self.enrolled_profile = global_embedding
        self.enrolled_prototypes = [np.array(p, dtype=np.float32) for p in prototypes]
        self.enrolled_metadata = {
            "user_name": user_name,
            "sample_rate": sample_rate,
            "feature_dim": self.FEATURE_DIM,
            "model_id": self.MODEL_ID,
            "profile_vector": global_embedding.tolist(),
            "prototypes": prototypes,
            "prompt_id": prompt_id,
            "enrollment_duration_sec": round(duration_sec, 3),
            "coverage_ratio": coverage_ratio,
            "enrolled_at": time.time(),
        }

        with open(self.profile_path, "w") as f:
            json.dump(self.enrolled_metadata, f, indent=2)

        print(
            f"[SpeakerID] User '{user_name}' voice profile enrolled "
            f"({len(prototypes)} prototypes, {self.FEATURE_DIM}D, model={self.MODEL_ID}) "
            f"saved to {self.profile_path}."
        )
        return True

    def load_enrolled_profile(self) -> bool:
        """Loads stored voice profile from file if present and valid for current model."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r") as f:
                    data = json.load(f)
                    prof_vec = data.get("profile_vector", [])
                    model_id = data.get("model_id")
                    feature_dim = data.get("feature_dim", len(prof_vec))
                    if (
                        len(prof_vec) == self.FEATURE_DIM
                        and feature_dim == self.FEATURE_DIM
                        and model_id == self.MODEL_ID
                    ):
                        self.enrolled_metadata = data
                        self.enrolled_profile = np.array(prof_vec, dtype=np.float32)

                        raw_protos = data.get("prototypes", [])
                        self.enrolled_prototypes = [
                            np.array(p, dtype=np.float32)
                            for p in raw_protos
                            if len(p) == self.FEATURE_DIM
                        ]
                        print(
                            f"[SpeakerID] Loaded target speaker profile for "
                            f"'{data.get('user_name', 'User')}' "
                            f"({len(self.enrolled_prototypes)} prototypes, model={model_id})."
                        )
                        return True
                    else:
                        print(
                            f"[SpeakerID] Incompatible profile detected "
                            f"(dim={len(prof_vec)}, model={model_id!r}). Re-enrollment required."
                        )
            except Exception as e:
                print(f"[SpeakerID] Error loading profile: {e}")
        self.enrolled_profile = None
        self.enrolled_prototypes = []
        return False

    def identify_speaker(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict[str, Any]:
        """
        Compares incoming speech audio against enrolled target user embedding using Cosine Similarity.
        Maintains an internal streaming audio ring buffer and EMA similarity smoothing.
        Returns speaker tag: 'User [Me]' vs 'External Speaker'.
        """
        if self.enrolled_profile is None:
            return {
                "speaker_id": "User [Me]",
                "is_user": True,
                "similarity_score": 1.0,
                "confidence": 1.0,
                "enrolled": False
            }

        samples = audio_data.astype(np.float32)
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / 32768.0

        # Append incoming chunk to ring buffer
        self.audio_ring_buffer.extend(samples.tolist())
        accumulated_audio = np.array(self.audio_ring_buffer, dtype=np.float32)

        sample_embedding = self.extract_features(accumulated_audio, sample_rate)

        # Global profile cosine similarity
        sim_global = float(np.dot(self.enrolled_profile, sample_embedding))

        # Prototype ensemble cosine similarity
        if self.enrolled_prototypes:
            proto_sims = [float(np.dot(p, sample_embedding)) for p in self.enrolled_prototypes]
            proto_sims.sort(reverse=True)
            top_k = min(3, len(proto_sims))
            sim_proto = float(np.mean(proto_sims[:top_k]))
            similarity = 0.5 * sim_global + 0.5 * sim_proto
        else:
            similarity = sim_global

        # Exponential Moving Average (EMA) smoothing
        if self.ema_similarity is None:
            self.ema_similarity = similarity
        else:
            self.ema_similarity = self.ema_alpha * similarity + (1 - self.ema_alpha) * self.ema_similarity

        smoothed_sim = float(self.ema_similarity)

        is_user = smoothed_sim >= self.similarity_threshold

        confidence = max(0.0, min(1.0, (smoothed_sim + 1.0) / 2.0))
        speaker_tag = "User [Me]" if is_user else "External Speaker"

        return {
            "speaker_id": speaker_tag,
            "is_user": is_user,
            "similarity_score": round(smoothed_sim, 4),
            "raw_similarity": round(similarity, 4),
            "confidence": round(confidence, 3),
            "enrolled": True
        }

    def clear_stream_history(self):
        """Resets stream ring buffer and similarity EMA when starting/stopping stream."""
        self.audio_ring_buffer.clear()
        self.ema_similarity = None
