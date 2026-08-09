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
    MAX_PROTOTYPES = 12
    PROTOTYPE_STRATEGY = "farthest_point_v1"
    WINDOW_SEC = 1.0
    WINDOW_STEP_SEC = 0.5
    SPEECH_RMS_THRESHOLD = 0.01
    # Require this much speech in the ring before allowing External flips.
    WARMUP_SPEECH_SEC = 0.5
    # Clear ring/EMA after this much continuous non-speech.
    SILENCE_RESET_SEC = 0.75
    # Leave-User hysteresis floor (enter-User uses similarity_threshold).
    EXIT_USER_THRESHOLD = 0.18
    # Inference ring length (seconds) for more stable ECAPA embeddings.
    RING_BUFFER_SEC = 2.0
    GLOBAL_SCORE_WEIGHT = 0.65
    PROTO_SCORE_WEIGHT = 0.35

    def __init__(self, profile_path: str = "user_voice_profile.json", similarity_threshold: float = 0.28):
        self.profile_path = profile_path
        self.similarity_threshold = similarity_threshold
        self.exit_user_threshold = min(self.EXIT_USER_THRESHOLD, similarity_threshold)
        self.enrolled_profile: np.ndarray | None = None
        self.enrolled_prototypes: list[np.ndarray] = []
        self.enrolled_metadata: dict[str, Any] = {}

        # Continuous streaming audio ring buffer (2.0s = 32000 samples at 16kHz)
        self.audio_ring_buffer: deque = deque(maxlen=int(16000 * self.RING_BUFFER_SEC))

        # Exponential Moving Average (EMA) of similarity scores for streaming stability
        self.ema_similarity: float | None = None
        self.ema_alpha: float = 0.3

        # Sticky identity + warm-up / silence tracking
        self.last_is_user: bool = True
        self.speech_samples_in_utterance: int = 0
        self.nonspeech_samples: int = 0
        self._identity_committed: bool = False

        self.load_enrolled_profile()

    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract an L2-normalized ECAPA speaker embedding."""
        return speaker_embedder.embed_audio(audio_data, sample_rate=sample_rate)

    def _window_is_speech(self, samples: np.ndarray) -> bool:
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
        return rms >= self.SPEECH_RMS_THRESHOLD

    def _select_diverse_prototypes(self, embeddings: list[np.ndarray], centroid: np.ndarray) -> list[np.ndarray]:
        """
        Farthest-point sampling: keep the window closest to the centroid, then
        iteratively add the embedding farthest from the already-selected set.
        """
        if not embeddings:
            return []
        if len(embeddings) <= self.MAX_PROTOTYPES:
            return [emb.astype(np.float32) for emb in embeddings]

        stacked = np.stack([emb.astype(np.float32) for emb in embeddings], axis=0)
        # Cosine distance via 1 - dot (vectors are L2-normalized).
        sims_to_centroid = stacked @ centroid.astype(np.float32)
        selected: list[int] = [int(np.argmax(sims_to_centroid))]

        # Min cosine similarity to any selected prototype (lower = farther).
        min_sims = stacked @ stacked[selected[0]]
        while len(selected) < self.MAX_PROTOTYPES:
            # Exclude already selected by setting their min_sims high.
            for idx in selected:
                min_sims[idx] = 2.0
            next_idx = int(np.argmin(min_sims))
            selected.append(next_idx)
            # Update running min similarity to the selected set.
            sims_to_new = stacked @ stacked[next_idx]
            min_sims = np.minimum(min_sims, sims_to_new)

        return [stacked[i] for i in selected]

    def enroll_user(
        self,
        audio_data: np.ndarray,
        user_name: str = "User",
        sample_rate: int = 16000,
        prompt_id: str | None = None,
        coverage_ratio: float | None = None,
    ) -> bool:
        """
        Enrolls user voice profile by extracting speech-gated ECAPA window embeddings,
        a global mean centroid, and a compact diverse prototype set.
        """
        samples = audio_data.astype(np.float32)
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / 32768.0

        window_size = int(sample_rate * self.WINDOW_SEC)
        window_step = int(sample_rate * self.WINDOW_STEP_SEC)
        duration_sec = float(len(samples) / sample_rate) if sample_rate else 0.0

        speech_embeddings: list[np.ndarray] = []

        if len(samples) >= window_size:
            for start in range(0, len(samples) - window_size + 1, window_step):
                sub_audio = samples[start : start + window_size]
                if self._window_is_speech(sub_audio):
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

        diverse = self._select_diverse_prototypes(speech_embeddings, global_embedding)
        prototypes = [emb.astype(np.float32).tolist() for emb in diverse]

        self.enrolled_profile = global_embedding
        self.enrolled_prototypes = [np.array(p, dtype=np.float32) for p in prototypes]
        self.enrolled_metadata = {
            "user_name": user_name,
            "sample_rate": sample_rate,
            "feature_dim": self.FEATURE_DIM,
            "model_id": self.MODEL_ID,
            "profile_vector": global_embedding.tolist(),
            "prototypes": prototypes,
            "prototype_strategy": self.PROTOTYPE_STRATEGY,
            "prompt_id": prompt_id,
            "enrollment_duration_sec": round(duration_sec, 3),
            "coverage_ratio": coverage_ratio,
            "enrolled_at": time.time(),
        }

        with open(self.profile_path, "w") as f:
            json.dump(self.enrolled_metadata, f, indent=2)

        print(
            f"[SpeakerID] User '{user_name}' voice profile enrolled "
            f"({len(prototypes)} prototypes, {self.FEATURE_DIM}D, model={self.MODEL_ID}, "
            f"strategy={self.PROTOTYPE_STRATEGY}) saved to {self.profile_path}."
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

    def _sticky_result(
        self,
        *,
        enrolled: bool,
        warmed: bool,
        stable: bool,
        raw_similarity: float | None = None,
    ) -> dict[str, Any]:
        is_user = bool(self.last_is_user)
        smoothed = float(self.ema_similarity) if self.ema_similarity is not None else (1.0 if is_user else 0.0)
        confidence = max(0.0, min(1.0, (smoothed + 1.0) / 2.0))
        return {
            "speaker_id": "User [Me]" if is_user else "External Speaker",
            "is_user": is_user,
            "similarity_score": round(smoothed, 4),
            "raw_similarity": round(raw_similarity if raw_similarity is not None else smoothed, 4),
            "confidence": round(confidence, 3),
            "enrolled": enrolled,
            "warmed": warmed,
            "stable": stable,
        }

    def note_nonspeech(self, sample_count: int = 0, sample_rate: int = 16000) -> dict[str, Any]:
        """
        Record a non-speech stretch without updating the ring/EMA.
        After SILENCE_RESET_SEC of continuous non-speech, clear onset state.
        """
        enrolled = self.enrolled_profile is not None
        if sample_count > 0:
            self.nonspeech_samples += int(sample_count)
        else:
            # Treat an explicit note as one silence quantum at 16kHz (~64ms).
            self.nonspeech_samples += max(1, int(sample_rate * 0.064))

        reset_samples = int(sample_rate * self.SILENCE_RESET_SEC)
        if self.nonspeech_samples >= reset_samples:
            self.audio_ring_buffer.clear()
            self.ema_similarity = None
            self.speech_samples_in_utterance = 0
            self.nonspeech_samples = 0
            # New utterance warm-up should not inherit a prior External sticky label.
            self.last_is_user = True
            self._identity_committed = False

        return self._sticky_result(enrolled=enrolled, warmed=False, stable=False)

    def identify_speaker(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        is_speech: bool = True,
    ) -> dict[str, Any]:
        """
        Compares incoming speech audio against enrolled target user embedding using Cosine Similarity.
        Maintains an internal streaming audio ring buffer and EMA similarity smoothing.
        Non-speech frames freeze the ring/EMA and return the sticky label.
        Returns speaker tag: 'User [Me]' vs 'External Speaker'.
        """
        if self.enrolled_profile is None:
            self.last_is_user = True
            return {
                "speaker_id": "User [Me]",
                "is_user": True,
                "similarity_score": 1.0,
                "raw_similarity": 1.0,
                "confidence": 1.0,
                "enrolled": False,
                "warmed": True,
                "stable": True,
            }

        samples = audio_data.astype(np.float32)
        if samples.size and np.max(np.abs(samples)) > 1.0:
            samples = samples / 32768.0

        if not is_speech:
            return self.note_nonspeech(sample_count=len(samples), sample_rate=sample_rate)

        # Speech frame: reset silence counter and accumulate warm-up speech.
        self.nonspeech_samples = 0
        self.speech_samples_in_utterance += len(samples)
        warmup_samples = int(sample_rate * self.WARMUP_SPEECH_SEC)
        warmed = self.speech_samples_in_utterance >= warmup_samples

        # Append incoming chunk to ring buffer
        self.audio_ring_buffer.extend(samples.tolist())
        accumulated_audio = np.array(self.audio_ring_buffer, dtype=np.float32)

        sample_embedding = self.extract_features(accumulated_audio, sample_rate)

        # Global profile cosine similarity
        sim_global = float(np.dot(self.enrolled_profile, sample_embedding))

        # Prototype similarity: max over diverse exemplars (style-tolerant).
        if self.enrolled_prototypes:
            proto_sims = [float(np.dot(p, sample_embedding)) for p in self.enrolled_prototypes]
            sim_proto = float(max(proto_sims))
            similarity = (
                self.GLOBAL_SCORE_WEIGHT * sim_global + self.PROTO_SCORE_WEIGHT * sim_proto
            )
        else:
            similarity = sim_global

        # Exponential Moving Average (EMA) smoothing
        if self.ema_similarity is None:
            self.ema_similarity = similarity
        else:
            self.ema_similarity = self.ema_alpha * similarity + (1 - self.ema_alpha) * self.ema_similarity

        smoothed_sim = float(self.ema_similarity)

        # Warm-up: hold sticky label (default User) and do not commit External yet.
        # First warmed decision uses the enter threshold; later frames use hysteresis.
        if not warmed:
            is_user = bool(self.last_is_user)
            stable = False
        elif not self._identity_committed:
            is_user = smoothed_sim >= self.similarity_threshold
            self._identity_committed = True
            self.last_is_user = bool(is_user)
            stable = True
        else:
            if self.last_is_user:
                is_user = smoothed_sim > self.exit_user_threshold
            else:
                is_user = smoothed_sim >= self.similarity_threshold
            self.last_is_user = bool(is_user)
            stable = True

        confidence = max(0.0, min(1.0, (smoothed_sim + 1.0) / 2.0))
        speaker_tag = "User [Me]" if is_user else "External Speaker"

        return {
            "speaker_id": speaker_tag,
            "is_user": is_user,
            "similarity_score": round(smoothed_sim, 4),
            "raw_similarity": round(similarity, 4),
            "confidence": round(confidence, 3),
            "enrolled": True,
            "warmed": warmed,
            "stable": stable,
        }

    def clear_stream_history(self):
        """Resets stream ring buffer and similarity EMA when starting/stopping stream."""
        self.audio_ring_buffer.clear()
        self.ema_similarity = None
        self.last_is_user = True
        self.speech_samples_in_utterance = 0
        self.nonspeech_samples = 0
        self._identity_committed = False
