import json
import os
from collections import deque
from typing import Any

import numpy as np
from scipy import signal
from scipy.fftpack import dct


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def create_mel_filterbank(num_filters: int = 26, nfft: int = 512, sample_rate: int = 16000, low_freq: float = 100, high_freq: float = 7500) -> np.ndarray:
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(min(high_freq, sample_rate // 2))
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_filters, nfft // 2 + 1), dtype=np.float32)
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return filters

class TargetSpeakerExtractor:
    """
    Target Speaker Extraction & Voice Biometric Diarization Module.
    Differentiates User ('Me') from Third-Party External Speakers ('Not Me')
    using multi-prototype acoustic feature embeddings (Liftered MFCCs, Spectral Contrast,
    Voiced-Frame Pitch Distribution, and Spectral Formant Shape).
    """

    FEATURE_DIM = 32

    def __init__(self, profile_path: str = "user_voice_profile.json", similarity_threshold: float = 0.65):
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

        # Mel filterbank cache
        self._mel_fb = create_mel_filterbank(num_filters=26, nfft=512, sample_rate=16000)

        self.load_enrolled_profile()

    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extracts a 32-dimensional normalized acoustic feature embedding vector.
        Features include:
        - 12 Liftered MFCC Means (C1 - C12: vocal tract geometry & formant envelopes)
        - 12 Liftered MFCC Standard Deviations across voiced speech frames
        - 4 Spectral Contrast band measures (peak vs valley energy ratios)
        - 4 Voiced-Frame Pitch (F0) statistics (mean, std, 10th percentile, 90th percentile)
        """
        samples = audio_data.astype(np.float32)
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / 32768.0

        if len(samples) < 512:
            samples = np.pad(samples, (0, 512 - len(samples)))

        nfft = 512
        frame_len = int(sample_rate * 0.025)
        frame_step = int(sample_rate * 0.010)

        if len(samples) < frame_len:
            samples = np.pad(samples, (0, frame_len - len(samples)))

        num_frames = 1 + int((len(samples) - frame_len) / frame_step)
        indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        frames = samples[indices]

        # Pre-emphasis filter
        frames = np.append(frames[:, 0:1], frames[:, 1:] - 0.97 * frames[:, :-1], axis=1)

        # Hamming window
        window = np.hamming(frame_len)
        frames = frames * window

        # Magnitude and Power Spectra
        mag_spectrum = np.abs(np.fft.rfft(frames, nfft))
        pow_spectrum = (1.0 / nfft) * (mag_spectrum ** 2)

        # Mel Filterbank Energies
        if sample_rate == 16000:
            fb = self._mel_fb
        else:
            fb = create_mel_filterbank(num_filters=26, nfft=nfft, sample_rate=sample_rate)

        filter_energies = np.dot(pow_spectrum, fb.T)
        filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
        log_filter_energies = np.log(filter_energies)

        # DCT to obtain MFCCs (C1 to C12)
        num_cep = 13
        mfccs = dct(log_filter_energies, type=2, axis=1, norm='ortho')[:, 1:num_cep]

        # Liftering
        n = np.arange(num_cep - 1)
        lift = 1.0 + (22.0 / 2.0) * np.sin(np.pi * n / 22.0)
        mfccs = mfccs * lift

        # Energy-based voiced speech frame detection
        frame_energies = np.sum(pow_spectrum, axis=1)
        energy_threshold = np.percentile(frame_energies, 25)
        voiced_mask = frame_energies >= energy_threshold

        if not np.any(voiced_mask):
            voiced_mask = np.ones(num_frames, dtype=bool)

        mfccs_voiced = mfccs[voiced_mask]
        mfcc_mean = np.mean(mfccs_voiced, axis=0)
        mfcc_std = np.std(mfccs_voiced, axis=0)

        # Spectral Contrast in 4 frequency bands
        freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
        bands = [(200, 700), (700, 1800), (1800, 3500), (3500, min(7500, sample_rate // 2))]
        contrasts = []
        for f_min, f_max in bands:
            mask = (freqs >= f_min) & (freqs < f_max)
            if np.any(mask):
                band_pow = pow_spectrum[:, mask]
                peaks = np.percentile(band_pow, 85, axis=1)
                valleys = np.percentile(band_pow, 15, axis=1)
                contrast = float(np.mean(np.log(peaks + 1e-6) - np.log(valleys + 1e-6)))
                contrasts.append(contrast)
            else:
                contrasts.append(0.0)

        # Pitch (F0) Estimation over voiced frames
        f0_list = []
        min_lag = int(sample_rate / 400)
        max_lag = int(sample_rate / 70)

        for i in np.where(voiced_mask)[0]:
            frame_data = samples[i * frame_step : i * frame_step + frame_len]
            if len(frame_data) < frame_len:
                continue
            corr = signal.correlate(frame_data, frame_data, mode='full')
            corr = corr[len(frame_data)-1:]
            if max_lag < len(corr):
                peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                r_val = corr[peak_lag] / (corr[0] + 1e-9)
                if r_val > 0.20:
                    f0 = sample_rate / peak_lag
                    f0_list.append(f0)

        if f0_list:
            f0_mean = float(np.mean(f0_list))
            f0_std = float(np.std(f0_list))
            f0_p10 = float(np.percentile(f0_list, 10))
            f0_p90 = float(np.percentile(f0_list, 90))
        else:
            f0_mean, f0_std, f0_p10, f0_p90 = 150.0, 20.0, 130.0, 170.0

        # Combine features
        raw_vector = np.hstack([
            mfcc_mean,
            mfcc_std,
            np.array(contrasts, dtype=np.float32),
            np.array([
                f0_mean / 300.0,
                f0_std / 100.0,
                f0_p10 / 300.0,
                f0_p90 / 300.0
            ], dtype=np.float32)
        ]).astype(np.float32)

        # L2 Normalization
        norm = np.linalg.norm(raw_vector)
        if norm > 0:
            return raw_vector / norm
        return raw_vector

    def enroll_user(self, audio_data: np.ndarray, user_name: str = "User", sample_rate: int = 16000) -> bool:
        """
        Enrolls user voice profile by extracting global embedding vector + multi-prototype window vectors.
        """
        samples = audio_data.astype(np.float32)
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / 32768.0

        global_embedding = self.extract_features(samples, sample_rate)

        # Extract window prototypes across enrollment audio (1.0s window, 0.5s step)
        window_size = int(sample_rate * 1.0)
        window_step = int(sample_rate * 0.5)
        prototypes = []

        if len(samples) >= window_size:
            for start in range(0, len(samples) - window_size + 1, window_step):
                sub_audio = samples[start : start + window_size]
                proto_emb = self.extract_features(sub_audio, sample_rate)
                prototypes.append(proto_emb.tolist())
        else:
            prototypes.append(global_embedding.tolist())

        self.enrolled_profile = global_embedding
        self.enrolled_prototypes = [np.array(p, dtype=np.float32) for p in prototypes]
        self.enrolled_metadata = {
            "user_name": user_name,
            "sample_rate": sample_rate,
            "feature_dim": self.FEATURE_DIM,
            "profile_vector": global_embedding.tolist(),
            "prototypes": prototypes
        }

        with open(self.profile_path, "w") as f:
            json.dump(self.enrolled_metadata, f, indent=2)

        print(f"[SpeakerID] User '{user_name}' voice profile enrolled ({len(prototypes)} prototypes, {self.FEATURE_DIM}D) saved to {self.profile_path}.")
        return True

    def load_enrolled_profile(self) -> bool:
        """Loads stored voice profile from file if present and valid."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r") as f:
                    data = json.load(f)
                    prof_vec = data.get("profile_vector", [])
                    if len(prof_vec) == self.FEATURE_DIM:
                        self.enrolled_metadata = data
                        self.enrolled_profile = np.array(prof_vec, dtype=np.float32)
                        
                        raw_protos = data.get("prototypes", [])
                        self.enrolled_prototypes = [
                            np.array(p, dtype=np.float32) for p in raw_protos 
                            if len(p) == self.FEATURE_DIM
                        ]
                        print(f"[SpeakerID] Loaded target speaker profile for '{data.get('user_name', 'User')}' ({len(self.enrolled_prototypes)} prototypes).")
                        return True
                    else:
                        print(f"[SpeakerID] Legacy profile format detected ({len(prof_vec)}D != {self.FEATURE_DIM}D). Re-enrollment recommended.")
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

