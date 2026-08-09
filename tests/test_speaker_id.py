import base64
import json
import os

import numpy as np
import pytest
from scipy import signal

from lyra.server import speaker_embedder
from lyra.server.enrollment_prompt import (
    ENROLLMENT_SCRIPT,
    MIN_COVERAGE_RATIO,
    PROMPT_ID,
    coverage_ratio,
    get_enrollment_prompt,
)
from lyra.server.speaker_id import TargetSpeakerExtractor


def generate_synthetic_audio(f0: float = 130.0, duration: float = 2.0, sample_rate: int = 16000, formants: list = [500, 1500, 2500]):
    """Generates synthetic voiced speech-like signal with specific pitch and formant resonance peaks."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal_raw = np.zeros_like(t)
    for h in range(1, 15):
        freq = f0 * h
        if freq < sample_rate / 2:
            amp = 1.0 / h
            signal_raw += amp * np.sin(2 * np.pi * freq * t)

    audio = signal_raw
    for f_c in formants:
        bw = f_c / 10.0
        r = np.exp(-np.pi * bw / sample_rate)
        theta = 2 * np.pi * f_c / sample_rate
        b = [1.0 - r]
        a = [1.0, -2 * r * np.cos(theta), r * r]
        audio = signal.lfilter(b, a, audio)

    audio = audio / np.max(np.abs(audio))
    return audio.astype(np.float32)


def mock_ecapa_embed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Deterministic stand-in for ECAPA with strong pitch/formant separation."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    dim = speaker_embedder.EMBED_DIM
    if samples.size == 0:
        return np.zeros(dim, dtype=np.float32)

    # Autocorrelation F0 estimate drives a sparse one-hot-ish code.
    frame = samples[: min(len(samples), sample_rate)]
    min_lag = max(20, int(sample_rate / 400))
    max_lag = min(len(frame) - 1, int(sample_rate / 70))
    f0 = 150.0
    if max_lag > min_lag:
        corr = signal.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]
        peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        f0 = float(sample_rate) / float(peak_lag)

    # Spectral centroid as a second axis (formant-ish).
    n = min(len(samples), max(512, sample_rate // 2))
    windowed = samples[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(windowed)) + 1e-8
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    centroid = float(np.sum(freqs * spec) / np.sum(spec))

    vec = np.zeros(dim, dtype=np.float32)
    f0_bin = int(np.clip(round(f0 / 5.0), 0, 79))
    cent_bin = int(np.clip(round(centroid / 80.0), 0, 79))
    vec[f0_bin] = 5.0
    vec[80 + cent_bin] = 4.0
    # Tiny neighborhood so tiny pitch jitter still matches the same speaker.
    for delta, weight in ((-1, 0.35), (1, 0.35)):
        i = f0_bin + delta
        if 0 <= i < 80:
            vec[i] += weight
        j = 80 + cent_bin + delta
        if 80 <= j < 160:
            vec[j] += weight

    # Weak residual from band energies for within-speaker variation.
    bands = np.array_split(spec, 32)
    for i, band in enumerate(bands):
        vec[160 + i] = 0.02 * float(np.log(np.mean(band)))

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


@pytest.fixture(autouse=True)
def _mock_speaker_embedder():
    speaker_embedder.set_embed_fn_override(mock_ecapa_embed)
    speaker_embedder.reset_embedder()
    yield
    speaker_embedder.set_embed_fn_override(None)
    speaker_embedder.reset_embedder()


@pytest.fixture
def temp_profile_path(tmp_path):
    return str(tmp_path / "test_user_voice_profile.json")


def test_feature_extraction_dimension(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path)
    audio = generate_synthetic_audio(f0=140.0, duration=1.5)
    features = extractor.extract_features(audio, sample_rate=16000)

    assert isinstance(features, np.ndarray)
    assert features.shape == (speaker_embedder.EMBED_DIM,)
    norm = np.linalg.norm(features)
    assert pytest.approx(norm, abs=1e-5) == 1.0


def test_user_enrollment_and_loading(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path)

    assert extractor.enrolled_profile is None

    enrollment_audio = generate_synthetic_audio(f0=130.0, duration=4.0)
    success = extractor.enroll_user(
        enrollment_audio,
        user_name="TestUser",
        sample_rate=16000,
        prompt_id=PROMPT_ID,
        coverage_ratio=0.8,
    )

    assert success is True
    assert extractor.enrolled_profile is not None
    assert len(extractor.enrolled_prototypes) > 0
    assert len(extractor.enrolled_prototypes) <= TargetSpeakerExtractor.MAX_PROTOTYPES
    assert os.path.exists(temp_profile_path)
    assert extractor.enrolled_metadata.get("model_id") == speaker_embedder.MODEL_ID
    assert extractor.enrolled_metadata.get("feature_dim") == speaker_embedder.EMBED_DIM
    assert extractor.enrolled_metadata.get("prompt_id") == PROMPT_ID
    assert extractor.enrolled_metadata.get("coverage_ratio") == 0.8
    assert extractor.enrolled_metadata.get("enrollment_duration_sec") == pytest.approx(4.0)

    extractor2 = TargetSpeakerExtractor(profile_path=temp_profile_path)
    assert extractor2.enrolled_profile is not None
    assert len(extractor2.enrolled_prototypes) == len(extractor.enrolled_prototypes)
    assert extractor2.enrolled_metadata.get("user_name") == "TestUser"
    assert extractor2.enrolled_metadata.get("model_id") == speaker_embedder.MODEL_ID


def test_speaker_differentiation(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path, similarity_threshold=0.40)

    user_audio = generate_synthetic_audio(f0=120.0, duration=5.0, formants=[500, 1500, 2500])
    extractor.enroll_user(user_audio, user_name="Shaun")

    user_test_audio = generate_synthetic_audio(f0=122.0, duration=1.5, formants=[510, 1490, 2510])
    res_user = extractor.identify_speaker(user_test_audio)

    assert res_user["is_user"] is True
    assert res_user["speaker_id"] == "User [Me]"
    assert res_user["similarity_score"] >= 0.40

    extractor.clear_stream_history()

    ext1_audio = generate_synthetic_audio(f0=220.0, duration=1.5, formants=[850, 1950, 2850])
    res_ext1 = extractor.identify_speaker(ext1_audio)

    assert res_ext1["is_user"] is False
    assert res_ext1["speaker_id"] == "External Speaker"
    assert res_ext1["similarity_score"] < 0.40

    extractor.clear_stream_history()

    ext2_audio = generate_synthetic_audio(f0=170.0, duration=1.5, formants=[350, 2200, 3100])
    res_ext2 = extractor.identify_speaker(ext2_audio)

    assert res_ext2["is_user"] is False
    assert res_ext2["speaker_id"] == "External Speaker"
    assert res_ext2["similarity_score"] < 0.40


def test_legacy_profile_handling(temp_profile_path):
    legacy_data = {
        "user_name": "LegacyUser",
        "sample_rate": 16000,
        "profile_vector": [0.1] * 15,
    }
    with open(temp_profile_path, "w") as f:
        json.dump(legacy_data, f)

    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path)
    assert extractor.enrolled_profile is None
    assert extractor.enrolled_prototypes == []


def test_wrong_model_profile_rejected(temp_profile_path):
    legacy_ecapa_dim = {
        "user_name": "OldModel",
        "sample_rate": 16000,
        "feature_dim": speaker_embedder.EMBED_DIM,
        "model_id": "some-other-model",
        "profile_vector": [0.0] * speaker_embedder.EMBED_DIM,
        "prototypes": [],
    }
    with open(temp_profile_path, "w") as f:
        json.dump(legacy_ecapa_dim, f)

    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path)
    assert extractor.enrolled_profile is None


def test_speech_gated_prototypes_skip_silence(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path)
    speech = generate_synthetic_audio(f0=140.0, duration=3.0)
    silence = np.zeros(int(16000 * 3.0), dtype=np.float32)

    mixed = np.concatenate([silence, speech])
    extractor.enroll_user(mixed, user_name="GateTest", sample_rate=16000)
    mixed_count = len(extractor.enrolled_prototypes)

    extractor_all = TargetSpeakerExtractor(profile_path=str(temp_profile_path) + ".all.json")
    all_speech = np.concatenate([speech, speech])
    extractor_all.enroll_user(all_speech, user_name="AllSpeech", sample_rate=16000)
    all_count = len(extractor_all.enrolled_prototypes)

    # Leading silence should drop several windows vs an equal-length all-speech take.
    assert mixed_count >= 1
    assert mixed_count < all_count


def test_nonspeech_does_not_update_ring_or_ema(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path, similarity_threshold=0.40)
    user_audio = generate_synthetic_audio(f0=120.0, duration=3.0)
    extractor.enroll_user(user_audio, user_name="Shaun")

    speech = generate_synthetic_audio(f0=122.0, duration=0.6)
    res = extractor.identify_speaker(speech, is_speech=True)
    assert res["enrolled"] is True
    ring_len = len(extractor.audio_ring_buffer)
    ema = extractor.ema_similarity

    silence = np.zeros(2048, dtype=np.float32)
    sticky = extractor.identify_speaker(silence, is_speech=False)
    assert sticky["is_user"] is True
    assert sticky["warmed"] is False
    assert sticky["stable"] is False
    assert len(extractor.audio_ring_buffer) == ring_len
    assert extractor.ema_similarity == ema


def test_warmup_holds_user_before_external_flip(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path, similarity_threshold=0.40)
    user_audio = generate_synthetic_audio(f0=120.0, duration=3.0, formants=[500, 1500, 2500])
    extractor.enroll_user(user_audio, user_name="Shaun")
    extractor.clear_stream_history()

    # ~0.25s external chunks during warm-up must stay User.
    chunk = generate_synthetic_audio(f0=220.0, duration=0.25, formants=[850, 1950, 2850])
    early = extractor.identify_speaker(chunk, is_speech=True)
    assert early["warmed"] is False
    assert early["is_user"] is True
    assert early["stable"] is False

    # Continue past 0.5s warm-up with more external audio → External.
    more = generate_synthetic_audio(f0=220.0, duration=0.5, formants=[850, 1950, 2850])
    late = extractor.identify_speaker(more, is_speech=True)
    assert late["warmed"] is True
    assert late["stable"] is True
    assert late["is_user"] is False
    assert late["speaker_id"] == "External Speaker"


def test_hysteresis_keeps_user_through_brief_dip(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path, similarity_threshold=0.40)
    user_audio = generate_synthetic_audio(f0=120.0, duration=3.0)
    extractor.enroll_user(user_audio, user_name="Shaun")

    match = generate_synthetic_audio(f0=122.0, duration=0.6)
    first = extractor.identify_speaker(match, is_speech=True)
    assert first["is_user"] is True
    assert first["warmed"] is True
    assert extractor._identity_committed is True

    # Freeze EMA so we can probe hysteresis thresholds directly.
    extractor.ema_alpha = 0.0
    chunk = generate_synthetic_audio(f0=122.0, duration=0.2)

    extractor.ema_similarity = 0.35
    mid = extractor.identify_speaker(chunk, is_speech=True)
    assert mid["is_user"] is True

    extractor.ema_similarity = 0.25
    low = extractor.identify_speaker(chunk, is_speech=True)
    assert low["is_user"] is False


def test_enrollment_prompt_payload_and_coverage():
    payload = get_enrollment_prompt()
    assert payload["prompt_id"] == PROMPT_ID
    assert payload["target_duration_sec"] == 60
    assert payload["min_coverage_ratio"] == MIN_COVERAGE_RATIO
    assert ENROLLMENT_SCRIPT in payload["script"] or payload["script"] == ENROLLMENT_SCRIPT
    assert len(payload["expected_words"]) > 50

    full = coverage_ratio(ENROLLMENT_SCRIPT)
    assert full == pytest.approx(1.0)
    partial = coverage_ratio("hello lyra coffee music hiking")
    assert 0.0 < partial < 1.0
    assert coverage_ratio("") == 0.0


def test_enroll_prompt_and_voice_validation_api(temp_profile_path, monkeypatch):
    from fastapi.testclient import TestClient

    import lyra.server.app as app_module

    # Keep the app's extractor pointed at a temp profile and mock embedder already active.
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path, similarity_threshold=0.40)
    monkeypatch.setattr(app_module, "speaker_extractor", extractor)

    client = TestClient(app_module.app)

    prompt_resp = client.get("/api/enroll_prompt")
    assert prompt_resp.status_code == 200
    prompt = prompt_resp.json()
    assert prompt["prompt_id"] == PROMPT_ID

    short = generate_synthetic_audio(duration=2.0)
    short_b64 = base64.b64encode(short.tobytes()).decode("ascii")
    short_resp = client.post(
        "/api/enroll_voice",
        json={"user_name": "A", "audio_base64": short_b64, "prompt_id": PROMPT_ID},
    )
    assert short_resp.status_code == 400
    assert "too short" in short_resp.json()["detail"].lower()

    long_audio = generate_synthetic_audio(duration=46.0)
    long_b64 = base64.b64encode(long_audio.tobytes()).decode("ascii")
    low_cov = client.post(
        "/api/enroll_voice",
        json={
            "user_name": "A",
            "audio_base64": long_b64,
            "prompt_id": PROMPT_ID,
            "heard_transcript": "hello only",
        },
    )
    assert low_cov.status_code == 400
    assert "coverage" in low_cov.json()["detail"].lower()

    ok = client.post(
        "/api/enroll_voice",
        json={
            "user_name": "A",
            "audio_base64": long_b64,
            "prompt_id": PROMPT_ID,
            "heard_transcript": ENROLLMENT_SCRIPT,
        },
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["success"] is True
    assert body["model_id"] == speaker_embedder.MODEL_ID
    assert body["prototype_count"] >= 1
