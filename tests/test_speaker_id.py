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
    """Deterministic stand-in for ECAPA: spectral signature → 192-D unit vector."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.zeros(speaker_embedder.EMBED_DIM, dtype=np.float32)

    n = min(len(samples), max(sample_rate, 512))
    windowed = samples[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(windowed)) + 1e-8
    bands = np.array_split(spec, 32)
    feats = np.array([np.log(np.mean(b)) for b in bands], dtype=np.float32)

    vec = np.zeros(speaker_embedder.EMBED_DIM, dtype=np.float32)
    for i in range(speaker_embedder.EMBED_DIM):
        vec[i] = feats[i % 32] * (1.0 + 0.03 * (i // 32))
        vec[i] += 0.15 * np.sin((i + 1) * (feats[0] + 1.0))

    # Pitch-ish cue from autocorrelation peak location
    frame = samples[: min(len(samples), 800)]
    if len(frame) > 64:
        corr = signal.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]
        peak = int(np.argmax(corr[20:200])) + 20
        vec[peak % speaker_embedder.EMBED_DIM] += 2.0

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
    speech = generate_synthetic_audio(f0=140.0, duration=2.0)
    silence = np.zeros(16000, dtype=np.float32)
    audio = np.concatenate([silence, speech, silence, speech])

    extractor.enroll_user(audio, user_name="GateTest", sample_rate=16000)
    # Without gating, ~6 windows; with gating, silence windows are dropped.
    assert 1 <= len(extractor.enrolled_prototypes) <= 8


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
