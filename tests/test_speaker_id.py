import json
import os

import numpy as np
import pytest
from scipy import signal

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

@pytest.fixture
def temp_profile_path(tmp_path):
    return str(tmp_path / "test_user_voice_profile.json")

def test_feature_extraction_dimension(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path)
    audio = generate_synthetic_audio(f0=140.0, duration=1.5)
    features = extractor.extract_features(audio, sample_rate=16000)

    assert isinstance(features, np.ndarray)
    assert features.shape == (32,)
    # Verify L2 normalization
    norm = np.linalg.norm(features)
    assert pytest.approx(norm, 1e-5) == 1.0

def test_user_enrollment_and_loading(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path)

    # Initially not enrolled
    assert extractor.enrolled_profile is None

    enrollment_audio = generate_synthetic_audio(f0=130.0, duration=4.0)
    success = extractor.enroll_user(enrollment_audio, user_name="TestUser", sample_rate=16000)

    assert success is True
    assert extractor.enrolled_profile is not None
    assert len(extractor.enrolled_prototypes) > 0
    assert os.path.exists(temp_profile_path)

    # Test reloading profile
    extractor2 = TargetSpeakerExtractor(profile_path=temp_profile_path)
    assert extractor2.enrolled_profile is not None
    assert len(extractor2.enrolled_prototypes) == len(extractor.enrolled_prototypes)
    assert extractor2.enrolled_metadata.get("user_name") == "TestUser"

def test_speaker_differentiation(temp_profile_path):
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path, similarity_threshold=0.65)

    # User enrollment: Male voice (120Hz pitch, standard male formants)
    user_audio = generate_synthetic_audio(f0=120.0, duration=5.0, formants=[500, 1500, 2500])
    extractor.enroll_user(user_audio, user_name="Shaun")

    # 1. Test target user speech (same speaker)
    user_test_audio = generate_synthetic_audio(f0=122.0, duration=1.5, formants=[510, 1490, 2510])
    res_user = extractor.identify_speaker(user_test_audio)

    assert res_user["is_user"] is True
    assert res_user["speaker_id"] == "User [Me]"
    assert res_user["similarity_score"] > 0.65

    # Reset streaming EMA history for independent external speaker evaluation
    extractor.clear_stream_history()

    # 2. Test external female / high-pitch speaker (220Hz pitch, higher formants)
    ext1_audio = generate_synthetic_audio(f0=220.0, duration=1.5, formants=[850, 1950, 2850])
    res_ext1 = extractor.identify_speaker(ext1_audio)

    assert res_ext1["is_user"] is False
    assert res_ext1["speaker_id"] == "External Speaker"
    assert res_ext1["similarity_score"] < 0.65

    extractor.clear_stream_history()

    # 3. Test external mid-pitch male speaker (170Hz pitch, different formants)
    ext2_audio = generate_synthetic_audio(f0=170.0, duration=1.5, formants=[350, 2200, 3100])
    res_ext2 = extractor.identify_speaker(ext2_audio)

    assert res_ext2["is_user"] is False
    assert res_ext2["speaker_id"] == "External Speaker"
    assert res_ext2["similarity_score"] < 0.65

def test_legacy_profile_handling(temp_profile_path):
    # Save a legacy 15-dim profile to file
    legacy_data = {
        "user_name": "LegacyUser",
        "sample_rate": 16000,
        "profile_vector": [0.1] * 15
    }
    with open(temp_profile_path, "w") as f:
        json.dump(legacy_data, f)

    # Initializing extractor should handle legacy file gracefully
    extractor = TargetSpeakerExtractor(profile_path=temp_profile_path)
    assert extractor.enrolled_profile is None
    assert extractor.enrolled_prototypes == []
