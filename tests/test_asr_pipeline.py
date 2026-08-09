"""Tests for utterance buffering, ASR preference, and transcript cleanup."""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lyra.server import asr as asr_mod
from lyra.server.asr import WhisperASR, set_transcribe_override
from lyra.server.transcript_cleanup import TranscriptCleaner, heuristic_cleanup
from lyra.server.utterance_buffer import UtteranceBuffer


@pytest.fixture(autouse=True)
def _reset_asr_override():
    set_transcribe_override(None)
    yield
    set_transcribe_override(None)


def test_utterance_buffer_emits_after_silence_hangover():
    buf = UtteranceBuffer(sample_rate=16000, silence_hangover_sec=0.2, min_utterance_sec=0.1)
    speech = np.ones(1600, dtype=np.float32) * 0.1  # 0.1s
    silence = np.zeros(3200, dtype=np.float32)  # 0.2s

    assert buf.push(speech, True) is None
    assert buf.push(speech, True) is None
    done = buf.push(silence, False)
    assert done is not None
    assert len(done) >= 1600


def test_utterance_buffer_ignores_short_blips():
    buf = UtteranceBuffer(sample_rate=16000, silence_hangover_sec=0.1, min_utterance_sec=0.5)
    speech = np.ones(800, dtype=np.float32) * 0.1  # 0.05s
    silence = np.zeros(2000, dtype=np.float32)
    assert buf.push(speech, True) is None
    assert buf.push(silence, False) is None  # too short to finalize as utterance


def test_heuristic_cleanup_strips_fillers():
    out = heuristic_cleanup("um hello uh there")
    assert "um" not in out.lower()
    assert "uh" not in out.lower()
    assert "hello" in out.lower()
    assert out.endswith(".")


def test_transcript_cleaner_falls_back_without_ollama():
    cleaner = TranscriptCleaner(ollama_client=None, enabled=True)
    assert "world" in cleaner.clean("um hello world").lower()


def test_whisper_asr_uses_override():
    set_transcribe_override(lambda audio, sr: "hello from mock")
    engine = WhisperASR(enabled=True)
    text = engine.transcribe(np.ones(1600, dtype=np.float32) * 0.05, 16000)
    assert text == "hello from mock"
    assert engine.status()["available"] is True


def test_server_asr_prefers_whisper_over_client_text(tmp_path, monkeypatch):
    import lyra.server.app as app_module
    from lyra.server.speaker_id import TargetSpeakerExtractor
    from lyra.server.qdrant_memory import FakeHybridEmbedder, QdrantEpisodicStore
    from lyra.server.rolling_memory import RollingMemoryEngine

    set_transcribe_override(lambda audio, sr: "server side transcript")

    store = QdrantEpisodicStore(
        location=":memory:",
        collection="asr_pref",
        embedder=FakeHybridEmbedder(dense_dim=32),
        vector_size=32,
    )
    engine = RollingMemoryEngine(episodic_store=store)
    extractor = TargetSpeakerExtractor(profile_path=str(tmp_path / "p.json"), similarity_threshold=0.28)

    monkeypatch.setattr(app_module, "memory_engine", engine)
    monkeypatch.setattr(app_module, "speaker_extractor", extractor)
    monkeypatch.setattr(app_module, "whisper_asr", WhisperASR(enabled=True))
    monkeypatch.setattr(
        app_module,
        "transcript_cleaner",
        TranscriptCleaner(ollama_client=None, enabled=True),
    )

    # Force VAD to treat chunks as speech then silence.
    monkeypatch.setattr(
        app_module.vad_detector,
        "is_speech",
        lambda audio: {
            "is_speech": float(np.mean(np.abs(audio))) > 0.01,
            "rms": float(np.mean(np.abs(audio))),
            "zcr": 0.1,
            "confidence": 0.9,
        },
    )

    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/ambient") as ws:
        # Client tries to send Web Speech text — should be ignored while ASR enabled.
        speech = (np.ones(8000, dtype=np.float32) * 0.05).tolist()
        ws.send_json({
            "type": "audio_chunk",
            "audio": speech,
            "transcript": "browser text should be ignored",
            "sample_rate": 16000,
            "is_final": False,
        })
        msg1 = ws.receive_json()
        assert msg1["asr_enabled"] is True

        # Hangover silence to finalize utterance (~0.6s default).
        silence = np.zeros(10000, dtype=np.float32).tolist()
        ws.send_json({
            "type": "audio_chunk",
            "audio": silence,
            "transcript": "",
            "sample_rate": 16000,
        })
        # May need a couple silence frames depending on hangover.
        entry = None
        for _ in range(4):
            msg = ws.receive_json()
            if msg.get("transcript_entry"):
                entry = msg["transcript_entry"]
                break
            ws.send_json({
                "type": "audio_chunk",
                "audio": silence,
                "transcript": "",
                "sample_rate": 16000,
            })

        assert entry is not None
        assert "server side" in entry["text"].lower()
        assert "browser text" not in entry["text"].lower()


def test_decode_audio_base64_roundtrip():
    import lyra.server.app as app_module

    pcm = np.linspace(-0.2, 0.2, 512, dtype=np.float32)
    b64 = base64.b64encode(pcm.tobytes()).decode("ascii")
    decoded = app_module._decode_ambient_audio({"audio_base64": b64})
    assert decoded.shape == pcm.shape
    assert np.allclose(decoded, pcm, atol=1e-6)
