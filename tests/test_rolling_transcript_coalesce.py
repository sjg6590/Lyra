"""Tests for ambient rolling-transcript coalesce / dedupe behavior."""

from lyra.server.qdrant_memory import FakeHybridEmbedder, QdrantEpisodicStore
from lyra.server.rolling_memory import RollingMemoryEngine


def _make_engine(collection: str = "test_coalesce") -> RollingMemoryEngine:
    store = QdrantEpisodicStore(
        location=":memory:",
        collection=collection,
        embedder=FakeHybridEmbedder(dense_dim=32),
        vector_size=32,
    )
    return RollingMemoryEngine(
        max_buffer_minutes=30,
        max_episodic_entries=100,
        episodic_store=store,
    )


def test_growing_interim_sequence_coalesces_to_one_entry():
    engine = _make_engine("coalesce_interim")
    speaker = "User [Me]"

    parts = [
        "so now",
        "so now you're",
        "so now you're hid",
        "so now you're hiding again",
        "so now you're hiding again from",
        "so now you're hiding again from me",
    ]
    last = None
    for text in parts:
        last = engine.add_transcript(speaker, text, is_user=True, is_final=False)

    final = engine.add_transcript(
        speaker, "so now you're hiding again from me", is_user=True, is_final=True
    )

    assert len(engine.rolling_buffer) == 1
    entry = engine.rolling_buffer[0]
    assert entry["text"] == "so now you're hiding again from me"
    assert entry["is_final"] is True
    assert last is not None and final is not None
    assert last["id"] == final["id"] == entry["id"]
    # Interim-only should not flood episodic; final writes once.
    assert engine.episodic_count() == 1


def test_exact_duplicate_finals_are_skipped():
    engine = _make_engine("coalesce_dup")
    speaker = "User [Me]"

    first = engine.add_transcript(speaker, "already", is_user=True, is_final=True)
    acks = [
        engine.add_transcript(speaker, "already", is_user=True, is_final=True)
        for _ in range(8)
    ]

    assert first is not None
    assert all(item is first for item in acks)
    assert len(engine.rolling_buffer) == 1
    assert engine.rolling_buffer[0]["text"] == "already"
    assert engine.episodic_count() == 1


def test_distinct_utterances_still_append():
    engine = _make_engine("coalesce_distinct")
    speaker = "User [Me]"

    a = engine.add_transcript(
        speaker, "I'm just checking if you have chocolate on your face", is_user=True
    )
    b = engine.add_transcript(
        speaker, "so now you're hiding again from me", is_user=True
    )
    c = engine.add_transcript(speaker, "already", is_user=True)

    assert a is not None and b is not None and c is not None
    assert a["id"] != b["id"] != c["id"]
    assert len(engine.rolling_buffer) == 3
    assert [e["text"] for e in engine.rolling_buffer] == [
        "I'm just checking if you have chocolate on your face",
        "so now you're hiding again from me",
        "already",
    ]


def test_sticky_client_related_finals_coalesce_without_is_final_flag():
    """Old clients default is_final=True but still re-send growing hypotheses."""
    engine = _make_engine("coalesce_sticky")
    speaker = "User [Me]"

    engine.add_transcript(speaker, "so now", is_user=True)  # default final
    engine.add_transcript(speaker, "so now you're", is_user=True)
    engine.add_transcript(speaker, "so now you're hiding again from me", is_user=True)
    engine.add_transcript(speaker, "so now you're hiding again from me", is_user=True)

    assert len(engine.rolling_buffer) == 1
    assert engine.rolling_buffer[0]["text"] == "so now you're hiding again from me"


def test_format_context_for_prompt_is_not_spammy():
    engine = _make_engine("coalesce_prompt")
    speaker = "User [Me]"

    for text in ["so now", "so now you're", "so now you're hiding again from me"]:
        engine.add_transcript(speaker, text, is_user=True, is_final=False)
    engine.add_transcript(
        speaker, "so now you're hiding again from me", is_user=True, is_final=True
    )
    for _ in range(10):
        engine.add_transcript(speaker, "already", is_user=True, is_final=True)
    engine.add_transcript(speaker, "all righty", is_user=True, is_final=True)

    formatted = engine.format_context_for_prompt(max_entries=15)
    lines = [line for line in formatted.splitlines() if line.strip()]
    # One coalesced utterance + one deduped "already" + one new phrase — not dozens of rows.
    assert len(lines) == 3
    assert "so now you're hiding again from me" in lines[0]
    assert "already" in lines[1]
    assert "all righty" in lines[2]
    assert formatted.count('"already"') == 1


def test_cross_speaker_related_open_asr_coalesces_to_user():
    """Onset External prefix + later User hypothesis must heal into one row."""
    engine = _make_engine("coalesce_speaker_flip")

    first = engine.add_transcript(
        "External Speaker",
        "hey lyra",
        is_user=False,
        is_final=False,
    )
    assert first is not None
    assert first["speaker"] == "External Speaker"

    healed = engine.add_transcript(
        "User [Me]",
        "hey lyra what time is it",
        is_user=True,
        is_final=False,
    )
    assert healed is not None
    assert healed["id"] == first["id"]
    assert len(engine.rolling_buffer) == 1
    assert engine.rolling_buffer[0]["text"] == "hey lyra what time is it"
    assert engine.rolling_buffer[0]["speaker"] == "User [Me]"
    assert engine.rolling_buffer[0]["is_user"] is True

    final = engine.add_transcript(
        "User [Me]",
        "hey lyra what time is it",
        is_user=True,
        is_final=True,
    )
    assert final is not None
    assert final["id"] == first["id"]
    assert len(engine.rolling_buffer) == 1
    assert engine.rolling_buffer[0]["is_final"] is True
    assert engine.episodic_count() == 1


def test_unrelated_speakers_still_append_separately():
    engine = _make_engine("coalesce_unrelated_speakers")
    a = engine.add_transcript(
        "External Speaker", "the weather looks great today", is_user=False, is_final=True
    )
    b = engine.add_transcript(
        "User [Me]", "remind me to call mom later", is_user=True, is_final=True
    )
    assert a is not None and b is not None
    assert a["id"] != b["id"]
    assert len(engine.rolling_buffer) == 2


def test_late_trailing_words_merge_into_open_utterance(monkeypatch):
    """Web Speech often delivers the last 1–2 words after a short pause."""
    engine = _make_engine("coalesce_late_final")
    speaker = "User [Me]"

    clock = {"t": 1_000.0}

    def fake_time() -> float:
        return clock["t"]

    monkeypatch.setattr("lyra.server.rolling_memory.time.time", fake_time)
    monkeypatch.setattr(
        "lyra.server.rolling_memory.time.strftime",
        lambda *_args, **_kwargs: "12:00:00",
    )

    first = engine.add_transcript(
        speaker, "what's the most recent thing i was talking", is_user=True, is_final=False
    )
    assert first is not None

    # Pause longer than the sticky-final coalesce gap, but within open-utterance grace.
    clock["t"] += 8.0
    late = engine.add_transcript(
        speaker,
        "what's the most recent thing i was talking about",
        is_user=True,
        is_final=True,
    )

    assert late is not None
    assert late["id"] == first["id"]
    assert len(engine.rolling_buffer) == 1
    assert engine.rolling_buffer[0]["text"] == (
        "what's the most recent thing i was talking about"
    )
    assert engine.rolling_buffer[0]["is_final"] is True
    assert engine.episodic_count() == 1
