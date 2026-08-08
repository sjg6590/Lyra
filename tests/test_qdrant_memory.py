"""Unit tests for Qdrant episodic memory (in-memory client, fake embedder)."""

from lyra.server.qdrant_memory import (
    FakeHybridEmbedder,
    QdrantEpisodicStore,
    document_prefix,
    query_prefix,
)
from lyra.server.rolling_memory import RollingMemoryEngine


def _make_store(collection: str = "test_episodic") -> QdrantEpisodicStore:
    return QdrantEpisodicStore(
        location=":memory:",
        collection=collection,
        embedder=FakeHybridEmbedder(dense_dim=32),
        vector_size=32,
    )


def test_embeddinggemma_prefixes():
    assert document_prefix("hello world") == "title: none | text: hello world"
    assert document_prefix("hello", title="notes") == "title: notes | text: hello"
    assert query_prefix("what time") == "task: search result | query: what time"


def test_upsert_search_and_clear():
    store = _make_store("test_upsert_search")
    store.upsert_entry(
        {
            "id": "utt_1",
            "timestamp": 1.0,
            "readable_time": "10:00:00",
            "speaker": "User",
            "is_user": True,
            "text": "the project deadline is Friday",
            "confidence": 0.9,
        }
    )
    store.upsert_entry(
        {
            "id": "utt_2",
            "timestamp": 2.0,
            "readable_time": "10:01:00",
            "speaker": "External",
            "is_user": False,
            "text": "I want pizza for dinner tonight",
            "confidence": 0.8,
        }
    )

    assert store.count() == 2
    results = store.search_hybrid("What is the project deadline?", top_k=2)
    assert results
    assert "deadline" in results[0]["text"].lower() or "project" in results[0]["text"].lower()
    assert "relevance_score" in results[0]

    store.clear()
    assert store.count() == 0
    assert store.search_hybrid("project deadline") == []


def test_scroll_all_sorted_by_timestamp():
    store = _make_store("test_scroll")
    store.upsert_entry(
        {
            "id": "utt_b",
            "timestamp": 20.0,
            "readable_time": "10:00:20",
            "speaker": "User",
            "is_user": True,
            "text": "second utterance",
            "confidence": 1.0,
        }
    )
    store.upsert_entry(
        {
            "id": "utt_a",
            "timestamp": 10.0,
            "readable_time": "10:00:10",
            "speaker": "User",
            "is_user": True,
            "text": "first utterance",
            "confidence": 1.0,
        }
    )
    entries = store.scroll_all()
    assert [e["id"] for e in entries] == ["utt_a", "utt_b"]


def test_prune_to_max():
    store = _make_store("test_prune")
    for i in range(5):
        store.upsert_entry(
            {
                "id": f"utt_{i}",
                "timestamp": float(i),
                "readable_time": f"10:00:0{i}",
                "speaker": "User",
                "is_user": True,
                "text": f"memory entry number {i}",
                "confidence": 1.0,
            }
        )
    store.prune_to_max(3)
    assert store.count() == 3
    ids = {e["id"] for e in store.scroll_all()}
    assert "utt_0" not in ids
    assert "utt_1" not in ids
    assert ids == {"utt_2", "utt_3", "utt_4"}


def test_rolling_memory_engine_with_qdrant():
    store = _make_store("test_rolling")
    engine = RollingMemoryEngine(
        max_buffer_minutes=30,
        max_episodic_entries=100,
        episodic_store=store,
    )
    engine.add_transcript("User", "meeting with Alice about budget", is_user=True)
    engine.add_transcript("External", "weather looks rainy tomorrow", is_user=False)

    assert len(engine.rolling_buffer) == 2
    assert engine.episodic_count() == 2
    assert len(engine.episodic_memory) == 2

    hits = engine.search_memory("Alice budget meeting", top_k=2)
    assert hits
    assert "alice" in hits[0]["text"].lower() or "budget" in hits[0]["text"].lower()

    status = engine.backend_status()
    assert status["episodic_backend"] == "qdrant"
    assert status["qdrant"]["ok"] is True

    engine.clear_memory()
    assert len(engine.rolling_buffer) == 0
    assert engine.episodic_count() == 0
