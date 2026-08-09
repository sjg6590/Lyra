"""Tests for short-term dialogue sessions and episodic Q&A write-back."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from lyra.server.agent import LyraAgentEngine
from lyra.server.dialogue_memory import DialogueSessionStore
from lyra.server.ollama_client import OllamaClient
from lyra.server.qdrant_memory import FakeHybridEmbedder, QdrantEpisodicStore
from lyra.server.rolling_memory import RollingMemoryEngine
from lyra.server.search import WebSearchEngine


def _memory_engine() -> RollingMemoryEngine:
    store = QdrantEpisodicStore(
        location=":memory:",
        collection="test_dialogue_episodic",
        embedder=FakeHybridEmbedder(dense_dim=32),
        vector_size=32,
    )
    return RollingMemoryEngine(max_buffer_minutes=30, episodic_store=store)


def test_dialogue_store_caps_history_turns():
    store = DialogueSessionStore(max_history_turns=2)
    sid = store.get_or_create(None)
    store.append_turn(sid, "q1", "a1")
    store.append_turn(sid, "q2", "a2")
    store.append_turn(sid, "q3", "a3")
    messages = store.get_messages(sid)
    assert len(messages) == 4
    assert messages[0] == {"role": "user", "content": "q2"}
    assert messages[-1] == {"role": "assistant", "content": "a3"}


def test_dialogue_store_get_or_create_reuses_id():
    store = DialogueSessionStore()
    sid = store.get_or_create(None)
    assert store.get_or_create(sid) == sid
    assert store.get_messages(sid) == []


def test_second_tap_includes_prior_assistant_in_ollama_messages():
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.model = "qwen3.5:4b-mlx"
    mock_ollama.chat.side_effect = [
        "Blue Bottle is a coffee shop.",
        "I mentioned Blue Bottle in my previous reply.",
    ]

    store = DialogueSessionStore(max_history_turns=6)
    agent = LyraAgentEngine(
        name="Lyra",
        search_engine=WebSearchEngine(max_results=1),
        ollama_client=mock_ollama,
        web_search_enabled=False,
        dialogue_store=store,
    )
    memory = _memory_engine()

    first = agent.process_tap_to_talk(
        "What coffee shop did we talk about?",
        memory,
        session_id=None,
        dialogue_store=store,
    )
    sid = first["session_id"]
    assert sid
    assert first["response"] == "Blue Bottle is a coffee shop."

    second = agent.process_tap_to_talk(
        "Tell me more about that",
        memory,
        session_id=sid,
        dialogue_store=store,
    )
    assert second["session_id"] == sid
    assert mock_ollama.chat.call_count == 2

    second_messages = mock_ollama.chat.call_args_list[1][0][0]
    assert second_messages[0]["role"] == "system"
    assert "Prior user/assistant chat turns" in second_messages[0]["content"]
    assert second_messages[1] == {
        "role": "user",
        "content": "What coffee shop did we talk about?",
    }
    assert second_messages[2] == {
        "role": "assistant",
        "content": "Blue Bottle is a coffee shop.",
    }
    assert second_messages[3] == {"role": "user", "content": "Tell me more about that"}


def test_missing_session_starts_empty_history():
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.model = "qwen3.5:4b-mlx"
    mock_ollama.chat.return_value = "Hello."

    store = DialogueSessionStore()
    agent = LyraAgentEngine(
        name="Lyra",
        search_engine=WebSearchEngine(max_results=1),
        ollama_client=mock_ollama,
        web_search_enabled=False,
        dialogue_store=store,
    )
    result = agent.process_tap_to_talk("Hi", _memory_engine(), dialogue_store=store)
    messages = mock_ollama.chat.call_args[0][0]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Hi"}
    assert result["session_id"].startswith("sess_")


def test_dialogue_turn_written_to_episodic_not_rolling():
    memory = _memory_engine()
    written = memory.record_dialogue_turn(
        user_text="What did Lyra say about Orion?",
        assistant_text="Orion is the project codename we discussed.",
        assistant_name="Lyra",
    )
    assert len(written) == 2
    assert len(memory.rolling_buffer) == 0

    hits = memory.search_memory("Orion project codename", top_k=3)
    texts = " ".join(h.get("text", "") for h in hits)
    speakers = {h.get("speaker") for h in hits}
    assert "Orion" in texts
    assert "User" in speakers or "Lyra" in speakers


def test_process_tap_persists_dialogue_to_episodic():
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.model = "qwen3.5:4b-mlx"
    mock_ollama.chat.return_value = "Nebula means the launch window next month."

    store = DialogueSessionStore()
    agent = LyraAgentEngine(
        name="Lyra",
        search_engine=WebSearchEngine(max_results=1),
        ollama_client=mock_ollama,
        web_search_enabled=False,
        dialogue_store=store,
    )
    memory = _memory_engine()
    result = agent.process_tap_to_talk(
        "What does Nebula mean?",
        memory,
        dialogue_store=store,
    )
    assert result["session_id"]
    assert len(memory.rolling_buffer) == 0
    hits = memory.search_memory("Nebula launch window", top_k=3)
    assert any("Nebula" in (h.get("text") or "") for h in hits)


def test_tap_to_talk_api_returns_session_id():
    from lyra.server import app as app_module

    fake = {
        "query": "Hi",
        "response": "Hello",
        "thoughts": [],
        "ambient_context_used": "",
        "memory_results": [],
        "search_results": [],
        "tts": {"text": "Hello", "rate": 1.05},
        "latency_ms": 5,
        "latency": {"rag_ms": 1, "search_ms": 0, "llm_ms": 3, "ttft_ms": 3, "total_ms": 5},
        "llm_backend": "ollama",
        "session_id": "sess_test123",
    }

    with patch.object(app_module.agent_engine, "process_tap_to_talk", return_value=fake):
        client = TestClient(app_module.app)
        resp = client.post("/api/tap_to_talk", json={"query": "Hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess_test123"
        app_module.agent_engine.process_tap_to_talk.assert_called_once()
        kwargs = app_module.agent_engine.process_tap_to_talk.call_args.kwargs
        assert "session_id" in kwargs
