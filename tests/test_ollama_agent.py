"""Unit tests for Ollama client + Lyra agent synthesis (mocked HTTP)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from lyra.server.agent import LyraAgentEngine, format_sse
from lyra.server.ollama_client import OllamaClient, strip_think_blocks
from lyra.server.qdrant_memory import FakeHybridEmbedder, QdrantEpisodicStore
from lyra.server.rolling_memory import RollingMemoryEngine
from lyra.server.search import WebSearchEngine


def _memory_engine() -> RollingMemoryEngine:
    """In-memory Qdrant + fake embedder so agent tests need no Docker/model download."""
    store = QdrantEpisodicStore(
        location=":memory:",
        collection="test_ollama_episodic",
        embedder=FakeHybridEmbedder(dense_dim=32),
        vector_size=32,
    )
    return RollingMemoryEngine(max_buffer_minutes=30, episodic_store=store)


def test_strip_think_blocks():
    raw = "<think>internal chain</think>\nHello there."
    assert strip_think_blocks(raw) == "Hello there."


def test_ollama_client_from_config_disabled():
    assert OllamaClient.from_config({"enabled": False}) is None
    assert OllamaClient.from_config(None) is None


def test_ollama_client_from_config_defaults():
    client = OllamaClient.from_config({"enabled": True, "model": "qwen3.5:9b-mlx"})
    assert client is not None
    assert client.model == "qwen3.5:9b-mlx"
    assert client.think is False
    assert client.num_ctx == 2048
    assert client.num_predict == 96
    assert client.temperature == 0.4
    assert client.keep_alive == -1


def test_ollama_chat_sends_latency_options():
    client = OllamaClient(host="http://127.0.0.1:11434", model="qwen3.5:9b-mlx", think=False)

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"message": {"content": "Concise spoken answer."}}

    with patch("lyra.server.ollama_client.requests.post", return_value=mock_resp) as post:
        text = client.chat([{"role": "user", "content": "Hi"}])

    assert text == "Concise spoken answer."
    args, kwargs = post.call_args
    assert args[0] == "http://127.0.0.1:11434/api/chat"
    payload = kwargs["json"]
    assert payload["model"] == "qwen3.5:9b-mlx"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["keep_alive"] == -1
    assert payload["options"]["num_ctx"] == 2048
    assert payload["options"]["num_predict"] == 96
    assert payload["options"]["temperature"] == 0.4


def test_ollama_chat_stream_yields_tokens():
    client = OllamaClient(model="qwen3.5:9b-mlx")

    lines = [
        json.dumps({"message": {"content": "Hello"}, "done": False}),
        json.dumps({"message": {"content": " world"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("lyra.server.ollama_client.requests.post", return_value=mock_resp) as post:
        chunks = list(client.chat_stream([{"role": "user", "content": "Hi"}]))

    assert chunks == ["Hello", " world"]
    payload = post.call_args.kwargs["json"]
    assert payload["stream"] is True
    assert payload["options"]["num_predict"] == 96


def test_ollama_chat_strips_think_and_rejects_empty():
    client = OllamaClient()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"message": {"content": "<think>x</think>\n"}}

    with patch("lyra.server.ollama_client.requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="empty"):
            client.chat([{"role": "user", "content": "Hi"}])


def test_has_model_match():
    client = OllamaClient(model="qwen3.5:9b-mlx")
    with patch.object(client, "list_models", return_value=["qwen3.5:9b-mlx"]):
        assert client.has_model() is True
    with patch.object(client, "list_models", return_value=["llama3.2:3b"]):
        assert client.has_model() is False


def test_agent_uses_ollama_when_available():
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.model = "qwen3.5:9b-mlx"
    mock_ollama.chat.return_value = "I heard them mention Blue Bottle earlier."

    search = WebSearchEngine(max_results=1)
    search.search = MagicMock(return_value=[])  # type: ignore[method-assign]

    agent = LyraAgentEngine(
        name="Lyra",
        search_engine=search,
        ollama_client=mock_ollama,
        web_search_enabled=False,
    )
    memory = _memory_engine()
    memory.add_transcript(speaker="User [Me]", text="Let's meet at Blue Bottle", confidence=0.9, is_user=True)

    result = agent.process_tap_to_talk("What restaurant did I mention?", memory, force_search=False)

    assert result["response"] == "I heard them mention Blue Bottle earlier."
    assert result["llm_backend"] == "ollama"
    assert "latency" in result
    assert result["latency"]["rag_ms"] is not None
    assert result["latency"]["search_ms"] == 0
    assert result["latency"]["llm_ms"] is not None
    assert result["latency"]["total_ms"] is not None
    search.search.assert_not_called()
    mock_ollama.chat.assert_called_once()
    messages = mock_ollama.chat.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert "Blue Bottle" in messages[0]["content"] or "Blue Bottle" in messages[1]["content"]
    assert "ambient transcript" in messages[0]["content"].lower() or "Ambient" in messages[0]["content"]


def test_agent_skips_web_search_when_disabled():
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.model = "qwen3.5:9b-mlx"
    mock_ollama.chat.return_value = "ok"

    search = WebSearchEngine(max_results=1)
    search.search = MagicMock(return_value=[{"title": "X", "snippet": "Y", "url": "https://x"}])  # type: ignore[method-assign]

    agent = LyraAgentEngine(
        name="Lyra",
        search_engine=search,
        ollama_client=mock_ollama,
        web_search_enabled=False,
    )
    result = agent.process_tap_to_talk("what is the latest news", _memory_engine(), force_search=False)
    search.search.assert_not_called()
    assert result["search_results"] == []
    assert any("disabled" in t.lower() for t in result["thoughts"])


def test_agent_force_search_overrides_disabled_flag():
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.model = "qwen3.5:9b-mlx"
    mock_ollama.chat.return_value = "ok"

    search = WebSearchEngine(max_results=1)
    search.search = MagicMock(return_value=[{"title": "X", "snippet": "Y", "url": "https://x"}])  # type: ignore[method-assign]

    agent = LyraAgentEngine(
        name="Lyra",
        search_engine=search,
        ollama_client=mock_ollama,
        web_search_enabled=False,
    )
    result = agent.process_tap_to_talk("hello", _memory_engine(), force_search=True)
    search.search.assert_called_once()
    assert result["latency"]["search_ms"] is not None
    assert result["latency"]["search_ms"] >= 0


def test_agent_stream_emits_token_then_done():
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.model = "qwen3.5:9b-mlx"
    mock_ollama.chat_stream.return_value = iter(["Fast ", "reply"])

    agent = LyraAgentEngine(
        name="Lyra",
        search_engine=WebSearchEngine(max_results=1),
        ollama_client=mock_ollama,
        web_search_enabled=False,
    )
    events = list(agent.process_tap_to_talk_stream("Hi", _memory_engine()))
    kinds = [e["event"] for e in events]
    assert "status" in kinds
    assert "token" in kinds
    assert kinds[-1] == "done"
    done = events[-1]["data"]
    assert done["response"] == "Fast reply"
    assert done["llm_backend"] == "ollama"
    assert done["latency"]["ttft_ms"] is not None
    assert done["latency"]["total_ms"] is not None


def test_agent_falls_back_when_ollama_fails():
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.model = "qwen3.5:9b-mlx"
    mock_ollama.chat.side_effect = requests.ConnectionError("refused")

    search = WebSearchEngine(max_results=1)
    search.search = MagicMock(return_value=[])  # type: ignore[method-assign]

    agent = LyraAgentEngine(
        name="Lyra",
        search_engine=search,
        ollama_client=mock_ollama,
        web_search_enabled=False,
    )
    memory = _memory_engine()
    memory.add_transcript(speaker="User [Me]", text="Acme Corp is hiring", confidence=0.95, is_user=True)

    result = agent.process_tap_to_talk("What company was mentioned?", memory, force_search=False)

    assert result["llm_backend"] == "heuristic"
    assert "Acme Corp" in result["response"] or "mentioned" in result["response"].lower()
    assert any("falling back" in t.lower() for t in result["thoughts"])


def test_agent_heuristic_without_ollama_client():
    agent = LyraAgentEngine(name="Lyra", ollama_client=None, search_engine=WebSearchEngine(max_results=1))
    memory = _memory_engine()
    result = agent.process_tap_to_talk("Hello Lyra", memory)
    assert result["llm_backend"] == "heuristic"
    assert isinstance(result["response"], str) and len(result["response"]) > 0


def test_build_system_prompt_includes_search():
    search = WebSearchEngine(max_results=2)
    agent = LyraAgentEngine(name="Lyra", search_engine=search, ollama_client=None)
    prompt = agent.build_system_prompt(
        recent_transcript='[12:00:00] User [Me]: "hello"',
        memory_results=[{"readable_time": "12:00:00", "speaker": "User [Me]", "text": "hello"}],
        search_results=[{"title": "Example", "snippet": "A site", "url": "https://example.com"}],
    )
    assert "Lyra" in prompt
    assert "hello" in prompt
    assert "Example" in prompt
    assert "https://example.com" in prompt


def test_format_sse():
    frame = format_sse({"event": "token", "text": "hi"})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[len("data: "):].strip())["text"] == "hi"


def test_tap_to_talk_stream_endpoint_sse():
    """Light FastAPI SSE test with mocked agent stream."""
    from lyra.server import app as app_module

    fake_events = [
        {"event": "status", "stage": "rag_done"},
        {"event": "token", "text": "Hello"},
        {
            "event": "done",
            "data": {
                "query": "Hi",
                "response": "Hello",
                "thoughts": [],
                "ambient_context_used": "",
                "memory_results": [],
                "search_results": [],
                "tts": {"text": "Hello", "rate": 1.05},
                "latency_ms": 12,
                "latency": {"rag_ms": 1, "search_ms": 0, "llm_ms": 10, "ttft_ms": 5, "total_ms": 12},
                "llm_backend": "ollama",
            },
        },
    ]

    with patch.object(
        app_module.agent_engine,
        "process_tap_to_talk_stream",
        return_value=iter(fake_events),
    ):
        client = TestClient(app_module.app)
        resp = client.post("/api/tap_to_talk/stream", json={"query": "Hi"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert '"event": "token"' in body or '"event":"token"' in body
        assert '"event": "done"' in body or '"event":"done"' in body
