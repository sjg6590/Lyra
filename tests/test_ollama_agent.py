"""Unit tests for Ollama client + Lyra agent synthesis (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from lyra.server.agent import LyraAgentEngine
from lyra.server.ollama_client import OllamaClient, strip_think_blocks
from lyra.server.rolling_memory import RollingMemoryEngine
from lyra.server.search import WebSearchEngine


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
    assert client.num_ctx == 8192


def test_ollama_chat_sends_think_false_and_options():
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
    assert payload["options"]["num_ctx"] == 8192


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

    agent = LyraAgentEngine(name="Lyra", search_engine=search, ollama_client=mock_ollama)
    memory = RollingMemoryEngine(max_buffer_minutes=30)
    memory.add_transcript(speaker="User [Me]", text="Let's meet at Blue Bottle", confidence=0.9, is_user=True)

    result = agent.process_tap_to_talk("What restaurant did I mention?", memory, force_search=False)

    assert result["response"] == "I heard them mention Blue Bottle earlier."
    assert result["llm_backend"] == "ollama"
    mock_ollama.chat.assert_called_once()
    messages = mock_ollama.chat.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert "Blue Bottle" in messages[0]["content"] or "Blue Bottle" in messages[1]["content"]
    assert "ambient transcript" in messages[0]["content"].lower() or "Ambient" in messages[0]["content"]


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
    )
    memory = RollingMemoryEngine(max_buffer_minutes=30)
    memory.add_transcript(speaker="User [Me]", text="Acme Corp is hiring", confidence=0.95, is_user=True)

    # Force memory path via keyword triggers in heuristic
    result = agent.process_tap_to_talk("What company was mentioned?", memory, force_search=False)

    assert result["llm_backend"] == "heuristic"
    assert "Acme Corp" in result["response"] or "mentioned" in result["response"].lower()
    assert any("falling back" in t.lower() for t in result["thoughts"])


def test_agent_heuristic_without_ollama_client():
    agent = LyraAgentEngine(name="Lyra", ollama_client=None, search_engine=WebSearchEngine(max_results=1))
    memory = RollingMemoryEngine(max_buffer_minutes=30)
    result = agent.process_tap_to_talk("Hello Lyra", memory)
    assert result["llm_backend"] == "heuristic"
    assert isinstance(result["response"], str) and len(result["response"]) > 0


def test_build_system_prompt_includes_search():
    search = WebSearchEngine(max_results=2)
    agent = LyraAgentEngine(name="Lyra", search_engine=search, ollama_client=None)
    prompt = agent.build_system_prompt(
        recent_transcript="[12:00:00] User [Me]: \"hello\"",
        memory_results=[{"readable_time": "12:00:00", "speaker": "User [Me]", "text": "hello"}],
        search_results=[{"title": "Example", "snippet": "A site", "url": "https://example.com"}],
    )
    assert "Lyra" in prompt
    assert "hello" in prompt
    assert "Example" in prompt
    assert "https://example.com" in prompt
