"""Thin HTTP client for a local Ollama instance."""

from __future__ import annotations

import re
from typing import Any

import requests

_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove residual <think>...</think> blocks from model output."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    return cleaned.strip()


class OllamaClient:
    """Minimal Ollama /api/chat wrapper using requests."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:9b-mlx",
        think: bool = False,
        num_ctx: int = 8192,
        temperature: float = 0.6,
        timeout_seconds: float = 90,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.think = think
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, ollama_cfg: dict[str, Any] | None) -> OllamaClient | None:
        """Build a client from config.json agent.ollama block, or None if disabled/missing."""
        if not ollama_cfg or not ollama_cfg.get("enabled", True):
            return None
        return cls(
            host=ollama_cfg.get("host", "http://127.0.0.1:11434"),
            model=ollama_cfg.get("model", "qwen3.5:9b-mlx"),
            think=bool(ollama_cfg.get("think", False)),
            num_ctx=int(ollama_cfg.get("num_ctx", 8192)),
            temperature=float(ollama_cfg.get("temperature", 0.6)),
            timeout_seconds=float(ollama_cfg.get("timeout_seconds", 90)),
        )

    def is_reachable(self) -> bool:
        """Return True if Ollama responds to /api/tags."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """Return installed model names from /api/tags."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            names: list[str] = []
            for item in models:
                name = item.get("name") or item.get("model")
                if name:
                    names.append(name)
            return names
        except (requests.RequestException, ValueError, KeyError):
            return []

    def has_model(self, model: str | None = None) -> bool:
        """Check whether the configured (or given) model is installed."""
        target = model or self.model
        for name in self.list_models():
            if name == target or name.startswith(f"{target}:") or target.startswith(f"{name}:"):
                return True
        return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Call POST /api/chat (non-streaming) and return cleaned assistant text.
        Raises requests.RequestException or RuntimeError on failure.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        }
        resp = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message") or {}
        content = message.get("content") or data.get("response") or ""
        if not isinstance(content, str):
            content = str(content)
        cleaned = strip_think_blocks(content)
        if not cleaned:
            raise RuntimeError("Ollama returned an empty response")
        return cleaned
