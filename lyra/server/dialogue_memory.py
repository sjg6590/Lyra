"""In-process short-term dialogue session store for multi-turn tap-to-talk."""

from __future__ import annotations

import time
import uuid
from typing import Any


class DialogueSessionStore:
    """
    Server-side chat history keyed by session_id.

    Stores user/assistant message pairs for immediate follow-ups.
    Caps history length and drops idle sessions to bound memory use.
    """

    def __init__(
        self,
        max_history_turns: int = 6,
        *,
        max_sessions: int = 100,
        session_ttl_seconds: float = 6 * 60 * 60,
    ):
        self.max_history_turns = max(1, int(max_history_turns))
        self.max_messages = self.max_history_turns * 2
        self.max_sessions = max(1, int(max_sessions))
        self.session_ttl_seconds = max(60.0, float(session_ttl_seconds))
        # session_id -> {"messages": [...], "updated_at": float}
        self._sessions: dict[str, dict[str, Any]] = {}

    def get_or_create(self, session_id: str | None = None) -> str:
        """Return an existing session id or create a new one."""
        self._prune_expired()
        if session_id and session_id in self._sessions:
            self._touch(session_id)
            return session_id
        if session_id and session_id.strip():
            sid = session_id.strip()
            self._sessions[sid] = {"messages": [], "updated_at": time.time()}
            self._enforce_session_cap()
            return sid
        sid = f"sess_{uuid.uuid4().hex}"
        self._sessions[sid] = {"messages": [], "updated_at": time.time()}
        self._enforce_session_cap()
        return sid

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        """Return a copy of prior user/assistant messages for the session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        self._touch(session_id)
        return [dict(m) for m in session["messages"]]

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Append one user/assistant pair and trim to max_history_turns."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {"messages": [], "updated_at": time.time()}
        messages: list[dict[str, str]] = self._sessions[session_id]["messages"]
        user = (user_text or "").strip()
        assistant = (assistant_text or "").strip()
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        if len(messages) > self.max_messages:
            # Drop oldest complete pairs (keep even length when possible).
            overflow = len(messages) - self.max_messages
            del messages[:overflow]
            if messages and messages[0].get("role") == "assistant":
                messages.pop(0)
        self._touch(session_id)
        self._enforce_session_cap()

    def clear(self, session_id: str) -> None:
        """Remove one session."""
        self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """Remove all sessions."""
        self._sessions.clear()

    def _touch(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session["updated_at"] = time.time()

    def _prune_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if now - float(session.get("updated_at", 0.0)) > self.session_ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

    def _enforce_session_cap(self) -> None:
        if len(self._sessions) <= self.max_sessions:
            return
        ordered = sorted(
            self._sessions.items(),
            key=lambda item: float(item[1].get("updated_at", 0.0)),
        )
        to_drop = len(self._sessions) - self.max_sessions
        for sid, _ in ordered[:to_drop]:
            del self._sessions[sid]
