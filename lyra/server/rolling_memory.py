import time
import uuid
from collections import deque
from typing import Any

from lyra.server.qdrant_memory import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    EMBEDDING_DIM,
    Embedder,
    QdrantEpisodicStore,
)


class RollingMemoryEngine:
    """
    Manages continuous ambient transcripts with:
    1. Short-term Rolling Buffer (last N minutes of speech)
    2. Long-term Episodic RAG Memory in Qdrant (EmbeddingGemma + BM25 hybrid)
    """

    def __init__(
        self,
        max_buffer_minutes: int = 30,
        max_episodic_entries: int = 1000,
        *,
        qdrant_url: str = "http://localhost:6333",
        qdrant_collection: str = "lyra_episodic",
        embedding_model: str = DEFAULT_DENSE_MODEL,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
        vector_size: int = EMBEDDING_DIM,
        episodic_store: QdrantEpisodicStore | None = None,
        embedder: Embedder | None = None,
    ):
        self.max_buffer_seconds = max_buffer_minutes * 60
        self.max_episodic_entries = max_episodic_entries
        self.rolling_buffer: deque = deque()

        if episodic_store is not None:
            self.episodic_store = episodic_store
        else:
            from lyra.server.qdrant_memory import FastEmbedHybridEmbedder

            resolved_embedder = embedder or FastEmbedHybridEmbedder(
                dense_model=embedding_model,
                sparse_model=sparse_model,
                dense_dim=vector_size,
            )
            self.episodic_store = QdrantEpisodicStore(
                url=qdrant_url,
                collection=qdrant_collection,
                embedder=resolved_embedder,
                vector_size=vector_size,
            )

    @property
    def episodic_memory(self) -> list[dict[str, Any]]:
        """Episodic entries from Qdrant (API/UI compatibility)."""
        try:
            return self.episodic_store.scroll_all(limit=max(self.max_episodic_entries * 2, 1000))
        except Exception as e:
            print(f"[Memory] Failed to scroll episodic memory: {e}")
            return []

    def add_transcript(
        self,
        speaker: str,
        text: str,
        confidence: float = 1.0,
        is_user: bool = True,
    ) -> dict[str, Any]:
        """
        Appends a new transcribed speech segment into the rolling buffer and episodic store.
        """
        now = time.time()
        readable_time = time.strftime("%H:%M:%S", time.localtime(now))

        entry = {
            "id": f"utt_{int(now * 1000)}_{uuid.uuid4().hex[:8]}",
            "timestamp": now,
            "readable_time": readable_time,
            "speaker": speaker,
            "is_user": is_user,
            "text": text.strip(),
            "confidence": confidence,
        }

        self.rolling_buffer.append(entry)
        self._prune_rolling_buffer(now)

        try:
            self.episodic_store.upsert_entry(entry)
            self.episodic_store.prune_to_max(self.max_episodic_entries)
        except Exception as e:
            print(f"[Memory] Qdrant upsert failed: {e}")

        return entry

    def _prune_rolling_buffer(self, current_time: float):
        """Removes entries older than max_buffer_seconds."""
        cutoff = current_time - self.max_buffer_seconds
        while self.rolling_buffer and self.rolling_buffer[0]["timestamp"] < cutoff:
            self.rolling_buffer.popleft()

    def get_recent_context(self, max_entries: int = 15, max_minutes: int = 15) -> list[dict[str, Any]]:
        """
        Returns recent transcript entries formatted for LLM system prompt context injection.
        """
        now = time.time()
        cutoff = now - (max_minutes * 60)
        recent = [entry for entry in self.rolling_buffer if entry["timestamp"] >= cutoff]
        return recent[-max_entries:]

    def format_context_for_prompt(self, max_entries: int = 15) -> str:
        """
        Formats recent rolling transcript as a readable string block for LLM prompts.
        """
        recent = self.get_recent_context(max_entries=max_entries)
        if not recent:
            return "(No recent ambient background conversation recorded)"

        lines = []
        for item in recent:
            time_str = item["readable_time"]
            speaker = item["speaker"]
            text = item["text"]
            lines.append(f'[{time_str}] {speaker}: "{text}"')

        return "\n".join(lines)

    def search_memory(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """
        Hybrid RAG search over episodic conversation memory (EmbeddingGemma + BM25 via Qdrant).
        """
        if not query.strip():
            return []
        try:
            return self.episodic_store.search_hybrid(query, top_k=top_k)
        except Exception as e:
            print(f"[Memory] Qdrant search error: {e}")
            return []

    def clear_memory(self):
        """Clears all rolling and episodic memories."""
        self.rolling_buffer.clear()
        try:
            self.episodic_store.clear()
        except Exception as e:
            print(f"[Memory] Qdrant clear failed: {e}")

    def episodic_count(self) -> int:
        try:
            return self.episodic_store.count()
        except Exception:
            return 0

    def backend_status(self) -> dict[str, Any]:
        health = self.episodic_store.health()
        return {
            "episodic_backend": "qdrant",
            "qdrant": health,
        }
