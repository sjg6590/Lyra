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

    # Same-speaker updates within this gap are treated as one utterance.
    COALESCE_GAP_SECONDS = 3.0
    # Open (non-final) hypotheses can finalize well after VAD drops; keep a
    # longer window so trailing words still merge into the same entry.
    OPEN_UTTERANCE_GRACE_SECONDS = 15.0
    # Finals that look cut off (ellipsis / mid-thought) get a longer merge window.
    INCOMPLETE_UTTERANCE_GRACE_SECONDS = 12.0

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
            try:
                self.episodic_store = QdrantEpisodicStore(
                    url=qdrant_url,
                    collection=qdrant_collection,
                    embedder=resolved_embedder,
                    vector_size=vector_size,
                )
                # Probe connectivity early so we can fall back before first upsert.
                self.episodic_store.count()
            except Exception as e:
                print(
                    f"[Memory] Qdrant unavailable at {qdrant_url} ({e}); "
                    "falling back to in-process Qdrant (:memory:)."
                )
                self.episodic_store = QdrantEpisodicStore(
                    location=":memory:",
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

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().split())

    @staticmethod
    def _strip_terminal_punct(text: str) -> str:
        return text.rstrip(" .!?…,;:-—").strip()

    @classmethod
    def _looks_incomplete(cls, text: str) -> bool:
        """True when ASR likely cut mid-thought (ellipsis / open dash)."""
        t = cls._normalize_text(text)
        if not t:
            return False
        if t.endswith(("...", "…", "-", "—", ",")):
            return True
        # Trailing ellipsis variants mixed with spaces.
        if t.rstrip().endswith(".."):
            return True
        return False

    @staticmethod
    def _is_related_hypothesis(prev: str, new: str) -> bool:
        """True when new text is a growing/shrinking ASR hypothesis of prev."""
        if not prev or not new:
            return False
        if prev == new:
            return True
        prev_core = RollingMemoryEngine._strip_terminal_punct(prev).lower()
        new_core = RollingMemoryEngine._strip_terminal_punct(new).lower()
        if prev_core and new_core and (
            new_core.startswith(prev_core) or prev_core.startswith(new_core)
        ):
            return True
        return new.startswith(prev) or prev.startswith(new)

    @classmethod
    def _merge_continuation(cls, prev: str, new: str) -> str | None:
        """
        Merge a pause-split continuation into the previous incomplete utterance.
        Returns merged text, or None when texts should stay separate.
        """
        if not prev or not new:
            return None
        if cls._is_related_hypothesis(prev, new):
            # Prefer the longer/growing hypothesis (already related).
            prev_core = cls._strip_terminal_punct(prev)
            new_core = cls._strip_terminal_punct(new)
            return new if len(new_core) >= len(prev_core) else prev

        if not cls._looks_incomplete(prev):
            return None

        prev_core = cls._strip_terminal_punct(prev)
        new_clean = cls._normalize_text(new)
        if not prev_core or not new_clean:
            return None

        # Overlap on shared trailing/leading words (common after pause split).
        prev_words = prev_core.split()
        new_words = new_clean.split()
        max_overlap = min(len(prev_words), len(new_words), 6)
        overlap = 0
        for n in range(max_overlap, 0, -1):
            if [w.lower() for w in prev_words[-n:]] == [w.lower() for w in new_words[:n]]:
                overlap = n
                break
        if overlap:
            merged_words = prev_words + new_words[overlap:]
            return " ".join(merged_words)

        # Incomplete prev + short continuation fragment: append.
        if len(new_words) <= 12:
            return f"{prev_core} {new_clean}".strip()
        return None

    def add_transcript(
        self,
        speaker: str,
        text: str,
        confidence: float = 1.0,
        is_user: bool = True,
        is_final: bool = True,
    ) -> dict[str, Any] | None:
        """
        Store a transcribed speech segment in the rolling buffer and episodic store.

        Coalesces interim ASR updates for the same speaker into one utterance so
        partial hypotheses do not flood the rolling context window.
        Returns None when the update is an exact duplicate of the last entry.
        """
        cleaned = self._normalize_text(text)
        if not cleaned:
            return None

        now = time.time()
        readable_time = time.strftime("%H:%M:%S", time.localtime(now))

        last = self.rolling_buffer[-1] if self.rolling_buffer else None
        if last is not None:
            last_text = self._normalize_text(str(last.get("text", "")))
            age = now - float(last.get("timestamp", 0.0))
            last_open = not bool(last.get("is_final", True))
            last_incomplete = self._looks_incomplete(last_text)
            if last_open:
                gap_limit = self.OPEN_UTTERANCE_GRACE_SECONDS
            elif last_incomplete:
                gap_limit = self.INCOMPLETE_UTTERANCE_GRACE_SECONDS
            else:
                gap_limit = self.COALESCE_GAP_SECONDS
            within_gap = age <= gap_limit
            same_speaker = last.get("speaker") == speaker
            related = self._is_related_hypothesis(last_text, cleaned)
            continuation = self._merge_continuation(last_text, cleaned)

            # Coalesce same-speaker updates, and also related/open ASR hypotheses
            # across speaker flips so onset External → later User becomes one row.
            can_coalesce = within_gap and (
                same_speaker or last_open or related or continuation is not None
            )

            if can_coalesce:
                # Exact duplicate (common when sticky transcripts are re-sent).
                if last_text == cleaned:
                    if is_final and last_open:
                        last["is_final"] = True
                        last["timestamp"] = now
                        last["readable_time"] = readable_time
                        last["confidence"] = confidence
                        last["speaker"] = speaker
                        last["is_user"] = is_user
                        self._upsert_episodic(last)
                        last["_episodic_written"] = True
                        return last
                    # Same speaker duplicate interim: idempotent. Cross-speaker
                    # duplicate still adopts the latest attribution in place.
                    if same_speaker:
                        return last if is_final else None
                    last["speaker"] = speaker
                    last["is_user"] = is_user
                    last["confidence"] = confidence
                    last["timestamp"] = now
                    last["readable_time"] = readable_time
                    if is_final:
                        last["is_final"] = True
                        self._upsert_episodic(last)
                        last["_episodic_written"] = True
                    elif last.get("_episodic_written"):
                        self._upsert_episodic(last)
                    return last

                # Replace open / related hypothesis, or stitch pause-split continuations.
                # Cross-speaker onset flips coalesce when the last row is still open
                # or the texts are related ASR hypotheses of each other.
                should_coalesce = (
                    last_open
                    or related
                    or (same_speaker and continuation is not None)
                    or (last_incomplete and continuation is not None)
                )
                if should_coalesce:
                    merged = continuation if continuation is not None else cleaned
                    if related and not last_incomplete:
                        merged = cleaned
                    last["text"] = merged
                    last["timestamp"] = now
                    last["readable_time"] = readable_time
                    last["confidence"] = confidence
                    last["speaker"] = speaker
                    last["is_user"] = is_user
                    # Keep incomplete finals open-ish for further trailing words:
                    # mark final only when the new text looks complete or caller says final
                    # and we are not still incomplete.
                    if is_final and not self._looks_incomplete(merged):
                        last["is_final"] = True
                    elif not is_final:
                        last["is_final"] = False
                    else:
                        # Final but still incomplete (ellipsis) — keep mergeable.
                        last["is_final"] = True
                    # Keep interim hypotheses out of episodic RAG until finalized;
                    # still upsert when already present / becoming final so one point updates.
                    if last["is_final"] or last.get("_episodic_written"):
                        self._upsert_episodic(last)
                        last["_episodic_written"] = True
                    self._prune_rolling_buffer(now)
                    return last

        entry = {
            "id": f"utt_{int(now * 1000)}_{uuid.uuid4().hex[:8]}",
            "timestamp": now,
            "readable_time": readable_time,
            "speaker": speaker,
            "is_user": is_user,
            "text": cleaned,
            "confidence": confidence,
            "is_final": bool(is_final),
        }

        self.rolling_buffer.append(entry)
        self._prune_rolling_buffer(now)

        # Only persist finalized utterances into episodic memory by default.
        if entry["is_final"]:
            self._upsert_episodic(entry)
            entry["_episodic_written"] = True

        return entry

    def _upsert_episodic(self, entry: dict[str, Any]) -> None:
        try:
            self.episodic_store.upsert_entry(entry)
            self.episodic_store.prune_to_max(self.max_episodic_entries)
        except Exception as e:
            print(f"[Memory] Qdrant upsert failed: {e}")

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
