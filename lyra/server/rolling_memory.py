import re
import time
from collections import deque
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RollingMemoryEngine:
    """
    Manages continuous ambient transcripts with:
    1. Short-term Rolling Buffer (last N minutes of speech)
    2. Long-term Episodic RAG Memory Store with Vector & Keyword Search
    """

    def __init__(self, max_buffer_minutes: int = 30, max_episodic_entries: int = 1000):
        self.max_buffer_seconds = max_buffer_minutes * 60
        self.max_episodic_entries = max_episodic_entries
        self.rolling_buffer: deque = deque()
        self.episodic_memory: list[dict[str, Any]] = []

        # Vectorizer for Episodic Memory RAG
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._is_vectorizer_fitted = False
        self._tfidf_matrix = None

    def add_transcript(self, speaker: str, text: str, confidence: float = 1.0, is_user: bool = True) -> dict[str, Any]:
        """
        Appends a new transcribed speech segment into the rolling buffer and episodic store.
        """
        now = time.time()
        readable_time = time.strftime("%H:%M:%S", time.localtime(now))

        entry = {
            "id": f"utt_{int(now * 1000)}",
            "timestamp": now,
            "readable_time": readable_time,
            "speaker": speaker,
            "is_user": is_user,
            "text": text.strip(),
            "confidence": confidence
        }

        # 1. Add to rolling buffer
        self.rolling_buffer.append(entry)
        self._prune_rolling_buffer(now)

        # 2. Add to episodic memory
        self.episodic_memory.append(entry)
        if len(self.episodic_memory) > self.max_episodic_entries:
            self.episodic_memory.pop(0)

        # Update RAG vector index
        self._update_vector_index()

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
            lines.append(f"[{time_str}] {speaker}: \"{text}\"")

        return "\n".join(lines)

    def _update_vector_index(self):
        """Re-fits TF-IDF vector index over episodic memory texts."""
        if not self.episodic_memory:
            return

        texts = [entry["text"] for entry in self.episodic_memory]
        try:
            self._tfidf_matrix = self.vectorizer.fit_transform(texts)
            self._is_vectorizer_fitted = True
        except Exception:
            self._is_vectorizer_fitted = False

    def search_memory(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """
        RAG Search over episodic conversation memory using hybrid vector TF-IDF cosine similarity & keyphrase matching.
        """
        if not self.episodic_memory or not query.strip():
            return []

        results = []

        # Vector RAG search if fitted
        if self._is_vectorizer_fitted and self._tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
                top_indices = np.argsort(similarities)[::-1][:top_k * 2]

                for idx in top_indices:
                    score = float(similarities[idx])
                    if score > 0.05:
                        entry = dict(self.episodic_memory[idx])
                        entry["relevance_score"] = round(score, 3)
                        results.append(entry)
            except Exception as e:
                print(f"[Memory] Vector search error: {e}")

        # Fallback / Keyword boost search if vector search returned few items
        if len(results) < top_k:
            keywords = re.findall(r'\w+', query.lower())
            for entry in reversed(self.episodic_memory):
                text_lower = entry["text"].lower()
                matches = sum(1 for kw in keywords if kw in text_lower)
                if matches > 0 and not any(r["id"] == entry["id"] for r in results):
                    item = dict(entry)
                    item["relevance_score"] = round(matches / (len(keywords) + 1), 3)
                    results.append(item)
                    if len(results) >= top_k:
                        break

        return results[:top_k]

    def clear_memory(self):
        """Clears all rolling and episodic memories."""
        self.rolling_buffer.clear()
        self.episodic_memory.clear()
        self._is_vectorizer_fitted = False
        self._tfidf_matrix = None
