import json
import re
import time
from collections.abc import Iterator
from typing import Any

from lyra.server.ollama_client import OllamaClient, strip_think_blocks
from lyra.server.rolling_memory import RollingMemoryEngine
from lyra.server.search import WebSearchEngine
from lyra.server.tts import TextToSpeechEngine

_EPISODIC_SNIPPET_CHARS = 180


class LyraAgentEngine:
    """
    Jarvis-style Personal Assistant Core Agent Engine.
    Handles context synthesis, proactive RAG retrieval, tool execution (Web Search),
    and fast response generation via local Ollama (with heuristic fallback).
    """

    def __init__(
        self,
        name: str = "Lyra",
        persona: str | None = None,
        search_engine: WebSearchEngine | None = None,
        ollama_client: OllamaClient | None = None,
        ollama_config: dict[str, Any] | None = None,
        web_search_enabled: bool = False,
        context_window_turns: int = 8,
    ):
        self.name = name
        self.persona = persona or (
            "Jarvis-style intelligent, ambient personal assistant. "
            "Concise, sharp, proactive, and contextually aware."
        )
        self.search_engine = search_engine or WebSearchEngine()
        self.tts_engine = TextToSpeechEngine()
        self.web_search_enabled = bool(web_search_enabled)
        self.context_window_turns = max(1, int(context_window_turns))
        if ollama_client is not None:
            self.ollama = ollama_client
        else:
            self.ollama = OllamaClient.from_config(ollama_config)

    def process_tap_to_talk(
        self,
        query: str,
        memory_engine: RollingMemoryEngine,
        force_search: bool = False,
    ) -> dict[str, Any]:
        """
        Executes the Tap-to-Talk context injection pipeline when triggered by earbud button or user prompt.
        """
        start_time = time.time()
        thoughts: list[str] = []
        latency: dict[str, int | None] = {
            "rag_ms": None,
            "search_ms": None,
            "llm_ms": None,
            "ttft_ms": None,
            "total_ms": None,
        }

        recent_transcript_str, memory_results, search_results = self._gather_context(
            query=query,
            memory_engine=memory_engine,
            force_search=force_search,
            thoughts=thoughts,
            latency=latency,
        )

        response_text, used_ollama = self._synthesize_response(
            query=query,
            recent_transcript=recent_transcript_str,
            memory_results=memory_results,
            search_results=search_results,
            thoughts=thoughts,
            latency=latency,
        )

        tts_payload = self.tts_engine.synthesize(response_text)
        latency["total_ms"] = int((time.time() - start_time) * 1000)
        backend = "ollama" if used_ollama else "heuristic"
        thoughts.append(f"Response synthesized via {backend} in {latency['total_ms']}ms.")

        return {
            "query": query,
            "response": response_text,
            "thoughts": thoughts,
            "ambient_context_used": recent_transcript_str,
            "memory_results": memory_results,
            "search_results": search_results,
            "tts": tts_payload,
            "latency_ms": latency["total_ms"],
            "latency": latency,
            "llm_backend": backend,
        }

    def process_tap_to_talk_stream(
        self,
        query: str,
        memory_engine: RollingMemoryEngine,
        force_search: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """
        Streaming tap-to-talk pipeline.
        Yields event dicts: status | token | done | error.
        """
        start_time = time.time()
        thoughts: list[str] = []
        latency: dict[str, int | None] = {
            "rag_ms": None,
            "search_ms": None,
            "llm_ms": None,
            "ttft_ms": None,
            "total_ms": None,
        }

        try:
            yield {"event": "status", "stage": "gathering_context"}
            recent_transcript_str, memory_results, search_results = self._gather_context(
                query=query,
                memory_engine=memory_engine,
                force_search=force_search,
                thoughts=thoughts,
                latency=latency,
            )
            yield {
                "event": "status",
                "stage": "rag_done",
                "rag_ms": latency["rag_ms"],
                "search_ms": latency["search_ms"],
            }

            system_prompt = self.build_system_prompt(
                recent_transcript=recent_transcript_str,
                memory_results=memory_results,
                search_results=search_results,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            used_ollama = False
            response_text = ""
            llm_start = time.time()

            if self.ollama is not None:
                try:
                    thoughts.append(f"Streaming Ollama model '{self.ollama.model}'.")
                    yield {"event": "status", "stage": "llm_start"}
                    raw_chunks: list[str] = []
                    for chunk in self.ollama.chat_stream(messages):
                        if latency["ttft_ms"] is None:
                            latency["ttft_ms"] = int((time.time() - start_time) * 1000)
                        raw_chunks.append(chunk)
                        yield {"event": "token", "text": chunk}
                    response_text = strip_think_blocks("".join(raw_chunks))
                    if not response_text:
                        raise RuntimeError("Ollama stream returned an empty response")
                    used_ollama = True
                    latency["llm_ms"] = int((time.time() - llm_start) * 1000)
                except Exception as e:
                    thoughts.append(f"Ollama stream unavailable ({e}); falling back to heuristic synthesizer.")
                    response_text = self._heuristic_synthesize(
                        query, recent_transcript_str, memory_results, search_results
                    )
                    latency["llm_ms"] = int((time.time() - llm_start) * 1000)
                    yield {"event": "token", "text": response_text}
            else:
                response_text = self._heuristic_synthesize(
                    query, recent_transcript_str, memory_results, search_results
                )
                latency["llm_ms"] = int((time.time() - llm_start) * 1000)
                if latency["ttft_ms"] is None:
                    latency["ttft_ms"] = int((time.time() - start_time) * 1000)
                yield {"event": "token", "text": response_text}

            tts_payload = self.tts_engine.synthesize(response_text)
            latency["total_ms"] = int((time.time() - start_time) * 1000)
            backend = "ollama" if used_ollama else "heuristic"
            thoughts.append(f"Response synthesized via {backend} in {latency['total_ms']}ms.")

            yield {
                "event": "done",
                "data": {
                    "query": query,
                    "response": response_text,
                    "thoughts": thoughts,
                    "ambient_context_used": recent_transcript_str,
                    "memory_results": memory_results,
                    "search_results": search_results,
                    "tts": tts_payload,
                    "latency_ms": latency["total_ms"],
                    "latency": latency,
                    "llm_backend": backend,
                },
            }
        except Exception as e:
            yield {"event": "error", "message": str(e)}

    def _gather_context(
        self,
        query: str,
        memory_engine: RollingMemoryEngine,
        force_search: bool,
        thoughts: list[str],
        latency: dict[str, int | None],
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, str]]]:
        recent_transcript_str = memory_engine.format_context_for_prompt(
            max_entries=self.context_window_turns
        )
        thoughts.append(
            f"Fetched recent ambient rolling context window ({self.context_window_turns} turns)."
        )

        rag_start = time.time()
        memory_results = memory_engine.search_memory(query, top_k=3)
        latency["rag_ms"] = int((time.time() - rag_start) * 1000)
        if memory_results:
            thoughts.append(
                f"Retrieved {len(memory_results)} relevant past conversation records "
                f"from Episodic RAG Memory ({latency['rag_ms']}ms)."
            )
        else:
            thoughts.append(f"No historical episodic memory records matched the query ({latency['rag_ms']}ms).")

        search_results: list[dict[str, str]] = []
        needs_search = force_search or (
            self.web_search_enabled and self._check_needs_search(query, recent_transcript_str)
        )
        if needs_search:
            search_start = time.time()
            search_query = self._extract_search_keywords(query, recent_transcript_str)
            thoughts.append(f"Triggered Live Web Search for: '{search_query}'")
            search_results = self.search_engine.search(search_query)
            latency["search_ms"] = int((time.time() - search_start) * 1000)
            thoughts.append(
                f"Retrieved {len(search_results)} live web search snippets ({latency['search_ms']}ms)."
            )
        else:
            latency["search_ms"] = 0
            if not self.web_search_enabled and not force_search:
                thoughts.append("Web search skipped (disabled in config).")

        return recent_transcript_str, memory_results, search_results

    def _check_needs_search(self, query: str, recent_context: str) -> bool:
        """Determines if query warrants external web search."""
        query_lower = query.lower()
        search_triggers = [
            "search", "what is", "who is", "latest", "news", "weather", "price",
            "stock", "restaurant", "meaning", "definition", "how to", "score",
            "company", "location", "address", "phone number", "specs", "current",
        ]

        if any(trigger in query_lower for trigger in search_triggers):
            return True

        if re.search(r"\b(that|the|what|who|which)\b", query_lower) and len(query.split()) > 2:
            return True

        return False

    def _extract_search_keywords(self, query: str, recent_context: str) -> str:
        """
        Derives an optimal web search query by combining user query with ambient context entities.
        """
        clean_query = re.sub(
            r"^(lyra|hey lyra|jarvis|hey jarvis|please|can you|what is|search for)\b",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip()

        entities = re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", recent_context)
        filtered_entities = [
            e for e in entities if e not in ["User", "Me", "External", "Speaker", "Lyra", "Jarvis"]
        ]

        if filtered_entities and ("that" in query.lower() or "they" in query.lower()):
            search_query = f"{clean_query} {' '.join(filtered_entities[-2:])}"
        else:
            search_query = clean_query if clean_query else query

        return search_query.strip()

    def build_system_prompt(
        self,
        recent_transcript: str,
        memory_results: list[dict[str, Any]],
        search_results: list[dict[str, str]],
    ) -> str:
        """Assemble the system prompt for Ollama from persona + ambient/RAG/search context."""
        memory_block = "(No episodic memory matches)"
        if memory_results:
            lines = []
            for item in memory_results:
                text = str(item.get("text", ""))
                if len(text) > _EPISODIC_SNIPPET_CHARS:
                    text = text[:_EPISODIC_SNIPPET_CHARS].rstrip() + "…"
                lines.append(
                    f"- [{item.get('readable_time', '?')}] {item.get('speaker', '?')}: \"{text}\""
                )
            memory_block = "\n".join(lines)

        search_block = self.search_engine.format_search_for_prompt(search_results)

        return (
            f"You are {self.name}, {self.persona}\n\n"
            "Respond as a spoken voice assistant: clear, helpful, and concise "
            "(usually 1–3 short sentences). Do not narrate your reasoning. "
            "Use the ambient transcript, episodic memory, and web search evidence when relevant. "
            "If evidence is missing, say so briefly instead of inventing facts.\n\n"
            f"## Recent ambient transcript\n{recent_transcript}\n\n"
            f"## Episodic memory matches\n{memory_block}\n\n"
            f"## Live web search\n{search_block}"
        )

    def _synthesize_response(
        self,
        query: str,
        recent_transcript: str,
        memory_results: list[dict[str, Any]],
        search_results: list[dict[str, str]],
        thoughts: list[str] | None = None,
        latency: dict[str, int | None] | None = None,
    ) -> tuple[str, bool]:
        """
        Synthesizes a Jarvis-style voice response via Ollama, with heuristic fallback.
        Returns (response_text, used_ollama).
        """
        thoughts = thoughts if thoughts is not None else []
        latency = latency if latency is not None else {}
        llm_start = time.time()

        if self.ollama is not None:
            try:
                system_prompt = self.build_system_prompt(
                    recent_transcript=recent_transcript,
                    memory_results=memory_results,
                    search_results=search_results,
                )
                thoughts.append(f"Calling Ollama model '{self.ollama.model}'.")
                text = self.ollama.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ]
                )
                latency["llm_ms"] = int((time.time() - llm_start) * 1000)
                if latency.get("ttft_ms") is None:
                    # Non-streaming: first token ≈ full completion for reporting.
                    latency["ttft_ms"] = latency["llm_ms"]
                return text, True
            except Exception as e:
                thoughts.append(f"Ollama unavailable ({e}); falling back to heuristic synthesizer.")
                latency["llm_ms"] = int((time.time() - llm_start) * 1000)

        result = self._heuristic_synthesize(query, recent_transcript, memory_results, search_results)
        latency["llm_ms"] = int((time.time() - llm_start) * 1000)
        if latency.get("ttft_ms") is None:
            latency["ttft_ms"] = latency["llm_ms"]
        return result, False

    def _heuristic_synthesize(
        self,
        query: str,
        recent_transcript: str,
        memory_results: list[dict[str, Any]],
        search_results: list[dict[str, str]],
    ) -> str:
        """Rule/keyword synthesizer used when Ollama is disabled or unreachable."""
        q_lower = query.lower()

        if (
            "restaurant" in q_lower
            or "company" in q_lower
            or "person" in q_lower
            or "mentioned" in q_lower
            or "say" in q_lower
        ):
            if memory_results:
                top_mem = memory_results[0]
                return (
                    f"Based on your recent conversation at {top_mem['readable_time']}, "
                    f"{top_mem['speaker']} mentioned: \"{top_mem['text']}\"."
                )

            if search_results:
                top_s = search_results[0]
                return f"According to live search, {top_s['title']} is described as: {top_s['snippet']}"

            if recent_transcript and "(No recent ambient" not in recent_transcript:
                return (
                    "Scanning your recent audio log: The latest context captured is: "
                    f"{recent_transcript.splitlines()[-1]}"
                )

        if search_results:
            top_s = search_results[0]
            answer = f"{top_s['title']}: {top_s['snippet']}"
            if len(search_results) > 1:
                answer += f" Additionally, {search_results[1]['title']} notes {search_results[1]['snippet']}"
            return answer[:350]

        if memory_results:
            top = memory_results[0]
            return (
                f"I recalled from your memory log at {top['readable_time']} "
                f"({top['speaker']}): \"{top['text']}\"."
            )

        return "Understood. I have logged your context. How else can I assist with this conversation?"


def format_sse(event: dict[str, Any]) -> str:
    """Serialize an agent stream event as an SSE data frame."""
    return f"data: {json.dumps(event, default=str)}\n\n"
